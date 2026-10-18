from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# from torch.func import rearrange

@dataclass
class RTDetrMultiScaleOutput:
    ''' Just for cleaner return '''
    logits: Optional[torch.Tensor] = None
    pred_boxes: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[List[torch.Tensor]] = None
    future_boxes: Optional[torch.Tensor] = None      # (B, N, T, 4)
    present_refined: Optional[torch.Tensor] = None   # (B, N, 4)
    # supervisione RT-DETR del ramo standard
    enc_topk_logits: Optional[torch.Tensor] = None      # (B, S, num_labels), proposte encoder ai topk
    enc_topk_bboxes: Optional[torch.Tensor] = None      # (B, S, 4)
    intermediate_logits: Optional[torch.Tensor] = None  # (B, L, Q, num_labels), ogni layer
    intermediate_boxes: Optional[torch.Tensor] = None   # (B, L, Q, 4), ogni layer
    hidden_states: Optional[torch.Tensor] = None        # (B, Q, D)

class LogFourierScaleEncoding(nn.Module):
    """Log-Fourier temporal encoding for duration information."""

    def __init__(self, dim: int, num_bands: int = 16):
        super().__init__()
        self.num_bands = num_bands
        self.dim = dim
        self.freqs = 2 ** torch.linspace(0, num_bands - 1, num_bands)
        self.proj = nn.Linear(2 * num_bands, dim)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dt: (B, N) accumulation lengths in milliseconds
        Returns:
            (B, N, D) temporal embeddings
        """
        dt = torch.log(dt + 1e-6).unsqueeze(-1)        # (B, N, 1)
        freqs = self.freqs.to(dt.device)
        x = dt * freqs                                   # (B, N, num_bands)
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.proj(pe)                             # (B, N, D)

class PastEncoder(nn.Module):

    def __init__(self, num_past_steps: int, d_model: int):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_past_steps*4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
    def forward(self, past_boxes: torch.Tensor) -> torch.Tensor:
        # x = rearrange(past_boxes, "b n p c -> b n (p c)")
        x = past_boxes.flatten(2)
        x =self.mlp(x)
        return x

def _cv_anchor(present_boxes: torch.Tensor, past_velocity: Optional[torch.Tensor], T: int) -> torch.Tensor:
    """anchor_t = present + (t+1) * velocity, oppure present se velocity è None."""
    anchor = present_boxes.unsqueeze(2)                       # (B, Q, 1, 4)
    if past_velocity is not None:
        steps = torch.arange(1, T + 1, device=present_boxes.device,
                             dtype=present_boxes.dtype).view(1, 1, T, 1)
        return anchor + steps * past_velocity.unsqueeze(2)    # (B, Q, T, 4)
    return anchor.expand(-1, -1, T, -1)


class ForecastingHead(nn.Module):
    """
    Forecasting head a transformer: un token presente per drone (raffina la box) e un token per
    step futuro, in cross-attention sulle feature encoder. future_time_deltas è ignorato.
    """
    def __init__(
        self,
        d_model: int,
        num_future_steps: int,
        nhead: int = 8,
        num_layers: int = 3,
        pool_size: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_bands: int = 16,
    ):
        super().__init__()
        self.num_future_steps = num_future_steps
        self.d_model = d_model
        self.pool_size = pool_size

        # embedding degli step futuri e del token presente
        self.step_embed = nn.Parameter(torch.randn(1, num_future_steps, d_model) * 0.02)
        self.present_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # memoria: feature encoder pooled
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.mem_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        # delta delle box future
        self.bbox = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 4),
        )
        # delta del presente, ultimo layer a zero (a init present_refined == pred_boxes)
        self.present_bbox = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 4),
        )
        nn.init.zeros_(self.present_bbox[-1].weight)
        nn.init.zeros_(self.present_bbox[-1].bias)

    def _memory(self, fused_features: List[torch.Tensor]) -> torch.Tensor:
        """[(B, D, H_s, W_s)] -> (B, S*P*P, D)"""
        parts = []
        for f in fused_features:
            p = self.pool(f).flatten(2).transpose(1, 2)                 # (B, P*P, D)
            parts.append(p)
        mem = torch.cat(parts, dim=1)
        return self.mem_proj(mem)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # (B, N, D) embedding per query-passato
        present_boxes: torch.Tensor,                 # (B, N, 4) box detected (grezze)
        fused_features: List[torch.Tensor],          # feature encoder fuse (memoria)
        future_time_deltas: Optional[torch.Tensor] = None,   # ignorato
        past_velocity: Optional[torch.Tensor] = None,        # (B, N, 4) velocità/step (ancora CV)
        use_present_refine: bool = True,             # False = presente grezzo
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ritorna (present_refined (B,N,4), future_boxes (B,N,T,4))."""
        B, N, D = hidden_states.shape
        T = self.num_future_steps

        memory = self._memory(fused_features)                          # (B, E, D)

        future_tok = (hidden_states.unsqueeze(2).expand(B, N, T, D)
                      + self.step_embed.view(1, 1, T, D)).reshape(B, N * T, D)

        if use_present_refine:
            # token presente + token futuri
            present_tok = hidden_states + self.present_embed           # (B, N, D)
            tgt = torch.cat([present_tok, future_tok], dim=1)          # (B, N + N*T, D)
            dec = self.decoder(tgt=tgt, memory=memory)
            present_dec = dec[:, :N]                                   # (B, N, D)
            future_dec = dec[:, N:].view(B, N, T, D)                   # (B, N, T, D)
            present_refined = (present_boxes + self.present_bbox(present_dec)).clamp(0.0, 1.0)
        else:
            # solo token futuri, presente grezzo
            future_dec = self.decoder(tgt=future_tok, memory=memory).view(B, N, T, D)
            present_refined = present_boxes

        # future = ancora CV + delta
        deltas = self.bbox(future_dec)                                 # (B, N, T, 4)
        anchor = _cv_anchor(present_refined, past_velocity, T)         # (B, N, T, 4)
        future_boxes = (anchor + deltas).clamp(0.0, 1.0)
        return present_refined, future_boxes


class ForecastingHeadMLP(nn.Module):
    """Baseline MLP con la stessa interfaccia di ForecastingHead."""
    def __init__(self, d_model: int, num_future_steps: int, hidden_dim: int = 512):
        super().__init__()
        self.num_future_steps = num_future_steps
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_future_steps * 4),
        )
        # refinement del presente, ultimo layer a zero
        self.present_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.present_mlp[-1].weight)
        nn.init.zeros_(self.present_mlp[-1].bias)

    def forward(self, hidden_states, present_boxes, fused_features=None,
                future_time_deltas=None, past_velocity=None, use_present_refine=True):
        B, N, _ = hidden_states.shape
        T = self.num_future_steps
        if use_present_refine:
            present_refined = (present_boxes + self.present_mlp(hidden_states)).clamp(0.0, 1.0)
        else:
            present_refined = present_boxes
        deltas = self.mlp(hidden_states).view(B, N, T, 4)
        anchor = _cv_anchor(present_refined, past_velocity, T)   # (B, N, T, 4)
        future_boxes = (anchor + deltas).clamp(0.0, 1.0)
        return present_refined, future_boxes


class MultiScaleAttentionFusion(nn.Module):
    """
    Attention-based fusion of multi-scale encoder features across temporal branches.

    Processes each spatial scale independently: for every scale, the features
    from all N duration branches are fused via cross-attention using a learned
    fusion token as query and temporally-encoded branch features as keys/values.
    The output is added as a residual on top of the mean across branches.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model

        self.temporal_encoding = LogFourierScaleEncoding(dim=d_model, num_bands=16)

        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
            for _ in range(num_scales)
        ])
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_scales)
        ])
        self.fusion_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(num_scales)
        ])

        for token in self.fusion_tokens:
            nn.init.normal_(token, std=0.02)

        # Zero-init so the module starts as a mean fusion (stable fine-tuning)
        for attn in self.scale_attentions:
            attn.out_proj.weight.data.zero_()
            attn.out_proj.bias.data.zero_()

    def forward(
        self,
        multi_scale_features: List[List[torch.Tensor]],
        durations: List[int],
    ) -> List[torch.Tensor]:
        """
        Args:
            multi_scale_features: [num_branches][num_scales] tensors, each (B, C, H, W)
            durations: duration in ms for each branch

        Returns:
            List of fused feature tensors per scale: [(B, C, H_s, W_s), ...]
        """
        num_branches = len(multi_scale_features)
        device = multi_scale_features[0][0].device
        fused_scales = []

        # time_embed dipende solo dalle durate: calcolato una volta per tutte le scale
        durations_t = torch.tensor(durations, device=device, dtype=torch.float32)  # (K,)
        time_embed = self.temporal_encoding(durations_t.unsqueeze(0))              # (1, K, D)

        for scale_idx in range(self.num_scales):
            scale_features = [multi_scale_features[b][scale_idx] for b in range(num_branches)]
            B, C, H, W = scale_features[0].shape
            L = H * W

            flattened = [f.flatten(2).transpose(1, 2) for f in scale_features]  # (B, L, C)
            stacked = torch.stack(flattened, dim=1)                            # (B, K, L, C)
            mean_fused = stacked.mean(dim=1)                                      # (B, L, C)

            K = num_branches
            reshaped = stacked.permute(0, 2, 1, 3).reshape(B * L, K, C)
            reshaped = self.scale_norms[scale_idx](reshaped)

            query = self.fusion_tokens[scale_idx].expand(B * L, -1, -1)          # (B*L, 1, C)
            key = reshaped + time_embed                                          # broadcast (1, K, D)
            value = reshaped

            attn_out, _ = self.scale_attentions[scale_idx](query, key, value)
            residual = attn_out.squeeze(1).reshape(B, L, C)

            fused = (mean_fused + residual).transpose(1, 2).reshape(B, C, H, W)
            fused_scales.append(fused)

        return fused_scales


class MultiDurationRTDetrAttention(nn.Module):

    def __init__(
        self,
        checkpoint: str = "PekingU/rtdetr_v2_r18vd",
        num_labels: int = 1,
        num_past_steps: int = 10,
        durations_ms: List[int] = [3, 33, 165, 330],
        use_shared_weights: bool = False,
        shared_cat: bool = False,
        phase: int = 1,
        num_standard_queries: int = 5,
        num_future_steps: int = 0,
        forecast_head_type: str = 'transformer',   # 'transformer' | 'mlp'
        vel_avg_k: int = 3,   # step su cui mediare la velocità per l'ancora CV
        use_cv_anchor: bool = True,   # False: future = present + delta
        use_present_refine: bool = True,   # False: presente grezzo
        block_diag_decoder_attn: bool = False,   # True: niente attenzione past <-> std nel decoder
        use_past_class_head: bool = False,   # True: class_embed separata per il ramo passato
    ):
        super().__init__()

        self.durations_ms = sorted(durations_ms)
        self.checkpoint = checkpoint
        self.num_scales = 3
        self.use_shared_weights = use_shared_weights
        self.shared_cat = shared_cat and use_shared_weights

        CLASS_TO_ID = {'drone': 0}
        ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

        if use_shared_weights:
            self.shared_branch = RTDetrV2ForObjectDetection.from_pretrained(
                checkpoint, num_labels=num_labels,
                id2label=ID_TO_CLASS, label2id=CLASS_TO_ID,
                ignore_mismatched_sizes=True, num_queries=50,
            )
            ref_branch = self.shared_branch
        else:
            self.branches = nn.ModuleDict()
            for d in self.durations_ms:
                self.branches[f'branch_{d}ms'] = RTDetrV2ForObjectDetection.from_pretrained(
                    checkpoint, num_labels=num_labels,
                    id2label=ID_TO_CLASS, label2id=CLASS_TO_ID,
                    ignore_mismatched_sizes=True, num_queries=50,
                )
            ref_branch = self.branches[f'branch_{self.durations_ms[0]}ms']

        self.config = ref_branch.config
        
        D = self.config.d_model  # 256
        P = num_past_steps
        self.past_encoder = PastEncoder(num_past_steps=P, d_model=D)

        self.fusion = MultiScaleAttentionFusion(d_model=D, num_heads=8, num_scales=self.num_scales)
        self.phase = phase

        self.num_standard_queries = num_standard_queries

        # forecasting head solo se num_future_steps > 0
        self.num_future_steps = num_future_steps
        self.vel_avg_k = vel_avg_k
        self.use_cv_anchor = use_cv_anchor
        self.use_present_refine = use_present_refine
        self.block_diag_decoder_attn = block_diag_decoder_attn
        # class_embed separata per le past-query, inizializzata da quella standard
        self.use_past_class_head = use_past_class_head
        if use_past_class_head:
            import copy
            self.past_class_embed = copy.deepcopy(ref_branch.class_embed[-1])
            print("[past-head] class_embed DEDICATO al ramo passato (init = copia della testa standard)")
        else:
            self.past_class_embed = None
        if num_future_steps > 0:
            if forecast_head_type == 'mlp':
                self.forecasting_head = ForecastingHeadMLP(d_model=D, num_future_steps=num_future_steps)
            else:
                self.forecasting_head = ForecastingHead(d_model=D, num_future_steps=num_future_steps)
            print(f"[forecasting] head = {forecast_head_type} | T = {num_future_steps} | "
                  f"cv_anchor = {'ON (present + t·vel)' if use_cv_anchor else 'OFF (present + delta)'} | "
                  f"present_refine = {'ON (Proposta C)' if use_present_refine else 'OFF (presente grezzo)'}")
        else:
            self.forecasting_head = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def primary_branch(self):
        """Branch che fornisce .loss_function e .config per la loss del ramo standard."""
        return self.shared_branch if self.use_shared_weights else self.branches[f'branch_{self.durations_ms[0]}ms']

    def _run_backbone_and_encoder(
        self,
        branch: RTDetrV2ForObjectDetection,
        pixel_values: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run ResNet backbone + hybrid encoder, return 3 multi-scale feature maps."""
        model = branch.model
        B, _, H, W = pixel_values.shape
        device = pixel_values.device

        pixel_mask = torch.ones((B, H, W), device=device, dtype=torch.long)
        backbone_outputs = model.backbone(pixel_values, pixel_mask)

        projected = [
            model.encoder_input_proj[lvl](feat)
            for lvl, (feat, _) in enumerate(backbone_outputs)
        ]
        return model.encoder(projected).last_hidden_state  # List[(B, D, H_s, W_s)]

    def _run_decoder_with_fused_features(
        self,
        branch: RTDetrV2ForObjectDetection,
        fused_encoder_features: List[torch.Tensor],
        labels: Optional[List[Dict]] = None,
        past_boxes: Optional[torch.Tensor] = None,
        past_mask: Optional[torch.Tensor] = None,
        past_ref_points: Optional[torch.Tensor] = None,
        query_mode: str = 'both',
        future_time_deltas: Optional[torch.Tensor] = None,
        past_velocity: Optional[torch.Tensor] = None,
    ) -> RTDetrMultiScaleOutput:

        model= branch.model
        device = fused_encoder_features[0].device
        dtype = fused_encoder_features[0].dtype
        batch_size = fused_encoder_features[0].shape[0]

        # 1. Project fused features through decoder input projections
        sources = [model.decoder_input_proj[lvl](src) for lvl, src in enumerate(fused_encoder_features)]

        # 2. Flatten spatial dimensions and build level metadata
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)

        for lvl, src in enumerate(sources):
            _, _, h, w = src.shape
            spatial_shapes[lvl, 0] = h
            spatial_shapes[lvl, 1] = w
            spatial_shapes_list.append((h, w)) # per deformable attention
            source_flatten.append(src.flatten(2).transpose(1, 2))  # (B, H*W, C)

        source_flatten = torch.cat(source_flatten, dim=1)        # (B, total_len, C)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])

        # 3. Generate anchors ovvero il punto di partenza per le proposte dell encoder
        anchors, valid_mask = model.generate_anchors(
            tuple(spatial_shapes_list), device=device, dtype=dtype
        )

        # 4. Encoder proposals → top-k query initialisation
        use_standard = (self.phase == 2 and query_mode in ('both', 'standard_only'))

        # proposte encoder ai topk, per branch.loss_function
        enc_topk_logits = None
        enc_topk_bboxes = None

        if use_standard: # solo qua calcolo le proposte dell encoder e prendo le topk
            memory = valid_mask.to(dtype) * source_flatten
            output_memory = model.enc_output(memory)
            enc_outputs_class = model.enc_score_head(output_memory)
            enc_outputs_coord_logits = model.enc_bbox_head(output_memory) + anchors  # logit space

            # prendo solo 50 proposte da usare come standard queries assieme alle past boxes queries
            _, topk_ind = torch.topk(
                enc_outputs_class.max(-1).values, 
                self.num_standard_queries, 
                dim=1
            )
        
            # creo i reference points per le standard queries
            std_ref = enc_outputs_coord_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
            )

            # score/box delle proposte ai topk (supervisionano enc_score_head)
            enc_topk_logits = enc_outputs_class.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
            )
            enc_topk_bboxes = torch.sigmoid(std_ref)

            if self.config.learn_initial_query:
                std_q = model.weight_embedding.tile([batch_size, 1, 1])[:, :self.num_standard_queries, :]
            else:
                std_q = output_memory.gather(
                    dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
                ).detach()


        past_ref_points_unact = torch.logit(past_ref_points.clamp(1e-4, 1-1e-4))

        # Quali query dare al decoder
        if self.phase == 1 or query_mode == 'past_only': 
            target = past_boxes # solo query del passato
            ref_points = past_ref_points_unact
        elif query_mode == 'standard_only':
            target = std_q # solo query standard
            ref_points = std_ref
        else:  
            target = torch.cat([past_boxes, std_q], dim=1)
            ref_points = torch.cat([past_ref_points_unact, std_ref], dim=1)


        # maschero le query passate di padding (e past <-> std con block_diag_decoder_attn)
        encoder_attention_mask = None
        block_diag = bool(getattr(self, 'block_diag_decoder_attn', False)) and query_mode == 'both' and self.phase == 2
        if (query_mode != 'standard_only' and past_mask is not None) or block_diag:
            if query_mode == 'both' and self.phase == 2:
                std_pad = torch.zeros(batch_size, self.num_standard_queries, dtype=torch.bool, device=device)   # tutte False
                if past_mask is not None:
                    key_padding = torch.cat([past_mask, std_pad], dim=1)     # (B, N+50)
                else:
                    past_pad = torch.zeros(batch_size, past_boxes.shape[1], dtype=torch.bool, device=device)
                    key_padding = torch.cat([past_pad, std_pad], dim=1)
            else:
                key_padding = past_mask

            # maschera con 0 (do attention) e valori negativi (do not do attention)
            Q = key_padding.shape[1]
            neg = torch.finfo(dtype).min

            encoder_attention_mask = torch.zeros(batch_size, 1, Q, Q, dtype=dtype, device=device)
            encoder_attention_mask = encoder_attention_mask.masked_fill(key_padding[:, None, None, :], neg)

            if block_diag:
                N = past_boxes.shape[1]                       # slot passato = [0:N]; standard = [N:]
                encoder_attention_mask[:, :, :N, N:] = neg    # le past non attendono le standard
                encoder_attention_mask[:, :, N:, :N] = neg    # le standard non attendono le past

        # 7. Transformer decoder
        decoder_outputs = model.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=encoder_attention_mask,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # 8. Predictions
        if self.phase == 2:
            hidden_states = decoder_outputs.last_hidden_state
        else:
            num_tracked_drones = past_boxes.shape[1]
            hidden_states = decoder_outputs.last_hidden_state[:, :num_tracked_drones]
        
        # con use_past_class_head le past-query usano past_class_embed
        if self.past_class_embed is not None and (self.phase == 1 or query_mode in ('both', 'past_only')):
            if query_mode == 'both' and self.phase == 2:
                N = past_boxes.shape[1]                       # slot passato = [0:N], standard = [N:]
                logits = torch.cat([
                    self.past_class_embed(hidden_states[:, :N]),
                    branch.class_embed[-1](hidden_states[:, N:]),
                ], dim=1)
            else:
                # solo past query
                logits = self.past_class_embed(hidden_states)
        else:
            logits = branch.class_embed[-1](hidden_states)
        pred_boxes= branch.bbox_embed[-1](hidden_states).sigmoid() # produce le box in coord 0,1 con la sigmoid

        # deep supervision RT-DETR: predizioni di ogni layer del decoder
        intermediate_logits = None
        intermediate_boxes = None
        if use_standard and getattr(self.config, 'auxiliary_loss', False) \
                and getattr(decoder_outputs, 'intermediate_hidden_states', None) is not None:
            inter_hs = decoder_outputs.intermediate_hidden_states   # (B, L, Q, D)
            _il, _ib = [], []
            for i in range(inter_hs.shape[1]):
                _lh = inter_hs[:, i]
                _il.append(branch.class_embed[i](_lh))
                _ib.append(branch.bbox_embed[i](_lh).sigmoid())
            intermediate_logits = torch.stack(_il, dim=1)   # (B, L, Q, num_labels)
            intermediate_boxes  = torch.stack(_ib, dim=1)   # (B, L, Q, 4)

        # forecasting e present refinement solo sulle query-passato (le prime N)
        future_boxes = None
        present_refined = None
        num_past_q = past_boxes.shape[1]                          # N (past_boxes = past_boxes_proj)
        run_forecast = (
            self.forecasting_head is not None
            and num_past_q > 0
            and (self.phase == 1 or query_mode in ('both', 'past_only'))
        )
        if run_forecast:
            hs_past = hidden_states[:, :num_past_q]               # (B, N, D)
            pb_past = pred_boxes[:, :num_past_q]                  # (B, N, 4)

            vel_past = None
            if past_velocity is not None:
                vel_past = torch.zeros(batch_size, num_past_q, 4, device=device, dtype=pred_boxes.dtype)
                n = min(past_velocity.shape[1], num_past_q)
                vel_past[:, :n] = past_velocity[:, :n].to(pred_boxes.dtype)

            present_refined, future_boxes = self.forecasting_head(
                hs_past, pb_past,
                fused_features=fused_encoder_features,
                future_time_deltas=future_time_deltas,
                past_velocity=vel_past,
                use_present_refine=self.use_present_refine,
            )   # present_refined: (B, N, 4) | future_boxes: (B, N, T, 4)

        return RTDetrMultiScaleOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            encoder_last_hidden_state=fused_encoder_features,
            future_boxes=future_boxes,
            present_refined=present_refined,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            intermediate_logits=intermediate_logits,
            intermediate_boxes=intermediate_boxes,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        event_frames: Dict[int, torch.Tensor],
        labels: Optional[List[Dict]] = None,
        past_boxes: Optional[torch.Tensor] = None,
        past_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        query_mode: str = 'both',
        future_time_deltas: Optional[torch.Tensor] = None,
    ) -> RTDetrMultiScaleOutput:
        """
        Args:
        """
        present_durations = [d for d in self.durations_ms if d in event_frames]

        # 1. Backbone + encoder for each temporal branch
        encoder_features_per_branch = []

        if self.shared_cat:
            B = next(iter(event_frames.values())).shape[0]
            stacked = torch.cat([event_frames[d] for d in present_durations], dim=0)
            all_feats = self._run_backbone_and_encoder(self.shared_branch, stacked)
            for i in range(len(present_durations)):
                encoder_features_per_branch.append([s[i*B:(i+1)*B] for s in all_feats])
        else:
            for d in present_durations:
                branch = self.shared_branch if self.use_shared_weights else self.branches[f'branch_{d}ms']
                encoder_features_per_branch.append(self._run_backbone_and_encoder(branch, event_frames[d]))

        # 2. Fuse encoder features across temporal branches
        fused = self.fusion(encoder_features_per_branch, present_durations)

        # 3. Decoder with fused features (past_boxes injected as extra queries)
        primary = self.shared_branch if self.use_shared_weights else self.branches[f'branch_{self.durations_ms[0]}ms']
        

        assert past_boxes is not None, "past_boxes is required"
        past_boxes_proj = self.past_encoder(past_boxes) # (B, N, D)
        last_ref_points = past_boxes[:, :, 0, :] # (B, N, 4) da viz_collate p_idx=0 e lo step più recente

        # velocità per l'ancora CV, media sugli ultimi k step (p_idx=0 = più recente)
        past_velocity = None
        if self.forecasting_head is not None and self.use_cv_anchor and past_boxes.shape[2] >= 2:
            k = max(1, min(self.vel_avg_k, past_boxes.shape[2] - 1))   # 1..P-1
            past_velocity = (past_boxes[:, :, 0, :] - past_boxes[:, :, k, :]) / k   # (B, N, 4)

        return self._run_decoder_with_fused_features(
            primary,
            fused,
            labels=labels,
            past_boxes=past_boxes_proj,
            past_mask=past_mask,
            past_ref_points=last_ref_points,
            query_mode=query_mode,
            future_time_deltas=future_time_deltas,
            past_velocity=past_velocity,
        )


# from transformers import RTDetrV2ForObjectDetection, RTDetrV2Config
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclass

# # from torch.func import rearrange

# @dataclass
# class RTDetrMultiScaleOutput:
#     ''' Just for cleaner return '''
#     logits: Optional[torch.Tensor] = None
#     pred_boxes: Optional[torch.Tensor] = None
#     encoder_last_hidden_state: Optional[List[torch.Tensor]] = None

# class LogFourierScaleEncoding(nn.Module):
#     """Log-Fourier temporal encoding for duration information."""

#     def __init__(self, dim: int, num_bands: int = 16):
#         super().__init__()
#         self.num_bands = num_bands
#         self.dim = dim
#         self.freqs = 2 ** torch.linspace(0, num_bands - 1, num_bands)
#         self.proj = nn.Linear(2 * num_bands, dim)

#     def forward(self, dt: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             dt: (B, N) accumulation lengths in milliseconds
#         Returns:
#             (B, N, D) temporal embeddings
#         """
#         dt = torch.log(dt + 1e-6).unsqueeze(-1)       # (B, N, 1)
#         freqs = self.freqs.to(dt.device)
#         x = dt * freqs                                  # (B, N, num_bands)
#         pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
#         return self.proj(pe)                            # (B, N, D)

# class PastEncoder(nn.Module):

#     def __init__(self, num_past_steps: int, d_model: int):
#         super().__init__()
        
#         self.mlp = nn.Sequential(
#             nn.Linear(num_past_steps*4, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#             nn.LayerNorm(d_model),
#         )
        
#     def forward(self, past_boxes: torch.Tensor) -> torch.Tensor:
#        # x = rearrange(past_boxes, "b n p c -> b n (p c)")
#         x = past_boxes.flatten(2)
#         x =self.mlp(x)
#         return x

# class MultiScaleAttentionFusion(nn.Module):
#     """
#     Attention-based fusion of multi-scale encoder features across temporal branches.

#     Processes each spatial scale independently: for every scale, the features
#     from all N duration branches are fused via cross-attention using a learned
#     fusion token as query and temporally-encoded branch features as keys/values.
#     The output is added as a residual on top of the mean across branches.
#     """

#     def __init__(self, d_model: int = 256, num_heads: int = 8, num_scales: int = 3):
#         super().__init__()
#         self.num_scales = num_scales
#         self.d_model = d_model

#         self.temporal_encoding = LogFourierScaleEncoding(dim=d_model, num_bands=16)

#         self.scale_attentions = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
#             for _ in range(num_scales)
#         ])
#         self.scale_norms = nn.ModuleList([
#             nn.LayerNorm(d_model) for _ in range(num_scales)
#         ])
#         self.fusion_tokens = nn.ParameterList([
#             nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(num_scales)
#         ])

#         for token in self.fusion_tokens:
#             nn.init.normal_(token, std=0.02)

#        # Zero-init so the module starts as a mean fusion (stable fine-tuning)
#         for attn in self.scale_attentions:
#             attn.out_proj.weight.data.zero_()
#             attn.out_proj.bias.data.zero_()

#     def forward(
#         self,
#         multi_scale_features: List[List[torch.Tensor]],
#         durations: List[int],
#     ) -> List[torch.Tensor]:
#         """
#         Args:
#             multi_scale_features: [num_branches][num_scales] tensors, each (B, C, H, W)
#             durations: duration in ms for each branch

#         Returns:
#             List of fused feature tensors per scale: [(B, C, H_s, W_s), ...]
#         """
#         num_branches = len(multi_scale_features)
#         device = multi_scale_features[0][0].device
#         fused_scales = []

#         for scale_idx in range(self.num_scales):
#             scale_features = [multi_scale_features[b][scale_idx] for b in range(num_branches)]
#             B, C, H, W = scale_features[0].shape
#             L = H * W

#             flattened = [f.flatten(2).transpose(1, 2) for f in scale_features] # (B, L, C)
#             stacked = torch.stack(flattened, dim=1)                           # (B, K, L, C)
#             mean_fused = stacked.mean(dim=1)                                     # (B, L, C)

#             K = num_branches
#             reshaped = stacked.permute(0, 2, 1, 3).reshape(B * L, K, C)
#             reshaped = self.scale_norms[scale_idx](reshaped)

#             durations_t = torch.tensor(durations, device=device, dtype=torch.float32)
#             dt_input = durations_t.unsqueeze(0).expand(B * L, -1)            # (B*L, K)
#             time_embed = self.temporal_encoding(dt_input)                       # (B*L, K, D)

#             query = self.fusion_tokens[scale_idx].expand(B * L, -1, -1)         # (B*L, 1, C)
#             key = reshaped + time_embed
#             value = reshaped

#             attn_out, _ = self.scale_attentions[scale_idx](query, key, value)
#             residual = attn_out.squeeze(1).reshape(B, L, C)

#             fused = (mean_fused + residual).transpose(1, 2).reshape(B, C, H, W)
#             fused_scales.append(fused)

#         return fused_scales


# class MultiDurationRTDetrAttention(nn.Module):

#     def __init__(
#         self,
#         checkpoint: str = "PekingU/rtdetr_v2_r18vd",
#         num_labels: int = 1,
#         num_past_steps: int = 10,
#         durations_ms: List[int] = [3, 33, 165, 330],
#         use_shared_weights: bool = False,
#         shared_cat: bool = False,
#         phase: int = 1,
#         num_standard_queries: int = 5,
#     ):
#         super().__init__()

#         self.durations_ms = sorted(durations_ms)
#         self.checkpoint = checkpoint
#         self.num_scales = 3
#         self.use_shared_weights = use_shared_weights
#         self.shared_cat = shared_cat and use_shared_weights

#         CLASS_TO_ID = {'drone': 0}
#         ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

#         if use_shared_weights:
#             self.shared_branch = RTDetrV2ForObjectDetection.from_pretrained(
#                 checkpoint, num_labels=num_labels,
#                 id2label=ID_TO_CLASS, label2id=CLASS_TO_ID,
#                 ignore_mismatched_sizes=True, num_queries=50,
#             )
#             ref_branch = self.shared_branch
#         else:
#             self.branches = nn.ModuleDict()
#             for d in self.durations_ms:
#                 self.branches[f'branch_{d}ms'] = RTDetrV2ForObjectDetection.from_pretrained(
#                     checkpoint, num_labels=num_labels,
#                     id2label=ID_TO_CLASS, label2id=CLASS_TO_ID,
#                     ignore_mismatched_sizes=True, num_queries=50,
#                 )
#             ref_branch = self.branches[f'branch_{self.durations_ms[0]}ms']

#         self.config = ref_branch.config
        
#         D = self.config.d_model # 256
#         P = num_past_steps

#         '''TEST PAST QUERY NON INFORMATIVO'''
#         # self.neutral_query = nn.Parameter(torch.zeros(1, 1, D))
#         # nn.init.normal_(self.neutral_query, std=0.02)
#         '''FINE TEST'''
        
#         self.past_encoder = PastEncoder(num_past_steps=P, d_model=D)

#         self.fusion = MultiScaleAttentionFusion(d_model=D, num_heads=8, num_scales=self.num_scales)
#         self.phase = phase
        
#         self.num_standard_queries = num_standard_queries

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def _run_backbone_and_encoder(
#         self,
#         branch: RTDetrV2ForObjectDetection,
#         pixel_values: torch.Tensor,
#     ) -> List[torch.Tensor]:
#         """Run ResNet backbone + hybrid encoder, return 3 multi-scale feature maps."""
#         model = branch.model
#         B, _, H, W = pixel_values.shape
#         device = pixel_values.device

#         pixel_mask = torch.ones((B, H, W), device=device, dtype=torch.long)
#         backbone_outputs = model.backbone(pixel_values, pixel_mask)

#         projected = [
#             model.encoder_input_proj[lvl](feat)
#             for lvl, (feat, _) in enumerate(backbone_outputs)
#         ]
#         return model.encoder(projected).last_hidden_state # List[(B, D, H_s, W_s)]

#     def _run_decoder_with_fused_features(
#         self,
#         branch: RTDetrV2ForObjectDetection,
#         fused_encoder_features: List[torch.Tensor],
#         labels: Optional[List[Dict]] = None,
#         past_boxes: Optional[torch.Tensor] = None,
#         past_mask: Optional[torch.Tensor] = None,
#         past_ref_points: Optional[torch.Tensor] = None,
#         query_mode: str = 'both',
#     ) -> RTDetrMultiScaleOutput:

#         model= branch.model
#         device = fused_encoder_features[0].device
#         dtype = fused_encoder_features[0].dtype
#         batch_size = fused_encoder_features[0].shape[0]

#         # 1. Project fused features through decoder input projections
#         sources = [model.decoder_input_proj[lvl](src) for lvl, src in enumerate(fused_encoder_features)]

#         # 2. Flatten spatial dimensions and build level metadata
#         source_flatten = []
#         spatial_shapes_list = []
#         spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)

#         for lvl, src in enumerate(sources):
#             _, _, h, w = src.shape
#             spatial_shapes[lvl, 0] = h
#             spatial_shapes[lvl, 1] = w
#             spatial_shapes_list.append((h, w)) # per deformable attention
#             source_flatten.append(src.flatten(2).transpose(1, 2)) # (B, H*W, C)

#         source_flatten = torch.cat(source_flatten, dim=1)       # (B, total_len, C)
#         level_start_index = torch.cat([
#             spatial_shapes.new_zeros((1,)),
#             spatial_shapes.prod(1).cumsum(0)[:-1],
#         ])

#         # 3. Generate anchors ovvero il punto di partenza per le proposte dell encoder
#         anchors, valid_mask = model.generate_anchors(
#             tuple(spatial_shapes_list), device=device, dtype=dtype
#         )

#         # 4. Encoder proposals → top-k query initialisation
#         use_standard = (self.phase == 2 and query_mode in ('both', 'standard_only'))

#         if use_standard: # solo qua calcolo le proposte dell encoder e prendo le topk
#             memory = valid_mask.to(dtype) * source_flatten
#             output_memory = model.enc_output(memory)
#             enc_outputs_class = model.enc_score_head(output_memory)
#             enc_outputs_coord_logits = model.enc_bbox_head(output_memory) + anchors  # logit space

#             # prendo solo 50 proposte da usare come standard queries assieme alle past boxes queries
#             _, topk_ind = torch.topk(
#                 enc_outputs_class.max(-1).values, 
#                 self.num_standard_queries, 
#                 dim=1
#             )
        
#             # creo i reference points per le standard queries
#             std_ref = enc_outputs_coord_logits.gather(
#                 dim=1, 
#                 index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
#             )

#             if self.config.learn_initial_query:
#                 std_q = model.weight_embedding.tile([batch_size, 1, 1])[:, :self.num_standard_queries, :]
#             else:
#                 std_q = output_memory.gather(
#                     dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
#                 ).detach()


#         past_ref_points_unact = torch.logit(past_ref_points.clamp(1e-4, 1-1e-4))

#         # Quali query dare al decoder
#         if self.phase == 1 or query_mode == 'past_only': 
#             target = past_boxes # solo query del passato
#             ref_points = past_ref_points_unact
#         elif query_mode == 'standard_only':
#             target = std_q # solo query standard
#             ref_points = std_ref
#         else:  
#             target = torch.cat([past_boxes, std_q], dim=1)
#             ref_points = torch.cat([past_ref_points_unact, std_ref], dim=1)


#         # encoder attention mask: potrei avere solo droni senza passato, maschero quinid le query passate non informative
#         encoder_attention_mask = None
#         if query_mode != 'standard_only' and past_mask is not None:
#             if query_mode == 'both' and self.phase == 2:
#                 std_pad = torch.zeros(batch_size, self.num_standard_queries, dtype=torch.bool, device=device)   # tutte False
#                 key_padding = torch.cat([past_mask, std_pad], dim=1)     # (B, N+5)
#             else:
#                 key_padding = past_mask 
        
#             # maschera con 0 (do attention) e valori negativi (do not do attention)
#             Q = key_padding.shape[1]
#             neg = torch.finfo(dtype).min

#             encoder_attention_mask = torch.zeros(batch_size, 1, Q, Q, dtype=dtype, device=device)
#             encoder_attention_mask = encoder_attention_mask.masked_fill(key_padding[:, None, None, :], neg)

#         # 7. Transformer decoder
#         decoder_outputs = model.decoder(
#             inputs_embeds=target,
#             encoder_hidden_states=source_flatten,
#             encoder_attention_mask=encoder_attention_mask,
#             reference_points=ref_points,
#             spatial_shapes=spatial_shapes,
#             spatial_shapes_list=spatial_shapes_list,
#             level_start_index=level_start_index,
#             output_attentions=False,
#             output_hidden_states=True,
#             return_dict=True,
#         )

#         # 8. Predictions
#         if self.phase == 2:
#             hidden_states = decoder_outputs.last_hidden_state
#         else:
#             num_tracked_drones = past_boxes.shape[1]
#             hidden_states = decoder_outputs.last_hidden_state[:, :num_tracked_drones]
        
#         # Applico le teste
#         logits = branch.class_embed[-1](hidden_states)
#         pred_boxes= branch.bbox_embed[-1](hidden_states).sigmoid() # produce le box in coord 0,1 con la sigmoid
        

#         return RTDetrMultiScaleOutput(
#             logits=logits,
#             pred_boxes=pred_boxes,
#             encoder_last_hidden_state=fused_encoder_features,
#         )

#     def forward(
#         self,
#         event_frames: Dict[int, torch.Tensor],
#         labels: Optional[List[Dict]] = None,
#         past_boxes: Optional[torch.Tensor] = None,
#         past_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = True,
#         query_mode: str = 'both', 
#     ) -> RTDetrMultiScaleOutput:

#         present_durations = [d for d in self.durations_ms if d in event_frames]

#         # 1. Backbone + encoder for each temporal branch
#         encoder_features_per_branch = []

#         if self.shared_cat:
#             B = next(iter(event_frames.values())).shape[0]
#             stacked = torch.cat([event_frames[d] for d in present_durations], dim=0)
#             all_feats = self._run_backbone_and_encoder(self.shared_branch, stacked)
#             for i in range(len(present_durations)):
#                 encoder_features_per_branch.append([s[i*B:(i+1)*B] for s in all_feats])
#         else:
#             for d in present_durations:
#                 branch = self.shared_branch if self.use_shared_weights else self.branches[f'branch_{d}ms']
#                 encoder_features_per_branch.append(self._run_backbone_and_encoder(branch, event_frames[d]))

#         # 2. Fuse encoder features across temporal branches
#         fused = self.fusion(encoder_features_per_branch, present_durations)

#         # 3. Decoder with fused features (past_boxes injected as extra queries)
#         primary = self.shared_branch if self.use_shared_weights else self.branches[f'branch_{self.durations_ms[0]}ms']
        

#         assert past_boxes is not None, "past_boxes is required"
#         past_boxes_proj = self.past_encoder(past_boxes) # (B, N, D)
        
#         ''''TEST NO PAST QUERY, SOLO REFPOINT (BASTA L INPUT MULTITEMPORALE?)'''
#         # print("Uso query_past non informative")
#         # past_boxes_proj = self.neutral_query.expand(past_boxes.shape[0], past_boxes.shape[1], -1)
#         '''FINE TEST'''
        
#         last_ref_points = past_boxes[:, :, 0, :] # (B, N, 4) da viz_collate p_idx=0 e lo step più recente


#         return self._run_decoder_with_fused_features(
#             primary, 
#             fused, 
#             labels=labels, 
#             past_boxes=past_boxes_proj,
#             past_mask=past_mask,
#             past_ref_points=last_ref_points,
#             query_mode=query_mode,
#         )

