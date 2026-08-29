import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train.trainers.base_trainer import BaseTrainer
from typing import Dict, List, Optional, Tuple
from data.dataset_factory import get_dataset
import torch.nn.functional as F
from utils.local_detr_utils import generalized_box_iou, paired_generalized_box_iou, center_to_corners_format
import os 
from PIL import Image as PILImage
import random
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

_DBG = {'n': 0}

def _seed_everything(seed: int):
    """Seed globale per riproducibilità (main process)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _seed_worker(worker_id):
    """Seed deterministico per ogni worker del DataLoader (augmentation riproducibile)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sigmoid_focal_loss(logits, targets, num_pos, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = (alpha_t * loss).sum() / max(num_pos, 1)
    return loss

def collide(traj, present_gt, thr=0.15):
    if present_gt.shape[0] == 0: # nessuna GT presente
        return False
    recent = traj[0].reshape(1, 4) # box più recente
    recent_xyxy  = center_to_corners_format(recent)   
    present_xyxy = center_to_corners_format(present_gt)
    iou = box_iou(recent_xyxy, present_xyxy)          
    return iou.max().item() >= thr

def collate_fn_detection_rtdetrv2_past_conditioned(batch, image_processor, config=None, augment=True):

    durations = list(batch[0][0].keys())

    phase = getattr(config, "phase", 1)

    # --- Query-mode misto (solo train, solo phase 2): randomizza la modalità per batch ---
    mode = 'both'
    if augment and phase == 2 and getattr(config, "use_mixed_query_mode", 0):
        r = random.random()
        p_both = getattr(config, "p_both", 0.6)
        p_past = getattr(config, "p_past", 0.2)
        if r < p_both:
            mode = 'both'
        elif r < p_both + p_past:
            mode = 'past_only'
        else:
            mode = 'standard_only'

    # --- Augmentation: solo train, solo phase 2, solo modalità 'both' ---
    is_both  = (mode == 'both')
    use_dropout = bool(augment and phase == 2 and is_both and getattr(config, "use_past_dropout", 0))
    drop_p = getattr(config, "past_dropout_p", 0.3)
    use_fake = bool(augment and phase == 2 and is_both and getattr(config, "use_fake_past", 0))
    fake_p = getattr(config, "fake_past_p", 0.3)
    fake_max_k = getattr(config, "fake_max_k", 3)
    fake_collide_thr = getattr(config, "fake_collide_thr", 0.15)

    valid_batch = []  # per i droni con passato
    skipped = 0

    ''' TEST DROPOUT PASSATI '''
    n_dropped = 0  
    n_total   = 0 
    '''FINE TEST '''

    for frames_i, target in batch:
        present_ids = target.get('ids', torch.zeros(0, dtype=torch.int64)) # id dei droni presenti ora
        past_ids_list = target.get('past_ids', []) # lista di 10 tensori (10 passati) ognuno con gli id dei droni in quel passato

        if len(present_ids) == 0: # nessun drone nella scena
            skipped += 1 
            continue

        if len(past_ids_list) == 0 and phase == 1: # in phase 1 salto se non ho passato 
            skipped += 1
            continue

        if len(past_ids_list) == 0:  
            valid_ids = []
        else: # ho droni con passato
            valid_ids = set(present_ids.tolist()) # prendo gli id dei droni presenti con passato
            for pid_tensor in past_ids_list:
                valid_ids &= set(pid_tensor.tolist()) # interseco id droni presenti con id di ogni step passato. In questo modo mi rimangono solo i drono che compaiono ovunque (presente e tutti i 10 passati)
            valid_ids = sorted(valid_ids)

        ''' TEST DROPOUT PASSATI '''
        # rimuovo randomicamente id di droni con passato, finisce in automatico in new_labels e non viene ricostruito il passato
        if use_dropout and drop_p > 0 and len(valid_ids) > 0:
            n_before = len(valid_ids)
            valid_ids = [vid for vid in valid_ids if random.random() > drop_p]
            n_dropped += (n_before - len(valid_ids))
            n_total   += n_before
        '''FINE TEST'''

        if len(valid_ids) > 0 or phase == 2: 
            valid_batch.append((frames_i, target, valid_ids))
        else:
            skipped += 1

    if len(valid_batch) == 0: 
        return None  # salto il batch

    batch_size = len(valid_batch) # quanto frame con droni con passato ho
    num_past = getattr(config, "num_past_annotations", 10)  

    # Build frames + present labels (ordered by valid_ids)
    frames_dict = {}
    present_labels = None

    for d in durations: # costruisco frame + label 
        images = []
        annotations = []

        for frames_i, target, valid_ids in valid_batch:

            img = frames_i[d]
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)
            images.append(img_np)

            h, w = img_np.shape[:2]

            # Map present id -> box
            present_ids = target['ids']
            present_boxes = target['boxes']
            id_to_box = {int(pid): box for pid, box in zip(present_ids, present_boxes)} # con questa mappa posso recuperare la box di un drone dal suo id

            # Build coco_anns in valid_ids order so label position i == drone valid_ids[i]
            coco_anns = []
            for vid in valid_ids: # labels solo per i droni validi
                cx, cy, bw, bh = id_to_box[vid].tolist()
                coco_anns.append({
                    'bbox': [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    'category_id': 0,
                    'area': bw * w * bh * h,
                    'iscrowd': 0,
                    'id': len(coco_anns),
                })
            annotations.append({
                'image_id': target.get('image_id', 0),
                'annotations': coco_anns,
            })

        encoding = image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
            do_rescale=False,
        )

        frames_dict[d] = encoding['pixel_values']

        if present_labels is None:
            present_labels = [
                {'class_labels': lbl['class_labels'], 'boxes': lbl['boxes']}
                for lbl in encoding['labels']
            ]

    # Build past_boxes (B, max_objects, P, 4) in valid_ids order
    max_objects = max(len(valid_ids) for _, _, valid_ids in valid_batch) # numero massimo di droni validi tra i sample
    past_boxes = torch.zeros(batch_size, max_objects, num_past, 4)
    past_mask = torch.ones(batch_size, max_objects, dtype=torch.bool)

    for b, (_, target, valid_ids) in enumerate(valid_batch): # costruisco past_boxes + past_mask
        past_mask[b, :len(valid_ids)] = False # gli slot relativi ai primi valid ids devono essere False (non voglio mascherarli)
        past_ids_list = target['past_ids']  
        past_boxes_list = target['past_boxes'] 

        # Mappa annidata id -> {timestep -> box}: per ogni step passato e drone in quello step salvo la sua box
        id_to_past: dict = {}
        for p_idx, (pid_t, pb_t) in enumerate(zip(past_ids_list, past_boxes_list)):
            for k, pid in enumerate(pid_t.tolist()):
                if pid not in id_to_past:
                    id_to_past[pid] = {}
                id_to_past[pid][p_idx] = pb_t[k]

        for drone_idx, vid in enumerate(valid_ids):
            for p_idx in range(num_past):
                past_boxes[b, drone_idx, p_idx] = id_to_past[vid][p_idx]

    # Build future_gt (B, max_objects, T, 4) + future_mask (B, max_objects, T) — forecasting, solo query-passato
    num_future = getattr(config, "num_future_steps", 0)
    future_gt = None
    future_mask = None
    if num_future > 0:
        future_gt = torch.zeros(batch_size, max_objects, num_future, 4)
        future_mask = torch.zeros(batch_size, max_objects, num_future, dtype=torch.bool)
        for b, (_, target, valid_ids) in enumerate(valid_batch):
            fut_ids_list = target.get('future_ids', [])
            fut_box_list = target.get('future_boxes', [])
            # id -> {timestep futuro -> box}
            id_to_future: dict = {}
            for t_idx, (fid_t, fb_t) in enumerate(zip(fut_ids_list, fut_box_list)):
                for k, fid in enumerate(fid_t.tolist()):
                    id_to_future.setdefault(fid, {})[t_idx] = fb_t[k]
            for drone_idx, vid in enumerate(valid_ids):
                fut = id_to_future.get(vid, {})
                for t_idx in range(num_future):
                    if t_idx in fut:                       # il drone esiste in quello step futuro
                        future_gt[b, drone_idx, t_idx] = fut[t_idx]
                        future_mask[b, drone_idx, t_idx] = True

    ''' TEST: inserisco nel frames query passato finte per cui sara assente il drone '''
    drones_with_past_in_batch = []
    for b, (_, target, valid_ids) in enumerate(valid_batch):
        for i in range(len(valid_ids)):
            drones_with_past_in_batch.append( (b, past_boxes[b, i]) )

    fake_per_sample = []
    if use_fake:
        for sample in range(batch_size):
            # presenza randomica: solo fake_p dei sample riceve finti
            if random.random() >= fake_p:
                fake_per_sample.append([])
                continue

            present_gt = valid_batch[sample][1]['boxes']
            candidates = [traj for (current, traj) in drones_with_past_in_batch if current != sample]
            random.shuffle(candidates)

            k = random.randint(1, fake_max_k)   # numero massimo di finti scelto a caso in [1, fake_max_k]
            scelte = []
            for traj in candidates:
                if len(scelte) == k:
                    break
                if not collide(traj, present_gt, thr=fake_collide_thr):
                    scelte.append(traj)
            fake_per_sample.append(scelte)
    else:
        # nessun finto: max_fake=0, i cat sottostanti sono no-op, fake_mask tutto False
        fake_per_sample = [[] for _ in range(batch_size)]

    # da fake_per_sample (lista) a fake_boxes + fake_valid in tensori
    max_fake = max((len(s) for s in fake_per_sample), default=0)
    fake_boxes = torch.zeros(batch_size, max_fake, num_past, 4)
    fake_valid = torch.zeros(batch_size, max_fake, dtype=torch.bool)
    for b, scelte in enumerate(fake_per_sample):
        for j, traj in enumerate(scelte):
            fake_boxes[b, j] = traj
            fake_valid[b, j] = True

    ## concateno le query finte a past_boxes
    past_boxes = torch.cat([past_boxes, fake_boxes], dim=1)

    ## la maschera per le query passate sara cosi
    past_mask = torch.cat([past_mask, ~fake_valid], dim=1)
    
    ## Ora mi serve la maschera per la loss: true solo sulle fake_past valide 
    ##avra la stessa dimensione di query_past 
    ## Mi serve per estrarre i logits associati a fake past validi che voglio classificare come background
    fake_mask = torch.cat([
        torch.zeros(batch_size, max_objects, dtype=torch.bool), # slot reali mai finto
        fake_valid # slot finti dove ho true sulle fake_past valide
    ], dim=1)

    '''FINE TEST'''

    # Build new_labels: present drones not in valid_ids, cioe i droni senza passato, devono avere box in new_Labels -> Hungarian targets
    new_labels = []
    for _, target, valid_ids in valid_batch:
        valid_set = set(valid_ids)
        nb, nl = [], []
        for pid, box in zip(target['ids'].tolist(), target['boxes']):
            if pid not in valid_set:
                nb.append(box)
                nl.append(0)
        new_labels.append({
            'class_labels': torch.tensor(nl, dtype=torch.int64),
            'boxes': torch.stack(nb) if nb else torch.zeros(0, 4),
        })

    ''' TEST DROPOUT PASSATI '''
    # if n_total > 0:
    #     frac = n_dropped / n_total
    #     tot_new = sum(nl['boxes'].shape[0] for nl in new_labels)
    #     if _DBG['n'] < 10:
    #         print(f"[dropout] droppati {n_dropped}/{n_total} droni con passato "
    #             f"({frac:.2f}, atteso ~{drop_p}) | new_labels totali nel batch: {tot_new}")

    #         _DBG['n'] += 1
    ''' FINE TEST'''
        
    ''' DEBUG FAKE TEST'''
    # if _DBG['n'] < 5:
    #     print(f"[fake] past_boxes={tuple(past_boxes.shape)} | max_real={max_objects} max_fake={max_fake} "
    #           f"| fake totali={int(fake_mask.sum())} | fake per-sample={fake_mask.sum(1).tolist()}")
    #     _DBG['n'] += 1

      
    return {
        'frames': frames_dict,
        'labels': present_labels, # droni con passato
        'past_boxes': past_boxes,
        'past_mask': past_mask, # maschera per il padding di droni che non esistono nel frame (past_boxes e grande max_objects)
        'new_labels': new_labels, # droni senza passato
        'fake_mask': fake_mask, # TEST FAKE PAST
        'query_mode': mode, # modalità campionata (mixed-mode training)
        'future_gt': future_gt, # (B, max_objects, T, 4) — forecasting (None se num_future_steps=0)
        'future_mask': future_mask, # (B, max_objects, T) — True dove il drone esiste nel futuro
    }

class PastConditionedTrainer(BaseTrainer):
    def __init__(self, config):
        _seed_everything(getattr(config, 'seed', 42))
        super().__init__(config)
        self.global_step = 0   # contatore monotòno globale (asse x di wandb)
        # early-stopping: inizializzati qui così esistono anche se la 1a val NON migliora
        # (es. in resume con best_val_loss già finito) -> niente AttributeError.
        self.patience = getattr(config, 'patience', 5)
        self.patience_counter = getattr(self, 'patience_counter', 0)

    def _set_model_mode(self):
        """eval() se il detector è congelato (le statistiche BatchNorm non driftano),
        altrimenti train(). Da usare al posto di self.model.train() durante il training."""
        if getattr(self.config, 'freeze_detector', 0):
            self.model.eval()
        else:
            self.model.train()

    def _setup_datasets(self):
        """Initialize datasets and dataloaders."""
        print(" Setting up datasets and dataloaders...")

        task_modality ='tracking'

        print(f" Loading training dataset: {self.config.dataset_name} | Split: train")
        print(f" Loading validation dataset: {self.config.dataset_name} | Split: val")
        print(f" Using index file: {self.config.index_path}")
        print(f" Task modality: {task_modality}")

        # train: augmentation ON (gestita dai flag di config); val: augmentation OFF (passato pulito)
        self.train_collate = lambda batch: collate_fn_detection_rtdetrv2_past_conditioned(
            batch, image_processor=self.image_processor, config=self.config, augment=True,
        )
        self.val_collate = lambda batch: collate_fn_detection_rtdetrv2_past_conditioned(
            batch, image_processor=self.image_processor, config=self.config, augment=False,
        )

        self.train_dataset = get_dataset(
            self.config.dataset_name,
            split='train',
            modality=task_modality,
            config=self.config,
        )
        self.val_dataset = get_dataset(
            self.config.dataset_name,
            split='val',   # val = SEQUENZE held-out del train (seq_split) -> basta col leak "val=test"
            modality=task_modality,
            config=self.config
        )
        
        persistent = bool(getattr(self.config, 'persistent_workers', 1))
        val_workers = int(getattr(self.config, 'val_num_workers', 2))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_collate,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent,
            prefetch_factor=self.config.prefetch_factor,
            worker_init_fn=_seed_worker,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            collate_fn=self.val_collate,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=_seed_worker,
        )

        print(f" Training samples: {len(self.train_dataset)} | Validation samples: {len(self.val_dataset)}")

    def _setup_model(self):
        """Create model based on config.model_name (DETR or RT-DETR)."""
        model_name = getattr(self.config, 'model_name', 'rtdetrv2_past_conditioned')
        print(f"Setting up model: {model_name}...")

        ID_TO_CLASS = {0: 'drone'}
        CLASS_TO_ID = {'drone': 0}

        # RT-DETR V2 setup
        from transformers import RTDetrImageProcessor
        from models.model_factory import get_model
        
        self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")

        if self.config.use_custom_normalization:
            print("Using custom dataset statistics for normalization") 
            self.image_processor.image_mean = [0.007950919680297375, 0.010960247367620468, 0.013969575986266136]
            self.image_processor.image_std = [0.048023562878370285, 0.054145995527505875, 0.066846564412117]

        print(f"[ImageProcessor] mean: {self.image_processor.image_mean}")
        print(f"[ImageProcessor] std: {self.image_processor.image_std}")

        self.model = get_model(model_name, self.device, self.config)

        ckpt_path = getattr(self.config, 'pretrained_detector_path', '')
        if ckpt_path:
            print(f"[pretrained] Carico detector: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[pretrained] missing: {len(missing)} | unexpected: {len(unexpected)}")

        # --- Opzionale: congela tutto tranne UN modulo (default past_encoder; forecasting -> forecasting_head) ---
        if getattr(self.config, 'freeze_detector', 0):
            keep = getattr(self.config, 'trainable_when_frozen', 'past_encoder')
            for name, param in self.model.named_parameters():
                param.requires_grad = (keep in name)
            n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            names = sorted({n.split('.')[0] for n, p in self.model.named_parameters() if p.requires_grad})
            print(f"[freeze] trainable ({keep}): {n_tr/1e3:.1f}K | moduli allenabili: {names}")

        self._set_model_mode()   # eval() se freeze_detector, altrimenti train()
        self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        
        print(f"Model: RT-DETR V2 (ResNet-18) | {num_params:.1f}M parameters")
        print(f"Durations: {self.config.durations} ms")
        print(f"Fusion: Multi-scale attention-based with temporal encoding")

    def _setup_optimizer(self):
        """Setup optimizer for MultiDurationDetrMultiScale model."""
        print("  Setting up optimizer for MultiDurationDetrMultiScale model...")
        import torch.optim as optim

        # Config is already flattened by merge_args(), access directly
        lr = getattr(self.config, 'learning_rate', 1e-4)
        weight_decay = getattr(self.config, 'weight_decay', 1e-4)

        # con freeze_detector ottimizza solo i parametri trainabili (past_encoder)
        if getattr(self.config, 'freeze_detector', 0):
            params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"[freeze] optimizer su {sum(p.numel() for p in params)/1e3:.1f}K parametri trainabili")
        else:
            params = self.model.parameters()

        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        print(f"✅ Optimizer set up: {self.config.optimizer} | LR: {lr} | Weight decay: {weight_decay}")

    def _setup_scheduler(self):
        
        """Step-based scheduler: linear warmup + cosine/step decay."""
        scheduler_type = getattr(self.config, "scheduler_type", "cosine")
        warmup_epochs = getattr(self.config, "warmup_epochs",  2)
        total_epochs = self.config.epochs
        warmup_start_factor = 0.1

        if scheduler_type == "none" or self.optimizer is None:
            self.scheduler = None
            print("[RTDetrTrainer] Scheduler: disabled")
            return

        steps_per_epoch = len(self.train_loader)
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        def warmup_lambda(step):
            if step < warmup_steps:
                alpha = float(step) / float(warmup_steps)
                return warmup_start_factor + (1.0 - warmup_start_factor) * alpha
            return 1.0

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_lambda
        )

        if scheduler_type == "cosine":
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=getattr(self.config, "min_lr", 1e-8),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_steps],
            )
            print(f"[RTDetrTrainer] Scheduler: warmup ({warmup_epochs} ep) -> cosine")

        elif scheduler_type == "step":
            step_size = getattr(self.config, "lr_step_size", 30) * steps_per_epoch
            gamma = getattr(self.config, "lr_gamma",     0.1)
            step_sched = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, step_sched],
                milestones=[warmup_steps],
            )
            print(f"[RTDetrTrainer] Scheduler: warmup ({warmup_epochs} ep) -> step")

        else:
            self.scheduler = warmup_sched
            print(f"[RTDetrTrainer] Scheduler: warmup only ({warmup_epochs} ep)")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""

        self._set_model_mode()   # eval() se freeze_detector, altrimenti train()

        epoch_loss = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_giou = 0.0

        # ── Benchmark throughput (attivo solo se --bench_batches > 0) ──
        import time as _time
        bench_n = int(getattr(self.config, 'bench_batches', 0) or 0)
        bench_warmup = int(getattr(self.config, 'bench_warmup', 10) or 0)
        _bench = {'t0': None, 'n': 0, 'data': 0.0, 'compute': 0.0, 'prev_end': None}
        _cuda = (self.device.type == 'cuda')

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):

            if batch is None:
                continue

            if bench_n > 0:
                if _cuda:
                    torch.cuda.synchronize()
                _t_ready = _time.perf_counter()
                if batch_idx >= bench_warmup:
                    if _bench['t0'] is None:
                        _bench['t0'] = _t_ready
                        if _cuda:
                            torch.cuda.reset_peak_memory_stats()
                    if _bench['prev_end'] is not None:
                        _bench['data'] += _t_ready - _bench['prev_end']
            
            # Move frames to device
            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            detr_labels = self._prepare_targets(batch['labels'])
            past_boxes=batch['past_boxes'].to(self.device)
            past_mask = batch["past_mask"].to(self.device)

            mode = batch.get('query_mode', 'both')   # mixed-mode: both / past_only / standard_only

            # Forward pass
            outputs = self.model(
                event_frames=frames_dict,
                labels=detr_labels,
                past_boxes=past_boxes,
                past_mask=past_mask,
                query_mode=mode,
            )

            pred_boxes = outputs.pred_boxes  # both:(B,N+S,4) past_only:(B,N,4) standard_only:(B,S,4)
            logits = outputs.logits
            encoder_last_hidden_state = outputs.encoder_last_hidden_state

            phase = getattr(self.config, 'phase', 1)
            B = len(detr_labels)
            N = past_boxes.shape[1]   # n. query-passato (reali + fake); usato solo quando il past gira
            device = self.device
            new_labels = batch.get('new_labels', None)
            fake_mask = batch.get('fake_mask', None)
            if fake_mask is not None:
                fake_mask = fake_mask.to(device)

            # ── Loss UFFICIALE RT-DETR sul ramo STANDARD (encoder-loss + deep supervision + VFL) ──
            loss_std_rtdetr = None
            _std_ldict = None # breakdown loss RT-DETR (per il logging wandb)
            if phase == 2 and mode in ('standard_only', 'both'):
                N_std = N if mode == 'both' else 0 # in both le query standard sono le slot [N:]
                branch = self.model.primary_branch
                std_labels = []
                for b in range(B):
                    _nb = new_labels[b]['boxes'].to(device) if new_labels is not None else torch.zeros(0, 4, device=device)
                    _boxes = torch.cat([detr_labels[b]['boxes'], _nb], dim=0)   # TUTTI i droni (con + senza passato)
                    std_labels.append({
                        'class_labels': torch.zeros(_boxes.shape[0], dtype=torch.int64, device=device),
                        'boxes': _boxes,
                    })
                _inter_l = outputs.intermediate_logits[:, :, N_std:] if outputs.intermediate_logits is not None else None
                _inter_b = outputs.intermediate_boxes[:, :, N_std:] if outputs.intermediate_boxes is not None else None
                loss_std_rtdetr, _std_ldict, _ = branch.loss_function(
                    logits[:, N_std:], std_labels, device, pred_boxes[:, N_std:], branch.config,
                    _inter_l, _inter_b,
                    enc_topk_logits=outputs.enc_topk_logits,
                    enc_topk_bboxes=outputs.enc_topk_bboxes,
                )

            box_pred_list, box_gt_list = [], []
            cls_logits_list, cls_targets_list = [], []
            # present-refinement dalla forecasting head (None se non attiva).
            # Con use_present_refine=0 il modello ritorna present_refined = pred_boxes (grezze):
            present_refined = outputs.present_refined if getattr(self.config, 'use_present_refine', 1) else None
            refined_pred_list, refined_gt_list = [], []
            num_pos = 0
            fake_score_sum = 0.0
            fake_score_cnt = 0

            def _match_standard(std_logits_b, std_boxes_b, gt_boxes):
                """Hungarian query-standard -> gt; aggiorna box/cls list e num_pos."""
                nonlocal num_pos
                std_targets_b = torch.zeros_like(std_logits_b)   # tutte background, poi correggo le matchate
                if gt_boxes.shape[0] > 0:
                    with torch.no_grad():
                        cost_l1 = torch.cdist(std_boxes_b, gt_boxes, p=1)
                        giou = generalized_box_iou(
                            center_to_corners_format(std_boxes_b),
                            center_to_corners_format(gt_boxes),
                        )
                        cost = 5.0 * cost_l1 + 2.0 * (1 - giou)
                    row, col = linear_sum_assignment(cost.cpu().numpy())
                    row = torch.as_tensor(row, device=device, dtype=torch.long)
                    col = torch.as_tensor(col, device=device, dtype=torch.long)
                    std_targets_b[row] = 1.0
                    box_pred_list.append(std_boxes_b[row])
                    box_gt_list.append(gt_boxes[col])
                    num_pos += row.numel()
                cls_logits_list.append(std_logits_b)
                cls_targets_list.append(std_targets_b)

            for b in range(B):
                n = detr_labels[b]['boxes'].shape[0]   # droni con passato (present_labels)

                # --- ramo PASSATO (both, past_only; in phase 1 il modello fa comunque past-only) ---
                if mode in ('both', 'past_only') and n > 0:
                    box_pred_list.append(pred_boxes[b, :n])
                    box_gt_list.append(detr_labels[b]['boxes'])
                    num_pos += n
                    if phase == 2:
                        cls_logits_list.append(logits[b, :n])
                        cls_targets_list.append(torch.ones_like(logits[b, :n]))   # foreground
                    # present-refinement supervisionato sulla STESSA present-GT
                    if present_refined is not None:
                        refined_pred_list.append(present_refined[b, :n])
                        refined_gt_list.append(detr_labels[b]['boxes'])

                # --- fake-past -> background (solo both) ---
                if phase == 2 and mode == 'both' and fake_mask is not None:
                    fm = fake_mask[b]
                    if fm.any():
                        fake_logits = logits[b, :N][fm]
                        cls_logits_list.append(fake_logits)
                        cls_targets_list.append(torch.zeros_like(fake_logits))
                        fake_score_sum += fake_logits.sigmoid().sum().item()
                        fake_score_cnt += fake_logits.numel()

            if len(box_pred_list) == 0 and len(cls_logits_list) == 0 and loss_std_rtdetr is None:
                continue  # niente box/logits manuali E niente loss standard RT-DETR -> skippo

            loss_bbox = torch.zeros((), device=device)
            loss_giou = torch.zeros((), device=device)

            if len(box_pred_list) > 0: # ci sono droni
                pred_flat = torch.cat(box_pred_list, dim=0) # tot_box, 4
                gt_flat = torch.cat(box_gt_list, dim=0) # tot_box, 4, in questo modo pred_flat[i] e gt_flat[i] sono allineati
                loss_bbox = F.l1_loss(pred_flat, gt_flat, reduction='sum') / max(num_pos, 1) # loss di regressione sulle coordinate
                giou_diag = paired_generalized_box_iou(          # O(N): GIoU appaiato pred_i↔gt_i
                    center_to_corners_format(pred_flat),
                    center_to_corners_format(gt_flat),
                )
                loss_giou = (1 - giou_diag).sum() / max(num_pos, 1)

            # classificazione (solo phase 2 perche in 1 non ce background, ho sempre un drone)
            loss_cls = torch.zeros((), device=device)
            if phase == 2 and len(cls_logits_list) > 0:
                cls_logits = torch.cat(cls_logits_list, dim=0)
                cls_targets = torch.cat(cls_targets_list, dim=0)
                loss_cls = sigmoid_focal_loss(cls_logits, cls_targets, num_pos)

            # ── Present-refinement: L1+GIoU present_refined↔present-GT ──
            loss_present_refine = torch.zeros((), device=device)
            if len(refined_pred_list) > 0:
                rp = torch.cat(refined_pred_list, dim=0)
                rg = torch.cat(refined_gt_list, dim=0)
                num_ref = max(rp.shape[0], 1)
                loss_ref_l1 = F.l1_loss(rp, rg, reduction='sum') / num_ref
                giou_ref = paired_generalized_box_iou(
                    center_to_corners_format(rp),
                    center_to_corners_format(rg),
                )
                loss_ref_giou = (1 - giou_ref).sum() / num_ref
                loss_present_refine = loss_ref_l1 * 5.0 + loss_ref_giou * 2.0

            # ── Forecasting: L1+GIoU su (query-passato, step) validi ──
            loss_forecast = torch.zeros((), device=device)
            future_gt = batch.get('future_gt', None)
            future_boxes = outputs.future_boxes                      # (B, Q, T, 4) o None
            if future_gt is not None and future_boxes is not None:
                future_gt = future_gt.to(device)                     # (B, M, T, 4)
                future_mask = batch['future_mask'].to(device)        # (B, M, T) bool
                M = future_gt.shape[1]                               # max_objects (past reali)
                fp = future_boxes[:, :M]                             # (B, M, T, 4) — query-passato reali
                if future_mask.any():
                    pred_sel = fp[future_mask]                       # (K, 4)
                    gt_sel   = future_gt[future_mask]                # (K, 4)
                    # Weighting progressivo per-step: i passi lontani (spostamento maggiore)
                    # pesano di più -> contrastano l'attrattore "stay put" del forecasting.
                    # w_t = 1 + ((t+1)/T)·(w_max−1): ramp da ~1 (t=0) a w_max (t=T−1). w_max=1 ⇒ media uniforme.
                    T_fut = fp.shape[2]
                    w_max = float(getattr(self.config, 'forecast_step_weight_max', 3.0))
                    step_w = 1.0 + (torch.arange(1, T_fut + 1, device=device, dtype=fp.dtype) / T_fut) * (w_max - 1.0)
                    w = step_w.view(1, 1, T_fut).expand(fp.shape[0], M, T_fut)[future_mask]   # (K,) peso per coppia
                    w_sum = w.sum().clamp(min=1e-6)
                    l1_per   = F.l1_loss(pred_sel, gt_sel, reduction='none').sum(dim=-1)       # (K,) L1 per box
                    giou_per = 1.0 - paired_generalized_box_iou(
                        center_to_corners_format(pred_sel),
                        center_to_corners_format(gt_sel),
                    )                                                                          # (K,)
                    loss_fut_l1   = (l1_per * w).sum() / w_sum       # media PESATA (w_max=1 ⇒ media semplice)
                    loss_fut_giou = (giou_per * w).sum() / w_sum
                    loss_forecast = loss_fut_l1 * 5.0 + loss_fut_giou * 2.0

            fw = getattr(self.config, 'forecast_loss_weight', 1.0)
            rw = getattr(self.config, 'present_refine_loss_weight', 1.0)   # peso del present-refinement
            loss = (loss_bbox * 5.0 + loss_giou * 2.0 + loss_cls * 1.0
                    + loss_forecast * fw + loss_present_refine * rw)
            if loss_std_rtdetr is not None:
                sw = getattr(self.config, 'std_loss_weight', 1.0)
                loss = loss + loss_std_rtdetr * sw

            '''DEBUG FAKE PAST'''
            if batch_idx % 200 == 0 and fake_score_cnt > 0:
                n_fg = int((cls_targets == 1).sum()); n_bg = int((cls_targets == 0).sum())
                print(f"[cls] fg={n_fg} bg={n_bg} | score medio FAKE={fake_score_sum/fake_score_cnt:.3f} "
                      f"| loss_cls={loss_cls.item():.3f}")
            ''''''
            
            # Visualization
            self.global_step = epoch * len(self.train_loader) + batch_idx
            if bench_n == 0 and self.global_step % self.config.vis_freq == 0:
                print(f"[viz] triggering visualization at step {self.global_step}, vis_freq={self.config.vis_freq}")
                self._save_visualization(
                    frames_dict, batch['labels'], past_boxes, loss.item(), epoch,
                    future_gt=batch.get('future_gt', None),
                    future_mask=batch.get('future_mask', None),
                )
        
            # Backward pass
            self.optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            loss_val          = loss.item()
            loss_bbox_val     = loss_bbox.item()
            loss_giou_val     = loss_giou.item()
            loss_forecast_val = loss_forecast.item()
            loss_refine_val   = loss_present_refine.item()

            # Accumulate losses
            epoch_loss      += loss_val
            epoch_loss_bbox += loss_bbox_val
            epoch_loss_giou += loss_giou_val

            if self.config.use_wandb and self.run is not None:
                import wandb
                _log = {
                    "epoch": epoch + 1,
                    "train/loss_total": loss_val,
                    "train/loss_bbox": loss_bbox_val,
                    "train/loss_giou": loss_giou_val,
                    "train/loss_forecast": loss_forecast_val,
                    "train/loss_present_refine": loss_refine_val,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                }
                # standard_only: breakdown della loss ufficiale RT-DETR (vfl/bbox/giou + copie enc/aux).
                if _std_ldict is not None and len(_std_ldict) > 0:
                    _sk = list(_std_ldict.keys())
                    _sv = torch.stack([
                        v.detach().float() if torch.is_tensor(v) else torch.tensor(float(v), device=device)
                        for v in _std_ldict.values()
                    ]).tolist()
                    for _k, _v in zip(_sk, _sv):
                        _log[f"train/std_{_k}"] = _v
                wandb.log(_log, step=self.global_step)


            pbar.set_postfix({'loss': f"{loss_val:.4f}", 'fcast': f"{loss_forecast_val:.3f}",
                              'refine': f"{loss_refine_val:.3f}"})

            # ── Benchmark: chiudo la misura di questa iterazione ed eventualmente esco ──
            if bench_n > 0:
                if _cuda:
                    torch.cuda.synchronize()
                _now = _time.perf_counter()
                if batch_idx >= bench_warmup:
                    _bench['compute'] += _now - _t_ready
                    _bench['n'] += 1
                _bench['prev_end'] = _now
                if _bench['n'] >= bench_n:
                    dt = _now - _bench['t0']
                    bs = int(self.config.train_batch_size)
                    nb = _bench['n']
                    sps = bs * nb / dt if dt > 0 else 0.0
                    peak_a = torch.cuda.max_memory_allocated() / (1024 ** 2) if _cuda else 0.0
                    peak_r = torch.cuda.max_memory_reserved() / (1024 ** 2) if _cuda else 0.0
                    print(
                        f"[BENCH] bs={bs} nw={self.config.num_workers} batches={nb} "
                        f"wall={dt:.1f}s sec/batch={dt/nb:.3f} samples_s={sps:.1f} "
                        f"data_ms={_bench['data']/nb*1000:.0f} compute_ms={_bench['compute']/nb*1000:.0f} "
                        f"gpu_alloc_MB={peak_a:.0f} gpu_resv_MB={peak_r:.0f}",
                        flush=True,
                    )
                    os._exit(0)   # uscita bulletproof: il benchmark è finito

        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            'loss_bbox': epoch_loss_bbox / num_batches,
            'loss_giou': epoch_loss_giou / num_batches,
        }
    

    def train(self):
        """Main training loop."""
        if self.model is None:
            self.setup()

        # ── Resume: carica il checkpoint DOPO setup (model/optimizer/scheduler pronti) ──
        resume_path = getattr(self.config, 'resume_from_checkpoint', '')
        if resume_path:
            if os.path.exists(resume_path):
                print(f"[resume] carico checkpoint: {resume_path}")
                self.load_checkpoint(resume_path)
                print(f"[resume] riprendo dall'epoca {self.current_epoch + 1}")
            else:
                print(f"⚠️  [resume] checkpoint non trovato: {resume_path} — parto da zero")

        if self.config.apply_bias_fix:
            print("Bias fixed")
            self.apply_bias_fix(self.model)

        
        if self.config.freeze_decoder_epochs > 0:
            for name, param in self.model.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad = False
            print(f"[Train] Backbone frozen per le prime {self.config.freeze_decoder_epochs} epoche")

        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config.epochs):

            if epoch == self.config.freeze_decoder_epochs and self.config.freeze_decoder_epochs > 0:
                for param in self.model.parameters():
                    param.requires_grad = True
                print(f"[Train] Epoch {epoch+1}: backbone sbloccato, lr backbone = {self.optimizer.param_groups[0]['lr']:.2e}")

            self.current_epoch = epoch

            # Train   
            metrics = self._train_epoch(epoch)
            self.train_losses.append(metrics['loss'])

            print(f"\nEpoch {epoch + 1} | Loss: {metrics['loss']:.4f} | "
                f"BBox: {metrics['loss_bbox']:.4f} | "
                f"GIoU: {metrics['loss_giou']:.4f}")
                
            # Save curves and checkpoint
            self._save_checkpoint(epoch)
            
            # Validate
            if self.val_loader and epoch % self.config.val_epoch_freq == 0:
                val_loss = self._validate(epoch)
                self.val_losses.append(val_loss)

                # Early Stopping
                if val_loss < self.best_val_loss:
                    print(f"New Best Val Loss: {val_loss:.4f} (was {self.best_val_loss:.4f})")
                    self.best_val_loss = val_loss
                    self.patience_counter = 0  
                    self.patience = self.config.patience 
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                    
                
                if self.patience_counter >= self.patience and self.config.subsample < 100:
                    print("\n" + "!" * 60)
                    print(f"EARLY STOPPING triggered at epoch {epoch + 1}")
                    print("!" * 60)
                    break

           
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Output: {self.output_dir}")
        
        return self.train_losses, self.val_losses

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation."""
        if self.val_loader is None:
            return float('inf')

        self.model.eval()

        val_loss = 0.0
        val_loss_bbox = 0.0
        val_loss_giou = 0.0
        val_loss_cls = 0.0
        val_loss_forecast = 0.0
        val_loss_present_refine = 0.0

        phase = getattr(self.config, 'phase', 1)
        device = self.device

        pbar = tqdm(self.val_loader, desc="Validation")
        for batch_idx, batch in enumerate(pbar):

            if batch is None:
                continue

            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            detr_labels = self._prepare_targets(batch['labels'])
            past_boxes = batch['past_boxes'].to(self.device)
            past_mask = batch['past_mask'].to(self.device)

            # Forward pass (past_mask ora passato!)
            outputs = self.model(
                event_frames=frames_dict,
                labels=detr_labels,
                past_boxes=past_boxes,
                past_mask=past_mask,
            )

            pred_boxes = outputs.pred_boxes   # (B, N+5, 4)
            logits = outputs.logits       # (B, N+5, 1)

            B = len(detr_labels)
            N = past_boxes.shape[1]

            # print(f"[shapes] pred_boxes={pred_boxes.shape}  logits={logits.shape}  N={N}  Q={pred_boxes.shape[1]}")

            new_labels = batch.get('new_labels', None)
            '''TEST FAKE_PAST'''
            fake_mask = batch.get('fake_mask', None)
            if fake_mask is not None:
                fake_mask = fake_mask.to(device)
            '''FINE TEST'''

            box_pred_list, box_gt_list = [], []
            cls_logits_list, cls_targets_list = [], []
            # Proposta C: refine-loss solo con use_present_refine attivo (vedi _train_epoch)
            present_refined = outputs.present_refined if getattr(self.config, 'use_present_refine', 1) else None
            refined_pred_list, refined_gt_list = [], []
            num_pos = 0

            for b in range(B):
                # ---- ramo diretto: query-passato ----
                n = detr_labels[b]['boxes'].shape[0]
                if n > 0:
                    box_pred_list.append(pred_boxes[b, :n])
                    box_gt_list.append(detr_labels[b]['boxes'])
                    num_pos += n
                    if phase == 2:
                        cls_logits_list.append(logits[b, :n])
                        cls_targets_list.append(torch.ones_like(logits[b, :n]))
                    if present_refined is not None:
                        refined_pred_list.append(present_refined[b, :n])
                        refined_gt_list.append(detr_labels[b]['boxes'])

                # ---- ramo standard: Hungarian ----
                if phase == 2:
                    std_logits_b = logits[b, N:]      # (5, 1)
                    std_boxes_b = pred_boxes[b, N:]   # (5, 4)
                    gt_new = new_labels[b]['boxes'].to(device)   # (m, 4)
                    m = gt_new.shape[0]

                    std_targets_b = torch.zeros_like(std_logits_b)
                    if m > 0:
                        cost_l1 = torch.cdist(std_boxes_b, gt_new, p=1)        # (5, m)
                        giou = generalized_box_iou(
                            center_to_corners_format(std_boxes_b),
                            center_to_corners_format(gt_new),
                        )                                                      # (5, m)
                        cost = 5.0 * cost_l1 + 2.0 * (1 - giou)
                        row, col = linear_sum_assignment(cost.cpu().numpy())
                        row = torch.as_tensor(row, device=device, dtype=torch.long)
                        col = torch.as_tensor(col, device=device, dtype=torch.long)

                        std_targets_b[row] = 1.0
                        box_pred_list.append(std_boxes_b[row])
                        box_gt_list.append(gt_new[col])
                        num_pos += row.numel()

                    cls_logits_list.append(std_logits_b)
                    cls_targets_list.append(std_targets_b)
                
                '''TEST FAKE_PAST query-passato finte = classificazione background'''
                if phase == 2 and fake_mask is not None:
                    fm = fake_mask[b]
                    if fm.any():
                        fake_logits = logits[b, :N][fm]
                        cls_logits_list.append(fake_logits)
                        cls_targets_list.append(torch.zeros_like(fake_logits))
                '''FINE TEST'''

            if len(box_pred_list) == 0 and len(cls_logits_list) == 0:
                continue

            # box loss
            loss_bbox = torch.zeros((), device=device)
            loss_giou = torch.zeros((), device=device)
            if len(box_pred_list) > 0:
                pred_flat = torch.cat(box_pred_list, dim=0)
                gt_flat = torch.cat(box_gt_list, dim=0)
                loss_bbox = F.l1_loss(pred_flat, gt_flat, reduction='sum') / max(num_pos, 1)
                giou_diag = paired_generalized_box_iou(
                    center_to_corners_format(pred_flat),
                    center_to_corners_format(gt_flat),
                )
                loss_giou = (1 - giou_diag).sum() / max(num_pos, 1)

            # classificazione (solo phase 2)
            loss_cls = torch.zeros((), device=device)
            if phase == 2 and len(cls_logits_list) > 0:
                cls_logits = torch.cat(cls_logits_list, dim=0)
                cls_targets = torch.cat(cls_targets_list, dim=0)
                loss_cls = sigmoid_focal_loss(cls_logits, cls_targets, num_pos)

            # ── Present-refinement — stessa loss del train ──
            loss_present_refine = torch.zeros((), device=device)
            if len(refined_pred_list) > 0:
                rp = torch.cat(refined_pred_list, dim=0)
                rg = torch.cat(refined_gt_list, dim=0)
                num_ref = max(rp.shape[0], 1)
                loss_ref_l1 = F.l1_loss(rp, rg, reduction='sum') / num_ref
                giou_ref = paired_generalized_box_iou(
                    center_to_corners_format(rp),
                    center_to_corners_format(rg),
                )
                loss_ref_giou = (1 - giou_ref).sum() / num_ref
                loss_present_refine = loss_ref_l1 * 5.0 + loss_ref_giou * 2.0

            # ── Forecasting: stessa loss del train, così best-model lo segue ──
            loss_forecast = torch.zeros((), device=device)
            future_gt = batch.get('future_gt', None)
            future_boxes = outputs.future_boxes
            if future_gt is not None and future_boxes is not None:
                future_gt = future_gt.to(device)
                future_mask = batch['future_mask'].to(device)
                M = future_gt.shape[1]
                fp = future_boxes[:, :M]
                if future_mask.any():
                    pred_sel = fp[future_mask]
                    gt_sel   = future_gt[future_mask]
                    # Weighting progressivo per-step (identico al train, così il best-model lo segue).
                    T_fut = fp.shape[2]
                    w_max = float(getattr(self.config, 'forecast_step_weight_max', 3.0))
                    step_w = 1.0 + (torch.arange(1, T_fut + 1, device=device, dtype=fp.dtype) / T_fut) * (w_max - 1.0)
                    w = step_w.view(1, 1, T_fut).expand(fp.shape[0], M, T_fut)[future_mask]   # (K,)
                    w_sum = w.sum().clamp(min=1e-6)
                    l1_per   = F.l1_loss(pred_sel, gt_sel, reduction='none').sum(dim=-1)       # (K,)
                    giou_per = 1.0 - paired_generalized_box_iou(
                        center_to_corners_format(pred_sel),
                        center_to_corners_format(gt_sel),
                    )
                    loss_fut_l1   = (l1_per * w).sum() / w_sum
                    loss_fut_giou = (giou_per * w).sum() / w_sum
                    loss_forecast = loss_fut_l1 * 5.0 + loss_fut_giou * 2.0

            fw = getattr(self.config, 'forecast_loss_weight', 1.0)
            rw = getattr(self.config, 'present_refine_loss_weight', 1.0)
            loss = (loss_bbox * 5.0 + loss_giou * 2.0 + loss_cls * 1.0
                    + loss_forecast * fw + loss_present_refine * rw)

            # Accumulate
            val_loss      += loss.item()
            val_loss_bbox += loss_bbox.item()
            val_loss_giou += loss_giou.item()
            val_loss_cls  += loss_cls.item()
            val_loss_forecast += loss_forecast.item()
            val_loss_present_refine += loss_present_refine.item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'fcast': f"{loss_forecast.item():.3f}"})

        num_batches = len(self.val_loader)
        n = max(num_batches, 1)
        avg_val_loss = val_loss / n

        # Un solo log di validazione per epoca, allo stesso global_step del train
        # (fine epoca): niente più campo "step" alternativo -> asse x coerente.
        if self.config.use_wandb and self.run is not None:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "val/loss_total": avg_val_loss,
                "val/loss_bbox": val_loss_bbox / n,
                "val/loss_giou": val_loss_giou / n,
                "val/loss_cls": val_loss_cls / n,
                "val/loss_forecast": val_loss_forecast / n,
                "val/loss_present_refine": val_loss_present_refine / n,
                "val/learning_rate": self.optimizer.param_groups[0]['lr'],
            }, step=self.global_step)
        return avg_val_loss
        

    @torch.no_grad()
    def _save_visualization(
        self,
        frames_dict: Dict[int, torch.Tensor],
        targets: List[Dict],
        past_boxes: torch.Tensor,
        loss: float,
        epoch: int,
        future_gt: Optional[torch.Tensor] = None,
        future_mask: Optional[torch.Tensor] = None,
    ):
        print(f"[viz] _save_visualization called at step {self.global_step}")
        """Save visualization during training showing all duration frames in a row."""

        PRED_COLORS = [
            '#FF8C00', '#00BFFF', '#FF00FF', '#FFD700',
            '#FF69B4', '#00CED1', '#FF6347', '#BA55D3', '#FFA07A',
        ]
        GT_COLOR = '#00FF00'
        FUTURE_GT_COLOR = '#00B050'   # verde medio — box GT future (sfumate)
        FORECAST_COLOR  = '#FF3333'   # rosso       — box future predette (sfumate)

        self.model.eval()

        outputs = self.model(event_frames=frames_dict, past_boxes=past_boxes)

        # Use _build_ground_truths_batch to convert targets to xyxy in orig_size coords
        # (targets here are the raw batch labels, which still carry orig_size)
        ground_truths = self._build_ground_truths_batch(targets)

        sorted_durations = sorted(frames_dict.keys(), reverse=False)
        num_durations = len(sorted_durations)
        main_duration = sorted_durations[0]

        # orig_size from last sample in batch (consistent with [-1] indexing below)
        orig_h, orig_w = ground_truths[-1]['orig_size']

        batch_size = next(iter(frames_dict.values())).shape[0]

        # Post-process using orig_size -> predicted boxes in orig_size pixel coords
        target_sizes = [(orig_h, orig_w)] * batch_size
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.processor_threshold_train,
        )

        # Aspect-ratio-correct figure
        aspect = orig_w / orig_h
        fig, axes = plt.subplots(1, num_durations, figsize=(5 * num_durations * aspect, 5))
        if num_durations == 1:
            axes = [axes]

        fig.suptitle(f"Step {self.global_step} | Epoch {epoch + 1} | Loss: {loss:.4f}", fontsize=12)

        for idx, duration in enumerate(sorted_durations):
            ax = axes[idx]
            image_tensor = frames_dict[duration][-1]  # Last sample in batch
            image = image_tensor.cpu().numpy()

            # Resize tensor to orig_size via PIL for correct aspect ratio
            if image.shape[0] == 1:
                image = image.squeeze(0)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image = np.array(pil_img) / 255.0
                ax.imshow(image, cmap='gray')
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = np.clip(image, 0, 1)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image = np.array(pil_img) / 255.0
                ax.imshow(image)
            else:
                image = image[0]
                ax.imshow(image, cmap='gray')

            ax.set_title(f"{duration}ms")

            if duration == main_duration:
                legend_entries = []
                pred_idx = 0  # outside both loops

                # GT boxes: already xyxy in orig_size coords from _build_ground_truths_batch
                gt = ground_truths[-1]
                if 'boxes' in gt:
                    gt_boxes = gt['boxes']
                    gt_labels = gt['labels']
                    if isinstance(gt_boxes,  torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
                    if isinstance(gt_labels, torch.Tensor): gt_labels = gt_labels.cpu().numpy()

                    for box, label in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=GT_COLOR, facecolor='none'
                        )
                        ax.add_patch(rect)
                        legend_entries.append((GT_COLOR, "GT"))

                # Predicted boxes: already in orig_size coords from post_process
                if len(results) > 0:
                    for box, score, label in zip(results[-1]['boxes'], results[-1]['scores'], results[-1]['labels']):
                        color = PRED_COLORS[pred_idx % len(PRED_COLORS)]
                        x1, y1, x2, y2 = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        # add text: score and predicted class
                        ax.text(
                            x1, y1 - 5,
                            f"P{pred_idx + 1} {score:.2f} ({label.item()})",
                            fontsize=6, fontfamily='monospace', color=color,
                            va='bottom', ha='left',
                            bbox=dict(
                                boxstyle='round,pad=0.15',
                                facecolor='black', alpha=0.45, linewidth=0,
                            ),
                        )
                        legend_entries.append((color, f"P{pred_idx + 1} {score:.2f}"))
                        pred_idx += 1

                        if pred_idx >= 10:
                            break  # limit to top 10 predictions for visualization

                # ── Forecasting: future GT (verde medio) + future predette (rosso) ──
                N = past_boxes.shape[1]   # query-passato (droni tracciati)

                # Future GT (verde medio, tratteggiate, sfumate nel tempo)
                if future_gt is not None and future_mask is not None:
                    fgt = future_gt[-1].cpu().numpy()     # (M, T, 4) cxcywh norm
                    fm  = future_mask[-1].cpu().numpy()   # (M, T) bool
                    T_gt = fgt.shape[1]
                    drew_gt = False
                    for i in range(fgt.shape[0]):
                        for t in range(T_gt):
                            if not fm[i, t]:
                                continue
                            cx, cy, bw, bh = fgt[i, t]
                            x1, y1 = (cx - bw / 2) * orig_w, (cy - bh / 2) * orig_h
                            a = max(0.2, 1.0 - t / max(T_gt, 1) * 0.8)
                            ax.add_patch(patches.Rectangle(
                                (x1, y1), bw * orig_w, bh * orig_h,
                                linewidth=1.2, edgecolor=FUTURE_GT_COLOR,
                                facecolor='none', linestyle='--', alpha=a))
                            drew_gt = True
                    if drew_gt:
                        legend_entries.append((FUTURE_GT_COLOR, "Future GT"))

                # Future predette dalla forecasting head (rosso, sfumate nel tempo)
                fut_pred = getattr(outputs, 'future_boxes', None)
                if fut_pred is not None:
                    fp = fut_pred[-1, :N].cpu().numpy()   # (N, T, 4) cxcywh norm
                    T_pred = fp.shape[1]
                    for i in range(fp.shape[0]):
                        for t in range(T_pred):
                            cx, cy, bw, bh = fp[i, t]
                            x1, y1 = (cx - bw / 2) * orig_w, (cy - bh / 2) * orig_h
                            a = max(0.2, 1.0 - t / max(T_pred, 1) * 0.8)
                            ax.add_patch(patches.Rectangle(
                                (x1, y1), bw * orig_w, bh * orig_h,
                                linewidth=1.4, edgecolor=FORECAST_COLOR,
                                facecolor='none', alpha=a))
                    if fp.shape[0] > 0:
                        legend_entries.append((FORECAST_COLOR, "Forecast"))

                if legend_entries:
                    row_h = 0.065
                    start_y = 0.98
                    for entry_idx, (color, text) in enumerate(legend_entries):
                        ax.text(
                            0.98, start_y - entry_idx * row_h,
                            f"● {text}",
                            transform=ax.transAxes, fontsize=7,
                            fontfamily='monospace', color=color,
                            va='top', ha='right',
                            bbox=dict(
                                boxstyle='round,pad=0.15',
                                facecolor='black', alpha=0.45, linewidth=0,
                            ),
                        )

            ax.axis('off')

        plt.tight_layout()

        filename = f"step_{self.global_step:06d}_epoch_{epoch + 1:03d}.png"
        save_path = os.path.join(self.output_dir, "visualizations", "train", filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")

        self._set_model_mode()   # ripristina eval()/train() coerente col freeze
