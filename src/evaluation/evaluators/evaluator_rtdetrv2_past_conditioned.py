import torch
import os
from evaluation.evaluators.evaluator_base import BaseEvaluator
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from models.model_factory import get_model
import numpy as np 
from data.dataset_factory import get_dataset
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import _apply_nms
from collections import deque


# tracker autoregressivo (condiviso con il rollout del training)
from evaluation.closed_loop import (   # noqa: F401
    box_iou_cxcywh, box_center_dist, Track, best_match_track, nms_std, duplicate_tracks, LegacyTracker,
)


class TrackingMetrics:
    """MOTA / MOTP / IDF1 / ID-switch (motmetrics) dei track rispetto agli ID della GT."""

    def __init__(self, iou_thr=0.5, img_w=1280, img_h=720):
        try:
            import motmetrics as mm
        except ImportError as e:
            raise ImportError("Le metriche di tracking usano motmetrics: pip install motmetrics") from e
        # np.asfarray non esiste più in NumPy 2.0
        if not hasattr(np, 'asfarray'):
            np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
        self.mm = mm
        self.iou_thr = float(iou_thr)
        self.img_w, self.img_h = img_w, img_h
        self._accs = {}       # nome sequenza -> [MOTAccumulator, ultimo frameid]
        self._cur = None
        self.summary = None   # DataFrame per-sequenza + OVERALL, riempito da compute()

    def new_sequence(self, name=None):
        """Cambio di video; un nome già visto riprende la sua sequenza."""
        if name is None:
            name = f"seq_{len(self._accs):04d}"
        if name not in self._accs:
            self._accs[name] = [self.mm.MOTAccumulator(auto_id=False), 0]
        self._cur = name

    def mot_tlwh(self, box):
        """cxcywh normalizzata -> [left, top, w, h] in pixel (come nei file MOT)."""
        cx, cy, bw, bh = (float(v) for v in box)
        W, H = self.img_w, self.img_h
        return [float(f"{v:.2f}") for v in ((cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H)]

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        """Un frame. boxes: (N, 4) cxcywh normalizzate; ids: sequenze di interi."""
        if self._cur is None:
            self.new_sequence()
        acc_frame = self._accs[self._cur]
        acc_frame[1] += 1
        gt = [self.mot_tlwh(b) for b in gt_boxes]
        pr = [self.mot_tlwh(b) for b in pred_boxes]
        if gt and pr:
            # distanza = 1 - IoU, NaN se IoU < iou_thr
            dist = self.mm.distances.iou_matrix(gt, pr, max_iou=1.0 - self.iou_thr)
        else:
            dist = np.empty((len(gt), len(pr)))
        acc_frame[0].update([int(i) for i in gt_ids], [int(i) for i in pred_ids], dist,
                            frameid=acc_frame[1])

    def compute(self):
        """Riga OVERALL di motmetrics: conteggi sommati su tutte le sequenze."""
        names = list(self._accs)
        if names:
            self.summary = self.mm.metrics.create().compute_many(
                [self._accs[n][0] for n in names], names=names,
                metrics=self.mm.metrics.motchallenge_metrics + ['num_objects', 'num_detections'],
                generate_overall=True,
            )
            ov = self.summary.loc['OVERALL']
        else:
            self.summary = None
            ov = {k: float('nan') for k in ('mota', 'motp', 'idf1', 'idp', 'idr', 'precision', 'recall')}
            ov.update({k: 0 for k in ('num_switches', 'num_fragmentations', 'mostly_tracked',
                                      'partially_tracked', 'mostly_lost', 'num_detections',
                                      'num_false_positives', 'num_misses', 'num_objects',
                                      'num_unique_objects')})
        return {
            'mota':        float(ov['mota']),
            'motp_iou':    1.0 - float(ov['motp']),   # motp di motmetrics = 1 - IoU medio
            'idf1':        float(ov['idf1']),
            'idp':         float(ov['idp']),
            'idr':         float(ov['idr']),
            'id_switches': int(ov['num_switches']),
            'num_fragmentations': int(ov['num_fragmentations']),
            'mostly_tracked':     int(ov['mostly_tracked']),
            'partially_tracked':  int(ov['partially_tracked']),
            'mostly_lost':        int(ov['mostly_lost']),
            'track_tp':    int(ov['num_detections']),   # match + switch
            'track_fp':    int(ov['num_false_positives']),
            'track_fn':    int(ov['num_misses']),
            'track_precision': float(ov['precision']),
            'track_recall':    float(ov['recall']),
            'gt_total':    int(ov['num_objects']),
            'num_unique_objects': int(ov['num_unique_objects']),
            'num_sequences': len(names),
        }

    def render(self):
        """Tabella per sequenza + OVERALL in formato MOTChallenge (dopo compute())."""
        mm = self.mm
        return mm.io.render_summary(self.summary, formatters=mm.metrics.create().formatters,
                                    namemap=mm.io.motchallenge_metric_names)


def collate_autoregressive(batch, image_processor, num_past=None):
    """
    Collate con batch=1 per l'eval autoregressivo (non salta frame). Ritorna
    (frames_dict, present_labels, present_ids, future_gt, future_ids, gt_past, gt_mask);
    gt_past/gt_mask servono solo alla pass oracle.
    """
    assert len(batch) == 1, "batch_size deve essere 1 per l'eval autoregressivo"
    frames_raw, target = batch[0]
    durations = list(frames_raw.keys())

    frames_dict, present_labels = {}, None
    for d in durations:
        img = frames_raw[d]
        img_np = img.permute(1, 2, 0).numpy() if isinstance(img, torch.Tensor) else np.array(img)
        h, w = img_np.shape[:2]

        coco = []
        for box, lbl in zip(
            target.get('boxes',  torch.zeros((0, 4))),
            target.get('labels', torch.zeros(0, dtype=torch.int64)),
        ):
            cx, cy, bw, bh = box.tolist()
            coco.append({
                'bbox':        [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                'category_id': int(lbl),
                'area':        bw * w * bh * h,
                'iscrowd':     0,
                'id':          len(coco),
            })
        enc = image_processor(
            images=[img_np],
            annotations=[{'image_id': 0, 'annotations': coco}],
            return_tensors='pt',
            do_rescale=False,
        )
        frames_dict[d] = enc['pixel_values']
        if present_labels is None:
            present_labels = [
                {'class_labels': l['class_labels'], 'boxes': l['boxes']}
                for l in enc['labels']
            ]

    # ID GT dei droni presenti, nello stesso ordine di present_labels
    present_ids = target.get('ids', torch.zeros(0, dtype=torch.int64))

    # passato GT per la pass oracle (droni presenti in tutti gli step passati)
    past_ids_list = target.get('past_ids', [])
    past_box_list = target.get('past_boxes', [])
    P_past = num_past if num_past is not None else len(past_ids_list)
    if len(past_ids_list) == 0 or len(present_ids) == 0 or P_past == 0:
        valid_ids = []
    else:
        valid_ids = set(present_ids.tolist())
        for pid in past_ids_list:
            valid_ids &= set(pid.tolist())
        valid_ids = sorted(valid_ids)
    max_obj = len(valid_ids)
    gt_past = torch.zeros(1, max_obj, P_past, 4)
    gt_mask = torch.ones(1, max_obj, dtype=torch.bool)
    if max_obj > 0:
        gt_mask[0, :max_obj] = False
        id_to_past = {}
        for p_idx, (pid_t, pb_t) in enumerate(zip(past_ids_list, past_box_list)):
            for k, pid in enumerate(pid_t.tolist()):
                id_to_past.setdefault(pid, {})[p_idx] = pb_t[k]
        for di, vid in enumerate(valid_ids):
            for p_idx in range(P_past):
                gt_past[0, di, p_idx] = id_to_past[vid][p_idx]

    # GT futuro (per la viz autoregressiva): None se num_future_annotations=0
    future_gt  = target.get('future_boxes', None)
    future_ids = target.get('future_ids', None)

    return frames_dict, present_labels, present_ids, future_gt, future_ids, gt_past, gt_mask


def collate_fn_detection_rtdetrv2_past_conditioned(batch, image_processor, config=None):
    """
    Returns:
        frames_dict: Dict[duration_ms --> (B, C, H, W)] normalized tensors
        present_labels: List[Dict{class_labels, boxes}]   
        past_boxes: Tensor(B, max_objects, P, 4) or None
    """
    durations = list(batch[0][0].keys())
    
    phase = getattr(config, "phase", 1)
    
    # Compute valid_ids per sample and filter empty ones 
    valid_batch = []  # list of (frames, target, valid_ids)
    skipped = 0

    for frames_i, target in batch:
        present_ids = target.get('ids', torch.zeros(0, dtype=torch.int64))
        past_ids_list = target.get('past_ids', [])

        if len(present_ids) == 0:
            skipped += 1 # samples with no present gt
            continue

        if len(past_ids_list) == 0 and phase == 1: # rtdetrv2 past conditioned
            skipped += 1
            continue
        
        if len(past_ids_list) == 0:
            valid_ids = []
        elif len(present_ids) == 0:
            valid_ids = []
        else:
            valid_ids = set(present_ids.tolist())
            for pid_tensor in past_ids_list:
                valid_ids &= set(pid_tensor.tolist())
            valid_ids = sorted(valid_ids)

        if len(valid_ids) > 0 or phase == 2:
            valid_batch.append((frames_i, target, valid_ids))
        else:
            skipped += 1


    if len(valid_batch) == 0:
        return None # da gestire

    batch_size = len(valid_batch)
    num_past = len(valid_batch[0][1]['past_ids'])

    # Build frames + present labels (ordered by valid_ids) 
    frames_dict = {}
    present_labels = None

    for d in durations:
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
            id_to_box = {int(pid): box for pid, box in zip(present_ids, present_boxes)}

            # Build coco_anns in valid_ids order so label position i == drone valid_ids[i]
            coco_anns = []
            for vid in valid_ids:
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
    max_objects = max(len(valid_ids) for _, _, valid_ids in valid_batch)
    past_boxes = torch.zeros(batch_size, max_objects, num_past, 4)
    
    past_mask = torch.ones(batch_size, max_objects, dtype=torch.bool)

    for b, (_, target, valid_ids) in enumerate(valid_batch):
        past_mask[b, :len(valid_ids)] = False
        past_ids_list = target['past_ids'] # List of P Tensors (n_t,)
        past_boxes_list = target['past_boxes']  # List of P Tensors (n_t, 4)

        # Build id -> {timestep -> box}
        id_to_past: dict = {}
        for p_idx, (pid_t, pb_t) in enumerate(zip(past_ids_list, past_boxes_list)):
            for k, pid in enumerate(pid_t.tolist()):
                if pid not in id_to_past:
                    id_to_past[pid] = {}
                id_to_past[pid][p_idx] = pb_t[k]

        for drone_idx, vid in enumerate(valid_ids):
            for p_idx in range(num_past):
                past_boxes[b, drone_idx, p_idx] = id_to_past[vid][p_idx]

    # future_gt (B, max_objects, T, 4) e future_mask (B, max_objects, T)
    num_future = getattr(config, "num_future_steps", 0)
    future_gt = None
    future_mask = None
    if num_future > 0:
        future_gt = torch.zeros(batch_size, max_objects, num_future, 4)
        future_mask = torch.zeros(batch_size, max_objects, num_future, dtype=torch.bool)
        for b, (_, target, valid_ids) in enumerate(valid_batch):
            fut_ids_list = target.get('future_ids', [])
            fut_box_list = target.get('future_boxes', [])
            id_to_future: dict = {}
            for t_idx, (fid_t, fb_t) in enumerate(zip(fut_ids_list, fut_box_list)):
                for k, fid in enumerate(fid_t.tolist()):
                    id_to_future.setdefault(fid, {})[t_idx] = fb_t[k]
            for drone_idx, vid in enumerate(valid_ids):
                fut = id_to_future.get(vid, {})
                for t_idx in range(num_future):
                    if t_idx in fut:
                        future_gt[b, drone_idx, t_idx] = fut[t_idx]
                        future_mask[b, drone_idx, t_idx] = True

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


    '''
     Test FRAME NERO
     - Obiettivo: Verificare quanto il modello si affida al passato, piuttosto che alle features del frame corrente, per fare detection
     - Idea: uso box passate reali ma frame corrente (in cui devo detectare) nero
    '''
    # print("TEST 1")
    # for d in durations:
    #     print(f"Frames {d}ms --> torch.zeros")
    #     frames_dict[d] = torch.zeros_like(frames_dict[d])
    
    ''' FINE TEST '''

    '''
     Test SHIFT PASSATO 
     - Obiettivo: Verificare quanto il modello si affida al passato, piuttosto che alle features del frame corrente, per fare detection
     - Idea: shifto le box passate reali cosi che nel frame corrente (in cui devo detectare) il drone sia spostato
    '''
    # print("TEST 2")
    # past_boxes[..., 0] += 0.02
    # past_boxes[..., 1] += 0.1
    
    ''' FINE TEST '''

    
    return {
        'frames': frames_dict,
        'labels': present_labels,
        'past_boxes': past_boxes,
        'past_mask': past_mask,
        'new_labels': new_labels,   # <-- aggiungi
        'future_gt': future_gt,     # (B, max_objects, T, 4)
        'future_mask': future_mask, # (B, max_objects, T)
    }


class PastConditionedEvaluator(BaseEvaluator):

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def evaluate(self):
        from torchvision.ops import box_iou
        """
        Run inference on test set and compute mAP + TP/FP/FN/Precision/Recall.
        """
        self.model.eval()
        self.metric = MeanAveragePrecision(iou_type='bbox')

        total_tp = 0
        total_fp = 0
        total_fn = 0

        iou_thr       = getattr(self.config, 'eval_iou_threshold', 0.5)
        map_score_thr = getattr(self.config, 'map_score_threshold', 0.0)   # ~0 per la mAP (curva PR completa)
        tp_score_thr  = getattr(self.config, 'tp_score_threshold', 0.5)    # soglia per TP/FP/Precision/Recall
        vis_score_thr = getattr(self.config, 'vis_score_threshold', 0.3)   # soglia per la visualizzazione
        query_mode    = self.config.query_mode
        phase         = getattr(self.config, 'phase', 1)

        # diagnostica: best-IoU di ogni GT su tutte le pred (localizzazione vs calibrazione)
        diag_best_iou = int(getattr(self.config, 'diag_best_iou', 0) or 0)
        diag_gt_count = 0          # n. GT totali visti
        diag_iou_sum = 0.0         # somma best-IoU per GT
        diag_score_sum = 0.0       # somma score@best-IoU per GT
        diag_loc_count = 0         # n. GT con best-IoU >= iou_thr
        diag_loc_score_sum = 0.0   # somma score@best-IoU sui GT localizzati
                             
        print("\nStarting evaluation...")
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Inference")):
            
            if batch is None:
                continue
            
            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            past_boxes = batch['past_boxes'].to(self.device)
            past_mask  = batch['past_mask'].to(self.device)

            if query_mode == 'past_only' and past_boxes.shape[1] == 0:
                continue
            
            # '''TEST GT-LAST NERO
            # Usciamo dal caso limite frame completamente nero,
            # il modello riuscira a predire con un certo delta in modo conforme al passato?
            # '''
            # last_boxes = past_boxes[:, :, 0, :]
            # gts = batch['labels']
            #
            # def _clamp(x1, y1, x2, y2, W, H):
            #     x1 = max(0, min(W, int(round(x1)))); x2 = max(0, min(W, int(round(x2))))
            #     y1 = max(0, min(H, int(round(y1)))); y2 = max(0, min(H, int(round(y2))))
            #     return x1, y1, x2, y2
            #
            # for b in range(len(gts)):
            #     for i in range(len(gts[b]['boxes'])):
            #         gcx, gcy, gbw, gbh = gts[b]['boxes'][i].tolist()
            #         lcx, lcy, lbw, lbh = last_boxes[b, i].tolist()
            #
            #         # GT (xyxy norm)
            #         gx1, gy1, gx2, gy2 = gcx-gbw/2, gcy-gbh/2, gcx+gbw/2, gcy+gbh/2
            #         # last completo (xyxy norm)
            #         fx1, fy1, fx2, fy2 = lcx-lbw/2, lcy-lbh/2, lcx+lbw/2, lcy+lbh/2
            #         # meta di last da PROTEGGERE = lato opposto al gt, taglio al centro sull'asse dominante
            #         tx1, ty1, tx2, ty2 = fx1, fy1, fx2, fy2
            #         dx, dy = gcx-lcx, gcy-lcy
            #         if abs(dx) >= abs(dy):
            #             if dx >= 0: tx2 = lcx     # gt a destra -> proteggo meta sinistra
            #             else:       tx1 = lcx     # gt a sinistra -> proteggo meta destra
            #         else:
            #             if dy >= 0: ty2 = lcy     # gt sotto -> proteggo meta alta
            #             else:       ty1 = lcy     # gt sopra -> proteggo meta bassa
            #
            #         for d in frames_dict:
            #             _, _, H, W = frames_dict[d].shape
            #             GX1,GY1,GX2,GY2 = _clamp(gx1*W,gy1*H,gx2*W,gy2*H,W,H)
            #             if GX2<=GX1 or GY2<=GY1:
            #                 continue
            #             FX1,FY1,FX2,FY2 = _clamp(fx1*W,fy1*H,fx2*W,fy2*H,W,H)  # last intero
            #             TX1,TY1,TX2,TY2 = _clamp(tx1*W,ty1*H,tx2*W,ty2*H,W,H)  # meta di coda (protetta)
            #
            #             if TX2>TX1 and TY2>TY1:
            #                 patch = frames_dict[d][b,:,TY1:TY2,TX1:TX2].clone()   # salvo la meta di coda
            #                 frames_dict[d][b,:,GY1:GY2,GX1:GX2] = 0               # azzero il GT
            #                 if FX2>FX1 and FY2>FY1:
            #                     frames_dict[d][b,:,FY1:FY2,FX1:FX2] = 0           # azzero tutto il last
            #                 frames_dict[d][b,:,TY1:TY2,TX1:TX2] = patch           # ripristino la meta di coda
            #             else:
            #                 frames_dict[d][b,:,GY1:GY2,GX1:GX2] = 0
            # ''' FINE TEST GT-LAST NERO '''

            # Forward pass
            outputs = self.model(
                event_frames=frames_dict,
                past_boxes=past_boxes,
                past_mask=past_mask,
                query_mode=query_mode,
            )

            # slot di padding delle query-passato: score 0, altrimenti il post-process li tiene come FP
            if (phase == 1 or query_mode in ('both', 'past_only')) and past_mask.any():
                n_past = past_boxes.shape[1]   # slot passato = [0:N] (in 'both' le standard sono [N:])
                logits = outputs.logits.clone()
                logits[:, :n_past] = logits[:, :n_past].masked_fill(past_mask.unsqueeze(-1), float('-inf'))
                outputs.logits = logits

            # DEBUG: score grezzi delle query
            if batch_idx % 200 == 0 and outputs.logits is not None and outputs.logits.numel() > 0:
                _sc = outputs.logits.sigmoid()
                print(f"[dbg {query_mode}] max score: {float(_sc.max()):.3f} | "
                      f"n>0.1: {int((_sc > 0.1).sum())} | n>0.5: {int((_sc > 0.5).sum())} | "
                      f"queries: {tuple(_sc.shape)}")

            # GT = droni con passato + droni nuovi (in phase 1 solo quelli con passato)
            targets = batch['labels']
            new_labels = batch.get('new_labels', None)
            # if query_mode in ('both', 'standard_only') and new_labels is not None: # in past_only mode il test set e diverso e forse non e comparabile con le altre due modalita
            if phase == 2 and query_mode in ('both', 'standard_only', 'past_only') and new_labels is not None:
                merged = []
                for b in range(len(batch['labels'])):
                    t = dict(batch['labels'][b])   # preserva eventuali altri campi (es. orig_size)
                    t['class_labels'] = torch.cat([batch['labels'][b]['class_labels'],
                                                   new_labels[b]['class_labels']])
                    t['boxes'] = torch.cat([batch['labels'][b]['boxes'],
                                            new_labels[b]['boxes']])
                    merged.append(t)
                targets = merged

            # Ground truths in xyxy pixel space
            ground_truths_batch = self._build_ground_truths_batch(targets)

            target_sizes = torch.tensor(
                [(720, 1280) for gt in ground_truths_batch]
            ).to(self.device)

            results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=map_score_thr,   # soglia bassa: la mAP integra tutta la curva PR
            )

            if self.config.use_nms:
                results = _apply_nms(results, iou_threshold=self.config.nms_threshold)

            for pred, gt in zip(results, ground_truths_batch):
                # diagnostica best-IoU
                if diag_best_iou:
                    gtb = gt['boxes']
                    pbx = pred['boxes']
                    if gtb.shape[0] > 0:
                        if pbx.shape[0] == 0:
                            diag_gt_count += int(gtb.shape[0])   # nessuna pred: best-IoU = 0
                        else:
                            iou_d = box_iou(gtb.cpu().float(), pbx.cpu().float())   # (n_gt, n_pred)
                            best_iou_d, best_idx_d = iou_d.max(dim=1)               # (n_gt,)
                            best_score_d = pred['scores'].cpu()[best_idx_d]         # score della box a IoU max
                            loc_d = best_iou_d >= iou_thr
                            diag_gt_count      += int(gtb.shape[0])
                            diag_iou_sum       += float(best_iou_d.sum())
                            diag_score_sum     += float(best_score_d.sum())
                            diag_loc_count     += int(loc_d.sum())
                            diag_loc_score_sum += float(best_score_d[loc_d].sum())

                # per TP/FP/Precision/Recall usa una soglia più alta;
                # la mAP invece usa tutte le predizioni (results, soglia ~0).
                keep = pred['scores'] >= tp_score_thr
                pred_boxes = pred['boxes'][keep]   # [N, 4] xyxy assoluto
                gt_boxes = gt['boxes']             # [M, 4] xyxy assoluto

                n_pred = len(pred_boxes)
                n_gt = len(gt_boxes)

                if n_pred == 0 and n_gt == 0:
                    continue                         
                elif n_pred == 0:
                    total_fn += n_gt # tutto mancato
                    continue
                elif n_gt == 0:
                    total_fp += n_pred # tutto falso positivo
                    continue

                # [N_pred, N_gt]
                iou_matrix = box_iou(pred_boxes.cpu(), gt_boxes.cpu())

                matched_gt = set()
                matched_pred = set()

                sorted_iou = (iou_matrix >= iou_thr).nonzero(as_tuple=False)
                iou_vals = [(iou_matrix[i, j].item(), i.item(), j.item())
                            for i, j in sorted_iou]
                iou_vals.sort(key=lambda x: -x[0])

                for iou_val, i_pred, i_gt in iou_vals:
                    if i_pred not in matched_pred and i_gt not in matched_gt:
                        matched_pred.add(i_pred)
                        matched_gt.add(i_gt)

                tp = len(matched_pred)
                fp = n_pred - tp
                fn = n_gt   - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn

            if batch_idx % self.config.vis_every_n_batches == 0:
                self._save_visualization(
                    frames_dict=frames_dict,
                    results=results,
                    ground_truths=ground_truths_batch,
                    batch_idx=batch_idx,
                    sample_idx=0,
                    past_boxes=past_boxes,
                    vis_score_thr=vis_score_thr,
                )

            gt_for_metric = [
                {'boxes': gt['boxes'], 'labels': gt['labels']}
                for gt in ground_truths_batch
            ]
            self.metric.update(results, gt_for_metric)

        print("\nComputing final mAP score...")
        final_results = self.metric.compute()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        print("\n" + "=" * 50)
        print(f"  IoU threshold per TP/FP/FN : {iou_thr}")
        print(f"  Score thr (mAP / TP-FP)    : {map_score_thr} / {tp_score_thr}")
        print(f"  Query mode                 : {query_mode}")
        print("-" * 50)
        print(f"  True  Positives (TP) : {total_tp}")
        print(f"  False Positives (FP) : {total_fp}  ← rilevazioni sbagliate")
        print(f"  False Negatives (FN) : {total_fn}  ← droni mancati")
        print("-" * 50)
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print("=" * 50)

        # diagnostica best-IoU
        if diag_best_iou and diag_gt_count > 0:
            mean_best_iou      = diag_iou_sum / diag_gt_count
            mean_score_at_best = diag_score_sum / diag_gt_count
            pct_localized      = diag_loc_count / diag_gt_count
            mean_score_loc     = diag_loc_score_sum / max(diag_loc_count, 1)
            print("\n" + "=" * 50)
            print(f"  [DIAG best-IoU]  soglia-free su TUTTE le pred | query_mode={query_mode}")
            print("-" * 50)
            print(f"  GT totali                    : {diag_gt_count}")
            print(f"  mean best-IoU per GT         : {mean_best_iou:.3f}")
            print(f"  mean score @ best-IoU box    : {mean_score_at_best:.3f}")
            print(f"  % GT localizzati (IoU>={iou_thr:.2f}) : {100*pct_localized:.1f}%")
            print(f"  mean score @ GT localizzati  : {mean_score_loc:.3f}")
            print("-" * 50)
            print("  IoU alta + score basso  => CALIBRAZIONE (le box ci sono, score sotto soglia)")
            print("  IoU/localizzati bassi    => LOCALIZZAZIONE (ramo non allenato)")
            print("=" * 50)
            final_results['diag_mean_best_iou']        = mean_best_iou
            final_results['diag_mean_score_at_best']   = mean_score_at_best
            final_results['diag_pct_localized']        = pct_localized
            final_results['diag_mean_score_localized'] = mean_score_loc

        import pprint
        pprint.pprint(final_results)

        final_results['tp'] = total_tp
        final_results['fp'] = total_fp
        final_results['fn'] = total_fn
        final_results['precision'] = precision
        final_results['recall'] = recall
        final_results['f1'] = f1

        return final_results

    @torch.no_grad()
    def evaluate_forecasting(self):
        """ADE/FDE (in pixel) + IoU sulle box future, sulle query-passato."""
        self.model.eval()
        query_mode = getattr(self.config, 'query_mode', 'both')
        W = getattr(self.config, 'img_width', 1280)
        H = getattr(self.config, 'img_height', 720)
        T = getattr(self.config, 'num_future_steps', 0)
        if T <= 0:
            raise RuntimeError("num_future_steps deve essere > 0 per evaluate_forecasting")
        scale = torch.tensor([W, H], device=self.device, dtype=torch.float32)

        def to_xyxy(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

        def piou(a, b):
            x1 = torch.max(a[:, 0], b[:, 0]); y1 = torch.max(a[:, 1], b[:, 1])
            x2 = torch.min(a[:, 2], b[:, 2]); y2 = torch.min(a[:, 3], b[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
            area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
            return inter / (area_a + area_b - inter).clamp(min=1e-6)

        total_disp = 0.0; total_cnt = 0
        fde_disp = 0.0;   fde_cnt = 0
        step_disp = [0.0] * T; step_cnt = [0] * T
        total_iou = 0.0;  iou_cnt = 0

        print("\nStarting FORECASTING evaluation...")
        vis_every = getattr(self.config, 'vis_every_n_batches', 200)
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Forecast inference")):
            if batch is None:
                continue
            future_gt = batch.get('future_gt', None)
            if future_gt is None or batch['past_boxes'].shape[1] == 0:
                continue

            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            past_boxes = batch['past_boxes'].to(self.device)
            past_mask  = batch['past_mask'].to(self.device)
            future_gt  = future_gt.to(self.device)                  # (B, M, T, 4)
            future_mask = batch['future_mask'].to(self.device)      # (B, M, T)

            outputs = self.model(
                event_frames=frames_dict, past_boxes=past_boxes,
                past_mask=past_mask, query_mode=query_mode,
            )
            if outputs.future_boxes is None:
                raise RuntimeError("future_boxes è None: crea il modello eval con num_future_steps>0")

            M = future_gt.shape[1]
            fp = outputs.future_boxes[:, :M]                        # (B, M, T, 4), query-passato

            # visualizzazione
            if vis_every and (batch_idx % vis_every == 0):
                self._save_visualization_forecast(
                    frames_dict, past_boxes, future_gt, fp, future_mask, batch_idx,
                )

            disp = ((fp[..., :2] - future_gt[..., :2]) * scale).norm(dim=-1)   # (B, M, T) px

            m = future_mask
            if not m.any():
                continue
            total_disp += disp[m].sum().item(); total_cnt += int(m.sum().item())
            for t in range(T):
                mt = m[:, :, t]
                if mt.any():
                    s = disp[:, :, t][mt].sum().item(); c = int(mt.sum().item())
                    step_disp[t] += s; step_cnt[t] += c
                    if t == T - 1:
                        fde_disp += s; fde_cnt += c
            pred_xy = to_xyxy(fp[m]); gt_xy = to_xyxy(future_gt[m])
            total_iou += piou(pred_xy, gt_xy).sum().item(); iou_cnt += pred_xy.shape[0]

        ade = total_disp / max(total_cnt, 1)
        fde = fde_disp / max(fde_cnt, 1)
        mean_iou = total_iou / max(iou_cnt, 1)
        per_step = [step_disp[t] / max(step_cnt[t], 1) for t in range(T)]

        # ADE/FDE per orizzonte (0.4 s e 0.8 s), step nominale di 33.3 ms
        step_dt = float(getattr(self.config, 'forecast_step_dt_s', 1.0 / 30.0))   # s/step
        horizons = {                                   # orizzonte in secondi
            'short_0_4s': float(getattr(self.config, 'ade_short_horizon_s', 0.4)),
            'mid_0_8s':   float(getattr(self.config, 'ade_mid_horizon_s',   0.8)),
        }

        def _horizon_metrics(H_s):
            n = min(T, max(1, int(round(H_s / step_dt))))        # n. step entro H (cap a T)
            disp = sum(step_disp[:n]); cnt = sum(step_cnt[:n])
            ade_h = disp / max(cnt, 1)
            fde_h = step_disp[n - 1] / max(step_cnt[n - 1], 1)   # errore all'ultimo step entro H
            covered = (n * step_dt) + 1e-9 >= H_s                # T copre davvero H?
            return ade_h, fde_h, n, covered

        horizon_out = {}
        for name, H_s in horizons.items():
            ade_h, fde_h, n_h, cov = _horizon_metrics(H_s)
            horizon_out[f'ade_{name}'] = ade_h
            horizon_out[f'fde_{name}'] = fde_h
            horizon_out[f'nsteps_{name}'] = n_h
            horizon_out[f'covered_{name}'] = cov

        full_horizon_s = T * step_dt

        print("\n" + "=" * 50)
        print("  FORECASTING")
        print(f"  query_mode          : {query_mode}")
        print(f"  coppie valutate     : {total_cnt}")
        print(f"  step_dt | T | span  : {step_dt*1000:.2f} ms/step | T={T} | full = {full_horizon_s:.3f} s")
        print(f"  ADE full      (px)  : {ade:.2f}   [0–{full_horizon_s:.3f}s, {T} step]")
        print(f"  FDE full      (px)  : {fde:.2f}   [@ t{T}]")
        for name, H_s in horizons.items():
            n_h = horizon_out[f'nsteps_{name}']; cov = horizon_out[f'covered_{name}']
            warn = '' if cov else f'  ⚠️ T={T} copre solo {full_horizon_s:.3f}s < {H_s:.1f}s → = full'
            print(f"  ADE {name:10s}(px) : {horizon_out[f'ade_{name}']:.2f}   [0–{H_s:.1f}s → {n_h} step]{warn}")
            print(f"  FDE {name:10s}(px) : {horizon_out[f'fde_{name}']:.2f}   [@ t{n_h}]")
        print(f"  mean future IoU     : {mean_iou:.4f}")
        print("  per-step ADE px     : " + "  ".join(f"t{t+1}={per_step[t]:.1f}" for t in range(T)))
        print("=" * 50)

        result = {
            'ade': ade, 'fde': fde, 'ade_full': ade, 'fde_full': fde,
            'mean_iou': mean_iou, 'per_step_ade': per_step,
            'step_dt_s': step_dt, 'full_horizon_s': full_horizon_s,
        }
        result.update(horizon_out)   # ade_short_0_4s, fde_short_0_4s, ade_mid_0_8s, fde_mid_0_8s, ...
        return result

    def _ar_forward(self, frames_dict, past_boxes, past_mask):
        """Forward AR di un frame, una passata in 'both' (la variante 2-pass la sovrascrive)."""
        return self.model(
            event_frames=frames_dict,
            past_boxes=past_boxes,
            past_mask=past_mask,
            query_mode='both',
        )

    @torch.no_grad()
    def evaluate_autoregressive(self):
        from torchvision.ops import box_iou, nms
        """
        Eval closed-loop (batch=1, sequenziale): il passato viene dai track del modello, che si
        resettano a ogni sequenza. Metriche come evaluate() sulle detection del tracker.
        """
        self.model.eval()
        self.metric = MeanAveragePrecision(iou_type='bbox')
        self.metric_fullpr = MeanAveragePrecision(iou_type='bbox')   # mAP su tutti gli output, come evaluate()

        # parametri
        iou_thr_eval = getattr(self.config, 'eval_iou_threshold', 0.5)
        tp_score_thr = getattr(self.config, 'tp_score_threshold', 0.5)
        map_score_thr = getattr(self.config, 'map_score_threshold', 0.0)   # come evaluate(): curva PR completa
        W = getattr(self.config, 'img_width', 1280)
        H = getattr(self.config, 'img_height', 720)
        P = getattr(self.config, 'num_past_steps',
                    getattr(self.config, 'num_past_annotations', 10))
        conf_thr  = getattr(self.config, 'ar_conf_thr', 0.35)
        coher_thr = getattr(self.config, 'ar_coher_thr', 0.10)
        # soglia sulla conf media della finestra; -1 = conf_thr
        _mc = getattr(self.config, 'ar_mean_conf_thr', -1.0)
        mean_conf_thr = conf_thr if _mc < 0 else _mc
        # 1 = vince la box standard
        std_box_priority = bool(getattr(self.config, 'ar_std_box_priority', 0))
        iou_thr   = getattr(self.config, 'ar_iou_thr', 0.2)
        dist_thr  = getattr(self.config, 'ar_dist_thr', 0.08)
        max_missed = getattr(self.config, 'ar_max_missed', 6)
        # scarta le standard sovrapposte a una box del passato
        std_past_iou_thr = getattr(self.config, 'ar_std_past_iou_thr', 0.3)
        # NMS del tracker (stessa soglia di evaluate())
        nms_thr = getattr(self.config, 'nms_threshold', 0.45)
        # forecast t+1 come reference point della query-passato (opzionale)
        use_fc_ref = bool(getattr(self.config, 'ar_use_forecast_refpoint', 0))
        # oracle: passato = GT (limite superiore), gli ID vengono comunque dal tracker
        oracle = bool(getattr(self.config, 'ar_oracle_past', 0))
        # 'legacy' | 'tracktrack' (evaluation/tracktrack_ar.py)
        tracker_type = str(getattr(self.config, 'ar_tracker', 'legacy')).lower()
        use_tt = tracker_type == 'tracktrack'
        if tracker_type not in ('legacy', 'tracktrack'):
            raise ValueError(f"ar_tracker sconosciuto: {tracker_type!r} (legacy | tracktrack)")
        if use_tt and oracle:
            raise ValueError("ar_tracker='tracktrack' non supporta ar_oracle_past=1: usa il tracker legacy")
        if use_tt:
            from evaluation.tracktrack_ar import TrackTrackAR, TTConfig
            tt_cfg = TTConfig.from_config(self.config)
            tt = TrackTrackAR(tt_cfg, P)
        # tracker legacy (evaluation/closed_loop.py)
        tracker = LegacyTracker(P, conf_thr=conf_thr, mean_conf_thr=mean_conf_thr, coher_thr=coher_thr,
                                iou_thr=iou_thr, dist_thr=dist_thr, max_missed=max_missed,
                                std_past_iou_thr=std_past_iou_thr, nms_thr=nms_thr,
                                std_box_priority=std_box_priority, use_fc_ref=use_fc_ref, oracle=oracle)

        # loader sequenziale con batch=1
        dataset = self.test_dataset
        # il tracker richiede frame consecutivi
        ar_subsample = getattr(self.config, 'subsample', 1)
        if ar_subsample != 1:
            raise RuntimeError(
                f"evaluate_autoregressive richiede subsample=1 (attuale: {ar_subsample}). "
                f"Con subsample>1 la storia dei track ha passo temporale errato → mAP/MOTA/IDF1 "
                f"autoregressivi silenziosamente sbagliati. Rilancia la pass AR con --subsample 1."
            )
        windows = getattr(dataset, 'windows', None)
        if windows is None:
            raise RuntimeError("Il dataset non espone .windows: impossibile rilevare i confini di sequenza.")

        # subsample per sequenza (frame intatti); ar_seq_offset per sottoinsiemi disgiunti
        seq_stride = int(getattr(self.config, 'ar_seq_subsample', 1) or 1)
        seq_offset = int(getattr(self.config, 'ar_seq_offset', 0) or 0)
        if seq_stride > 1 or seq_offset > 0:
            from torch.utils.data import Subset
            seen = list(dict.fromkeys(w['hdf5_path'] for w in windows))   # sequenze uniche, ordine di apparizione
            kept = set(seen[seq_offset::seq_stride])
            keep_idx = [i for i, w in enumerate(windows) if w['hdf5_path'] in kept]
            windows = [windows[i] for i in keep_idx]                      # allineato al Subset
            dataset = Subset(dataset, keep_idx)
            print(f"[ar_seq_subsample stride={seq_stride} offset={seq_offset}] "
                  f"tengo {len(kept)}/{len(seen)} sequenze → {len(keep_idx)} frame")

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=lambda b: collate_autoregressive(b, self.image_processor, P),
            pin_memory=True,
        )

        def to_xyxy_pix(boxes):  # (N,4) cxcywh norm -> (N,4) xyxy pixel
            if boxes.numel() == 0:
                return boxes.reshape(0, 4)
            cx, cy, bw, bh = boxes.unbind(-1)
            return torch.stack([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H], dim=-1)

        total_tp = total_fp = total_fn = 0
        tracks = []
        prev_seq = None
        # metriche di tracking (motmetrics)
        track_metrics = TrackingMetrics(iou_thr=iou_thr_eval, img_w=W, img_h=H)

        # diagnostica: frame con query-passato e motivi per cui i track non sono validi
        diag = {'frames': 0, 'with_valid': 0, 'no_tracks': 0, 'tracks_but_none_valid': 0,
                'valid_sum': 0, 'fail_len': 0, 'fail_conf': 0, 'fail_coher': 0,
                'std_nms': 0, 'dup_tracks': 0}   # standard scartate dalla NMS / track doppioni eliminati

        # diagnostica: FP per ramo di provenienza (past / standard), solo conteggi
        diag_fp_source = bool(getattr(self.config, 'diag_fp_source', 0))
        emit_src = {'past': 0, 'standard': 0}   # det emesse (score>=tp_score_thr) per ramo
        tp_src   = {'past': 0, 'standard': 0}   # di quelle, matchate a un GT
        fp_src   = {'past': 0, 'standard': 0}   # di quelle, non matchate (FP)

        # export MOTChallenge per eval_tracker.py: un .txt per sequenza in mot_tracks/{pred,gt}/
        export_mot = bool(getattr(self.config, 'ar_export_mot', 0))
        _mot_sub = ('mot_tracks_oracle' if oracle else
                    'mot_tracks_tracktrack' if use_tt else 'mot_tracks')   # non sovrascrivere le altre pass
        mot_pred, mot_gt, mot_frame = {}, {}, {}
        # nome univoco per sequenza (il basename dell'hdf5 è uguale per tutte)
        seq_name_map = {}
        if export_mot:
            mot_pred_dir = os.path.join(self.output_dir, _mot_sub, 'pred')
            mot_gt_dir   = os.path.join(self.output_dir, _mot_sub, 'gt')
            os.makedirs(mot_pred_dir, exist_ok=True)
            os.makedirs(mot_gt_dir, exist_ok=True)

        def _why_invalid(t):
            """Prima condizione di is_valid che fallisce, per il breakdown diagnostico."""
            if len(t.boxes) < P:
                return 'fail_len'
            if float(np.mean(t.scores)) < mean_conf_thr:
                return 'fail_conf'
            return 'fail_coher'

        print("\nStarting AUTOREGRESSIVE evaluation...")
        for idx, batch in enumerate(tqdm(loader, desc="Autoreg inference")):
            # nuova sequenza: reset del tracker
            seq_id = windows[idx]['hdf5_path']
            if seq_id not in seq_name_map:
                seq_name_map[seq_id] = f"seq_{len(seq_name_map):04d}"
            seq_name = seq_name_map[seq_id]   # stesso nome per l'accumulatore motmetrics e il file MOT
            if seq_id != prev_seq:
                tracks = []
                tracker.reset()
                if use_tt:
                    tt.reset()
                track_metrics.new_sequence(seq_name)
                prev_seq = seq_id

            frames_dict, present_labels, present_ids, future_gt, future_ids, gt_past, gt_mask = batch
            frames_dict = {d: f.to(self.device) for d, f in frames_dict.items()}

            # passato dato al modello
            diag['frames'] += 1
            if oracle:
                valid = None
                past_boxes = gt_past.to(self.device)
                past_mask  = gt_mask.to(self.device)
            elif use_tt:
                # track confermati, anche persi da pochi frame
                valid = tt.query_tracks()
                diag['valid_sum'] += len(valid)
                if valid:
                    diag['with_valid'] += 1
                elif not tt.tracks:
                    diag['no_tracks'] += 1
                else:
                    diag['tracks_but_none_valid'] += 1
                    for r, n in tt.query_breakdown().items():
                        diag[r] = diag.get(r, 0) + n
                past_boxes = torch.from_numpy(tt.query_array(valid)).float().unsqueeze(0).to(self.device)
                past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=self.device)
            else:
                # finestre dei track validi
                valid = tracker.query_tracks()
                diag['valid_sum'] += len(valid)
                if valid:
                    diag['with_valid'] += 1
                elif not tracker.tracks:
                    diag['no_tracks'] += 1              # nessun track: problema di detection
                else:
                    diag['tracks_but_none_valid'] += 1  # track esistono ma il gate li rifiuta
                    for t in tracker.tracks:
                        diag[_why_invalid(t)] += 1
                past_boxes = torch.from_numpy(tracker.query_array(valid)).float().unsqueeze(0).to(self.device)
                past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=self.device)

            outputs = self._ar_forward(frames_dict, past_boxes, past_mask)

            if outputs.pred_boxes.shape[1] > 0:
                scores = outputs.logits[0].sigmoid().squeeze(-1)   # (Q,)
                pred_boxes = outputs.pred_boxes[0]                 # (Q, 4)
            else:
                scores = torch.zeros(0, device=self.device)
                pred_boxes = torch.zeros(0, 4, device=self.device)

            # update del tracker
            N = past_boxes.shape[1]
            past_b = pred_boxes[:N].cpu().numpy(); past_s = scores[:N].cpu().numpy()
            std_b  = pred_boxes[N:].cpu().numpy(); std_s  = scores[N:].cpu().numpy()
            if use_tt:
                fut = (outputs.future_boxes[0, :N].detach().cpu().numpy()
                       if outputs.future_boxes is not None and N > 0 else None)
                emb_p = emb_s = None
                if tt_cfg.w_app > 0 and getattr(outputs, 'hidden_states', None) is not None:
                    hs = outputs.hidden_states[0].detach().float().cpu().numpy()
                    emb_p, emb_s = hs[:N], hs[N:]
                current_dets, current_dets_src = tt.step(past_b, past_s, std_b, std_s, valid, fut, emb_p, emb_s)
                tracks = tt.tracks          # per le visualizzazioni (stessa interfaccia: .boxes)
            else:
                # fc_next: forecast t+1 dei track validi (per use_fc_ref)
                fc_next = (outputs.future_boxes[0, :N, 0].detach().cpu().numpy()
                           if (not oracle and outputs.future_boxes is not None and N > 0) else None)
                current_dets, current_dets_src, _st = tracker.step(past_b, past_s, std_b, std_s, valid, fc_next)
                diag['std_nms'] += _st['std_nms']
                diag['dup_tracks'] += _st['dup_tracks']
                tracks = tracker.tracks

            # visualizzazione
            if idx % getattr(self.config, 'vis_every_n_batches', 200) == 0:
                out_dir = os.path.join(self.output_dir, "visualizations", "test_autoregressive")
                # forecast dei track validi (con eval_forecasting=1)
                future_pred = None
                if getattr(self.config, 'eval_forecasting', 0) and outputs.future_boxes is not None and N > 0:
                    future_pred = outputs.future_boxes[0, :N].detach().cpu().numpy()   # (N, T, 4)
                self._save_visualization_autoregressive(
                    frames_dict, present_labels, tracks, current_dets, idx, out_dir,
                    future_pred=future_pred, future_gt=future_gt, future_ids=future_ids,
                )

            # metriche di tracking sulle stesse detection della mAP
            gt_boxes_norm = present_labels[0]['boxes'].cpu().numpy() if present_labels else np.zeros((0, 4))
            gt_ids_np = (present_ids.cpu().numpy() if torch.is_tensor(present_ids)
                         else np.asarray(present_ids))
            n_pair = min(len(gt_ids_np), gt_boxes_norm.shape[0])   # robustezza se il processor filtra
            track_metrics.update(
                gt_ids=gt_ids_np[:n_pair],
                gt_boxes=gt_boxes_norm[:n_pair],
                pred_ids=[d[2] for d in current_dets],
                pred_boxes=[d[0] for d in current_dets],
            )

            # export MOT del frame (pixel left, top, w, h)
            if export_mot:
                fno = mot_frame.get(seq_name, 0) + 1
                mot_frame[seq_name] = fno
                pred_l = mot_pred.setdefault(seq_name, [])
                for b, s, tid in current_dets:
                    x, y, w, h = track_metrics.mot_tlwh(b)
                    pred_l.append(f"{fno},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(s):.4f},-1,-1,-1")
                gt_l = mot_gt.setdefault(seq_name, [])
                for i in range(n_pair):
                    x, y, w, h = track_metrics.mot_tlwh(gt_boxes_norm[i])
                    gt_l.append(f"{fno},{int(gt_ids_np[i])},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

            # metriche: current_dets vs droni presenti
            if current_dets:
                pred_boxes_xyxy = to_xyxy_pix(torch.tensor(
                    np.stack([d[0] for d in current_dets], axis=0), dtype=torch.float32))
                pred_scores = torch.tensor([d[1] for d in current_dets], dtype=torch.float32)
            else:
                pred_boxes_xyxy = torch.zeros(0, 4)
                pred_scores = torch.zeros(0)
            pred_labels = torch.zeros(len(current_dets), dtype=torch.long)

            gt_boxes_xyxy = to_xyxy_pix(present_labels[0]['boxes'].float())
            gt_labels = torch.zeros(gt_boxes_xyxy.shape[0], dtype=torch.long)

            # mAP (operating point: current_dets sono già > conf_thr)
            self.metric.update(
                [{'boxes': pred_boxes_xyxy, 'scores': pred_scores, 'labels': pred_labels}],
                [{'boxes': gt_boxes_xyxy, 'labels': gt_labels}],
            )

            # mAP full-PR: tutti gli output del modello, con soglia e NMS di evaluate()
            if pred_boxes.shape[0] > 0:
                _allb = to_xyxy_pix(pred_boxes.detach().cpu())
                _alls = scores.detach().cpu().reshape(-1)
                _k = _alls > map_score_thr
                _allb, _alls = _allb[_k], _alls[_k]
                if self.config.use_nms and _alls.numel() > 0:
                    _keep = nms(_allb.float(), _alls.float(), self.config.nms_threshold)
                    _allb, _alls = _allb[_keep], _alls[_keep]
            else:
                _allb, _alls = torch.zeros(0, 4), torch.zeros(0)
            self.metric_fullpr.update(
                [{'boxes': _allb, 'scores': _alls, 'labels': torch.zeros(_alls.numel(), dtype=torch.long)}],
                [{'boxes': gt_boxes_xyxy, 'labels': gt_labels}],
            )

            # TP/FP/FN a tp_score_thr (greedy IoU, come in evaluate())
            keep = pred_scores >= tp_score_thr
            pb_keep = pred_boxes_xyxy[keep]
            n_pred, n_gt = len(pb_keep), len(gt_boxes_xyxy)
            # ramo di provenienza delle det tenute, in ordine pb_keep
            src_keep = ([current_dets_src[i] for i, k in enumerate(keep.tolist()) if k]
                        if diag_fp_source else None)
            if n_pred == 0 and n_gt == 0:
                pass
            elif n_pred == 0:
                total_fn += n_gt
            elif n_gt == 0:
                total_fp += n_pred
                if diag_fp_source:
                    for s in src_keep:                       # nessun GT: tutte FP
                        emit_src[s] += 1; fp_src[s] += 1
            else:
                iou_matrix = box_iou(pb_keep, gt_boxes_xyxy)
                matched_pred, matched_gt = set(), set()
                pairs = [(iou_matrix[i, j].item(), i.item(), j.item())
                         for i, j in (iou_matrix >= iou_thr_eval).nonzero(as_tuple=False)]
                pairs.sort(key=lambda x: -x[0])
                for _, ip, ig in pairs:
                    if ip not in matched_pred and ig not in matched_gt:
                        matched_pred.add(ip); matched_gt.add(ig)
                tp = len(matched_pred)
                total_tp += tp
                total_fp += n_pred - tp
                total_fn += n_gt - tp
                if diag_fp_source:
                    for ip in range(n_pred):
                        s = src_keep[ip]; emit_src[s] += 1
                        if ip in matched_pred: tp_src[s] += 1
                        else:                  fp_src[s] += 1

        # risultati
        print("\nComputing final mAP score (autoregressive)...")
        final_results = self.metric.compute()
        final_fullpr = self.metric_fullpr.compute()
        tm = track_metrics.compute()   # riga OVERALL di motmetrics

        # mAP full-PR e mAP del sistema (current_dets)
        _mp = lambda d, k: float(d[k]) if k in d else float('nan')
        print("\n  mAP AR FULL-PR (tutti gli output, soglia ~0 + NMS come evaluate() → confrontabile con standard/both):")
        print(f"    map={_mp(final_fullpr,'map'):.4f}  map50={_mp(final_fullpr,'map_50'):.4f}  map75={_mp(final_fullpr,'map_75'):.4f}")
        print("  mAP AR system (solo current_dets > conf_thr → punto operativo del tracker):")
        print(f"    map={_mp(final_results,'map'):.4f}  map50={_mp(final_results,'map_50'):.4f}  map75={_mp(final_results,'map_75'):.4f}")
        final_results['map_fullpr']   = _mp(final_fullpr, 'map')
        final_results['map50_fullpr'] = _mp(final_fullpr, 'map_50')
        final_results['map75_fullpr'] = _mp(final_fullpr, 'map_75')

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        print("\n" + "=" * 50)
        _mode_str = ("AUTOREGRESSIVE (oracle-past)" if oracle else
                     "AUTOREGRESSIVE (closed-loop, TrackTrack-AR)" if use_tt else "AUTOREGRESSIVE (closed-loop)")
        print(f"  MODE                       : {_mode_str}")
        print(f"  IoU threshold per TP/FP/FN : {iou_thr_eval}")
        print(f"  conf_thr / mean_conf_thr   : {conf_thr} / {mean_conf_thr}   (detection corrente / media finestra)")
        print(f"  tp_score_thr               : {tp_score_thr}")
        print(f"  P / max_missed             : {P} / {max_missed}")
        print(f"  NMS tracker (IoU)          : {nms_thr}   (standard doppie + track doppioni)")
        print("-" * 50)
        print(f"  True  Positives (TP) : {total_tp}")
        print(f"  False Positives (FP) : {total_fp}")
        print(f"  False Negatives (FN) : {total_fn}")
        print("-" * 50)
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print("-" * 50)
        if diag_fp_source:
            tot_fp_s = max(fp_src['past'] + fp_src['standard'], 1)
            print(f"  FP-BY-SOURCE (da quale ramo vengono gli FP)  [tot FP: {total_fp}]")
            for s in ('past', 'standard'):
                print(f"  {s:8s} : emesse={emit_src[s]}  TP={tp_src[s]}  "
                      f"FP={fp_src[s]}  ({100*fp_src[s]/tot_fp_s:.1f}% degli FP)")
            print("-" * 50)
        print("  TRACKING — motmetrics, riga OVERALL (identità del tracker vs ID di traccia GT)")
        print(f"  MOTA          : {tm['mota']:.4f}")
        print(f"  IDF1          : {tm['idf1']:.4f}")
        print(f"  IDP / IDR     : {tm['idp']:.4f} / {tm['idr']:.4f}")
        print(f"  ID switches   : {tm['id_switches']}")
        print(f"  MOTP (IoU)    : {tm['motp_iou']:.4f}")
        print(f"  TP/FP/FN      : {tm['track_tp']} / {tm['track_fp']} / {tm['track_fn']}  (GT: {tm['gt_total']})")
        print(f"  Prec / Rec    : {tm['track_precision']:.4f} / {tm['track_recall']:.4f}")
        print(f"  MT / PT / ML  : {tm['mostly_tracked']} / {tm['partially_tracked']} / {tm['mostly_lost']}"
              f"  (su {tm['num_unique_objects']} ID GT)  |  Frag: {tm['num_fragmentations']}")
        print(f"  sequenze      : {tm['num_sequences']}  |  IoU thr: {iou_thr_eval}")
        print("-" * 50)
        if not oracle:
            nf = max(diag['frames'], 1)
            print("  PAST QUERY (il modello sta davvero usando il passato?)")
            print(f"  frame con >=1 track valido : {diag['with_valid']}/{diag['frames']} "
                  f"({100*diag['with_valid']/nf:.1f}%)")
            print(f"  media track validi/frame   : {diag['valid_sum']/nf:.2f}")
            print(f"  frame senza alcun track    : {diag['no_tracks']} "
                  f"({100*diag['no_tracks']/nf:.1f}%)   → detection/associazione")
            print(f"  frame con track ma 0 validi: {diag['tracks_but_none_valid']} "
                  f"({100*diag['tracks_but_none_valid']/nf:.1f}%)   → gate is_valid")
            tot_fail = max(diag['fail_len'] + diag['fail_conf'] + diag['fail_coher'], 1)
            print(f"  causa del rifiuto  finestra<{P}: {100*diag['fail_len']/tot_fail:.1f}%  |  "
                  f"mean<{mean_conf_thr}: {100*diag['fail_conf']/tot_fail:.1f}%  |  "
                  f"coerenza IoU<{coher_thr}: {100*diag['fail_coher']/tot_fail:.1f}%")
            print("-" * 50)
        if use_tt:
            print(f"  tracker                    : TrackTrack-AR (TPA + TAI, CVPR 2025)  {tt_cfg}")
            print(f"  track creati (ID totali)   : {tt.next_id}")
            print(f"  TrackTrack-AR              : {tt.summary()}")
        else:
            print(f"  track creati (ID totali)   : {Track._next_id}")
            print(f"  NMS: standard scartate     : {diag['std_nms']}  |  track doppioni eliminati: {diag['dup_tracks']}")
        print("=" * 50)

        # file MOTChallenge, uno per sequenza
        if export_mot:
            for seq_name, lines in mot_pred.items():
                with open(os.path.join(mot_pred_dir, seq_name + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
            for seq_name, lines in mot_gt.items():
                with open(os.path.join(mot_gt_dir, seq_name + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
            print(f"\n[MOT] {len(mot_pred)} sequenze esportate (eval_tracker.py su questi file dà gli stessi numeri)")
            print(f"[MOT]   pred → {mot_pred_dir}")
            print(f"[MOT]   gt   → {mot_gt_dir}")

        # motmetrics per sequenza + OVERALL
        if track_metrics.summary is not None:
            csv_path = os.path.join(
                self.output_dir,
                'tracking_motmetrics_oracle.csv' if oracle else
                'tracking_motmetrics_tracktrack.csv' if use_tt else 'tracking_motmetrics.csv')
            print("\n[MOT] === metriche motmetrics per sequenza (riga OVERALL = TRACKING sopra) ===")
            print(track_metrics.render())
            track_metrics.summary.to_csv(csv_path)
            print(f"[MOT] CSV: {csv_path}")

        import pprint
        pprint.pprint(final_results)

        final_results['tp'] = total_tp
        final_results['fp'] = total_fp
        final_results['fn'] = total_fn
        final_results['precision'] = precision
        final_results['recall'] = recall
        final_results['f1'] = f1
        final_results.update(tm)   # mota, idf1, id_switches, motp_iou, track_*, ...
        final_results['ar_std_nms'] = diag['std_nms']        # standard scartate dalla NMS del tracker
        final_results['ar_dup_tracks'] = diag['dup_tracks']  # track doppioni eliminati

        # wandb, se c'è una run attiva
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"track/{k}": v for k, v in tm.items()})
        except Exception:
            pass

        return final_results

    @torch.no_grad()
    def _save_visualization_autoregressive(self, frames_dict, present_labels, tracks,
                                           current_dets, frame_idx, out_dir, future_pred=None,
                                           future_gt=None, future_ids=None):
        """
        Frame dell'eval autoregressivo: track (blu), GT (verde), detection correnti (arancione),
        forecast (rosso) e GT futuro (verde scuro). future_pred: (N, T, 4); future_gt/future_ids:
        liste di T tensori.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # frame a durata massima
        main_d = max(frames_dict.keys())
        img = frames_dict[main_d][0].cpu().float().numpy()
        img = np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img[0]
        img = np.clip(img, 0.0, 1.0)
        H, W = img.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        fig.patch.set_facecolor('#111111'); ax.set_facecolor('#111111')
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img, vmin=0, vmax=1)
        ax.axis('off')

        def draw(box, color, ls='-', lw=1.5, alpha=1.0, label=None):
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            ax.add_patch(patches.Rectangle(
                (x1, y1), bw * W, bh * H,
                linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls, alpha=alpha))
            if label:
                ax.text(x1, max(y1 - 2, 0), label,
                        fontsize=6, color=color, fontfamily='monospace',
                        bbox=dict(facecolor='black', alpha=0.5, pad=0.1, linewidth=0))

        # 1. Traiettorie dei track (blu, recente più opaco)
        for t in tracks:
            bs = list(t.boxes)
            n = max(len(bs) - 1, 1)
            for j, box in enumerate(bs):
                alpha = max(0.2, 0.3 + 0.6 * j / n)
                draw(box, '#4169E1', '--', 1.0, alpha)

        # 2. GT presente (verde)
        if present_labels is not None:
            for box in present_labels[0]['boxes'].cpu().numpy():
                draw(box, '#00FF00', '-', 2.0, label='GT')

        # 3. Detection correnti (arancione) con ID di traccia + score
        for det in current_dets:
            box, score = det[0], det[1]
            tid = det[2] if len(det) > 2 else None
            lbl = f'#{tid} {score:.2f}' if tid is not None else f'{score:.2f}'
            draw(box, '#FF8C00', '-', 2.0, label=lbl)

        # 4. forecast (rosso)
        if future_pred is not None and len(future_pred) > 0:
            T = future_pred.shape[1]
            for n in range(future_pred.shape[0]):
                cxs, cys = [], []
                for t in range(T):
                    fb = future_pred[n, t]
                    if float(np.abs(fb).sum()) == 0:      # slot vuoto/padding
                        continue
                    a = 1.0 - 0.6 * (t / max(T - 1, 1))   # più trasparente con lo step
                    draw(fb.tolist(), '#FF3333', '-', 1.4, alpha=a)
                    cxs.append(fb[0] * W); cys.append(fb[1] * H)
                if len(cxs) > 1:
                    ax.plot(cxs, cys, '-', color='#FF3333', lw=0.8, alpha=0.6)

        # 5. GT futuro (verde scuro), traiettorie collegate con future_ids
        if future_gt is not None:
            T_gt = len(future_gt)
            if future_ids is not None:
                id2traj = {}   # id -> {t: [cx, cy, bw, bh]}
                for t in range(T_gt):
                    boxes_t, ids_t = future_gt[t], future_ids[t]
                    for k in range(len(boxes_t)):
                        id2traj.setdefault(int(ids_t[k].item()), {})[t] = boxes_t[k].tolist()
                for traj in id2traj.values():
                    gxs, gys = [], []
                    for t in sorted(traj):
                        a = 1.0 - 0.6 * (t / max(T_gt - 1, 1))
                        box = traj[t]
                        draw(box, '#00B050', '--', 1.4, alpha=a)
                        gxs.append(box[0] * W); gys.append(box[1] * H)
                    if len(gxs) > 1:
                        ax.plot(gxs, gys, '--', color='#00B050', lw=0.8, alpha=0.6)
            else:
                for t in range(T_gt):
                    a = 1.0 - 0.6 * (t / max(T_gt - 1, 1))
                    for box in future_gt[t]:
                        draw(box.tolist(), '#00B050', '--', 1.4, alpha=a)

        fig.suptitle(
            f'Frame {frame_idx:06d} | AUTOREG | tracks={len(tracks)} | det={len(current_dets)}',
            fontsize=9, color='white',
        )
        ax.legend(
            handles=[
                patches.Patch(facecolor='none', edgecolor='#4169E1', linestyle='--', label='Track (pred. passate)'),
                patches.Patch(facecolor='none', edgecolor='#00FF00', linestyle='-',  label='Present GT'),
                patches.Patch(facecolor='none', edgecolor='#FF8C00', linestyle='-',  label='Detection'),
                patches.Patch(facecolor='none', edgecolor='#FF3333', linestyle='-',  label='Future pred'),
                patches.Patch(facecolor='none', edgecolor='#00B050', linestyle='--', label='Future GT'),
            ],
            loc='upper right', fontsize=6,
            facecolor='#222222', edgecolor='#555555', labelcolor='white', framealpha=0.75,
        )

        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    def _setup_model(self):
        """Load rtdetrv2 model from checkpoint."""
        print("Setting up rtdetrv2 model...")

        if not hasattr(self.config, 'checkpoint_path') or self.config.checkpoint_path is None:
            raise ValueError("config.checkpoint_path must be specified for evaluation")

        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint_path}")

        print(f"  Loading from: {self.config.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Initialize model
        print(f"Loading model from PekingU/rtdetr_v2_r18vd...")
        self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")

        if self.config.use_custom_normalization:
            print("Using custom dataset statistics for normalization") # Edo
            self.image_processor.image_mean = [0.007950919680297375, 0.010960247367620468, 0.013969575986266136]
            self.image_processor.image_std = [0.048023562878370285, 0.054145995527505875, 0.066846564412117]

            print(f"[ImageProcessor] mean: {self.image_processor.image_mean}")
            print(f"[ImageProcessor] std: {self.image_processor.image_std}")

        self.model = get_model("rtdetrv2_past_conditioned", self.device, self.config)

        print("phase:", getattr(self.config, 'phase', 1),
      "| ha standard_queries:", hasattr(self.model, 'standard_queries'))
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model loaded: {num_params:.1f}M parameters")
        print(f"Durations: {self.config.durations} ms")

        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    def _setup_dataset(self):
        """Initialize test dataset and dataloader."""
        print("💿 Setting up test dataset and dataloader...")

        task_modality = getattr(self.config, 'task_modality', 'detection')
        if task_modality not in ['detection', 'tracking']:
            task_modality = 'detection'

        print(f"  📂 Loading test dataset: {self.config.dataset_name} | Split: test")
        print(f"  🔢 Using index file: {self.config.index_path}")
        print(f"  🎯 Task modality: {task_modality}")

        self.collate_fn = lambda batch: collate_fn_detection_rtdetrv2_past_conditioned(
            batch,
            image_processor=self.image_processor,
            config=self.config,
        )

        self.test_dataset = get_dataset(
            self.config.dataset_name,
            split='test',
            modality=task_modality,
            config=self.config
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        print(f"🚚 Test samples: {len(self.test_dataset)} | Test batches: {len(self.test_loader)}")

    @torch.no_grad()
    def _save_visualization(self, frames_dict, results, ground_truths,
                            batch_idx, sample_idx=0, past_boxes=None, vis_score_thr=0.0):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image as PILImage

        orig_h, orig_w = ground_truths[sample_idx].get('orig_size', (720, 1280))

        # durata principale (la più lunga, come nel base)
        main_duration = sorted(frames_dict.keys())[-1]
        img = frames_dict[main_duration][sample_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 7))
        if img.shape[0] == 1:
            pil = PILImage.fromarray((img[0] * 255).astype(np.uint8), mode='L').resize((orig_w, orig_h))
            ax.imshow(np.array(pil) / 255.0, cmap='gray')
        else:
            chw = np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
            pil = PILImage.fromarray((chw * 255).astype(np.uint8)).resize((orig_w, orig_h))
            ax.imshow(np.array(pil) / 255.0)

        def draw(box_xyxy, color, ls='-', lw=2.0):
            x1, y1, x2, y2 = box_xyxy
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls))

        def denorm(cxcywh):
            cx, cy, bw, bh = cxcywh
            return ((cx - bw/2)*orig_w, (cy - bh/2)*orig_h,
                    (cx + bw/2)*orig_w, (cy + bh/2)*orig_h)

        # GT (già xyxy pixel da _build_ground_truths_batch) — verde
        for b in ground_truths[sample_idx]['boxes']:
            draw(b.tolist() if hasattr(b, 'tolist') else b, '#00FF00', '-', 2.0)

        # Predizioni (xyxy pixel da post_process) — rosso
        # Predizioni (xyxy pixel da post_process) — rosso, con score
        pred_boxes_s  = results[sample_idx]['boxes'].cpu().numpy()
        pred_scores_s = results[sample_idx]['scores'].cpu().numpy()
        for box, score in zip(pred_boxes_s, pred_scores_s):
            if score < vis_score_thr:        # solo per la visualizzazione
                continue
            draw(box, '#FF3333', '-', 1.5)
            x1, y1, x2, y2 = box
            ax.text(
                x1, y1 - 4,
                f"{score:.2f}",
                fontsize=7, color='#FF3333', fontweight='bold',
                va='bottom', ha='left',
                bbox=dict(facecolor='black', alpha=0.5, pad=1, linewidth=0),
            )

        # Ultimo passato (cxcywh norm, indice 0) — giallo tratteggiato
        if past_boxes is not None:
            last_past = past_boxes[sample_idx, :, 0, :].cpu()   # (max_objects, 4)
            for pb in last_past:
                if pb.abs().sum() == 0: # slot padding
                    continue
                draw(denorm(pb.tolist()), '#FFD700', '--', 1.5)

        # coloro anche le altre box passate
        if past_boxes is not None:
            all_past = past_boxes[sample_idx].cpu()   # (max_objects, P, 4)
            P = all_past.shape[1]
            for drone in range(all_past.shape[0]):
                for p_idx in range(P):
                    pb = all_past[drone, p_idx]
                    if pb.abs().sum() == 0: # slot padding
                        continue
                    if p_idx == 0:
                        continue  
                    else:
                        draw(denorm(pb.tolist()), '#8A2BE2', '--', 1.0)  

        ax.legend(handles=[
            patches.Patch(edgecolor='#00FF00', facecolor='none', label='GT'),
            patches.Patch(edgecolor='#FF3333', facecolor='none', label='Pred'),
            patches.Patch(edgecolor='#FFD700', facecolor='none', label='Last past'),
        ], loc='upper right', fontsize=8)
        ax.set_title(f"Batch {batch_idx} | Sample {sample_idx}")
        ax.axis('off')

        plt.tight_layout()
        filename   = f"batch_{batch_idx:06d}_sample_{sample_idx:02d}.png"
        query_mode = getattr(self.config, 'query_mode', 'both')
        out_dir    = os.path.join(self.output_dir, "visualizations", f"test_{query_mode}")
        os.makedirs(out_dir, exist_ok=True)            # crea test_{query_mode} se manca
        save_path  = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def _save_visualization_forecast(self, frames_dict, past_boxes, future_gt,
                                     future_pred, future_mask, batch_idx, sample_idx=0):
        """Frame di forecasting: presente (giallo), futuro GT (verde) e predetto (rosso)."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image as PILImage

        orig_w = getattr(self.config, 'img_width', 1280)
        orig_h = getattr(self.config, 'img_height', 720)

        main_duration = sorted(frames_dict.keys())[-1]
        img = frames_dict[main_duration][sample_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 7))
        if img.shape[0] == 1:
            pil = PILImage.fromarray((img[0] * 255).astype(np.uint8), mode='L').resize((orig_w, orig_h))
            ax.imshow(np.array(pil) / 255.0, cmap='gray')
        else:
            chw = np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
            pil = PILImage.fromarray((chw * 255).astype(np.uint8)).resize((orig_w, orig_h))
            ax.imshow(np.array(pil) / 255.0)

        def denorm(cxcywh):
            cx, cy, bw, bh = cxcywh
            return ((cx - bw/2)*orig_w, (cy - bh/2)*orig_h,
                    (cx + bw/2)*orig_w, (cy + bh/2)*orig_h)

        def draw(box_xyxy, color, ls='-', lw=1.8, alpha=1.0):
            x1, y1, x2, y2 = box_xyxy
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=lw, edgecolor=color, facecolor='none',
                        linestyle=ls, alpha=alpha))

        pb  = past_boxes[sample_idx].cpu()     # (M, P, 4)
        fgt = future_gt[sample_idx].cpu()      # (M, T, 4)
        fpr = future_pred[sample_idx].cpu()    # (M, T, 4)
        fm  = future_mask[sample_idx].cpu()    # (M, T)
        M, T = fm.shape

        for m in range(M):
            if not bool(fm[m].any()):
                continue
            # presente = passato più recente (indice 0)
            present = pb[m, 0]
            has_present = bool(present.abs().sum() > 0)
            gx, gy, px, py = [], [], [], []
            if has_present:
                draw(denorm(present.tolist()), '#FFD700', '-', 2.0, alpha=0.9)
                gx.append(present[0].item()*orig_w); gy.append(present[1].item()*orig_h)
                px.append(present[0].item()*orig_w); py.append(present[1].item()*orig_h)
            for t in range(T):
                if not bool(fm[m, t]):
                    continue
                a = 1.0 - 0.6 * (t / max(T - 1, 1))          # più trasparente con lo step
                gt_b = fgt[m, t]; pr_b = fpr[m, t]
                draw(denorm(gt_b.tolist()), '#00FF00', '--', 1.6, alpha=a)   # futuro GT
                draw(denorm(pr_b.tolist()), '#FF3333', '-',  1.6, alpha=a)   # futuro predetto
                gx.append(gt_b[0].item()*orig_w); gy.append(gt_b[1].item()*orig_h)
                px.append(pr_b[0].item()*orig_w); py.append(pr_b[1].item()*orig_h)
            if len(gx) > 1:
                ax.plot(gx, gy, '-', color='#00FF00', lw=1.0, alpha=0.6)
            if len(px) > 1:
                ax.plot(px, py, '-', color='#FF3333', lw=1.0, alpha=0.6)

        ax.legend(handles=[
            patches.Patch(edgecolor='#FFD700', facecolor='none', label='Present (last past)'),
            patches.Patch(edgecolor='#00FF00', facecolor='none', label='Future GT'),
            patches.Patch(edgecolor='#FF3333', facecolor='none', label='Future pred'),
        ], loc='upper right', fontsize=8)
        ax.set_title(f"Forecast | Batch {batch_idx} | Sample {sample_idx}")
        ax.axis('off')

        plt.tight_layout()
        out_dir = os.path.join(self.output_dir, "visualizations", "test_forecast")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"batch_{batch_idx:06d}_sample_{sample_idx:02d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
