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
from scipy.optimize import linear_sum_assignment


def box_iou_cxcywh(a, b):
    """IoU tra due box (4,) cxcywh normalizzate."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def box_center_dist(a, b):
    """Distanza euclidea normalizzata tra i centri di due box cxcywh."""
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


class Track:
    """Finestra scorrevole delle ultime P box predette dal modello (indice 0 = più vecchio)."""

    _next_id = 0 # contatore globale: ogni track nuova riceve un ID univoco e mai riusato

    def __init__(self, box, score, P):
        Track._next_id += 1
        self.track_id = Track._next_id   # identità del tracker (serve per MOTA/IDF1/ID-switch)
        self.boxes = deque([box], maxlen=P)
        self.scores = deque([score], maxlen=P)
        self.missed = 0
        # box t+1 predetta dalla forecasting head al frame precedente
        # (None finché il track non è stato dato come query-passato con head attiva).
        self.forecast_next = None

    def update(self, box, score):
        self.boxes.append(box)
        self.scores.append(score)
        self.missed = 0

    @property
    def last(self):
        return self.boxes[-1]

    @property
    def predicted_next(self):
        """
        Posizione stimata al frame corrente per l'associazione.
        Preferisce la predizione appresa dalla forecasting head (t+1 calcolata al frame precedente); 
        fallback su estrapolazione lineare se non disponibile.
        """
        if self.forecast_next is not None:
            return np.clip(self.forecast_next, 0.0, 1.0)
        if len(self.boxes) < 2:
            return self.last
        velocity = self.boxes[-1] - self.boxes[-2]
        return np.clip(self.last + velocity, 0.0, 1.0)

    def is_valid(self, P, conf_thr, coher_thr):
        """Valido (inviabile come past) se finestra piena, confidenza media alta, box coerenti."""
        if len(self.boxes) < P:
            return False
        if float(np.mean(self.scores)) < conf_thr:
            return False
        bs = list(self.boxes)
        return all(box_iou_cxcywh(a, b) >= coher_thr for a, b in zip(bs[:-1], bs[1:]))

    def trajectory_recent_first(self, P):
        """(P, 4) con indice 0 = più recente — formato atteso dal modello."""
        return np.stack(list(self.boxes)[::-1], axis=0)


def best_match_track(tracks, box, iou_thr, dist_thr, exclude):
    """Track migliore per una detection: 
    IoU su predicted_next (sopra iou_thr),
    poi fallback su distanza centri (sotto dist_thr)."""
   
    best_iou, best_iou_track = iou_thr, None
    for t in tracks:
        if t in exclude:
            continue
        v = box_iou_cxcywh(t.predicted_next, box)
        if v > best_iou:
            best_iou, best_iou_track = v, t
    if best_iou_track is not None:
        return best_iou_track

    best_dist, best_dist_track = dist_thr, None
    for t in tracks:
        if t in exclude:
            continue
        d = box_center_dist(t.predicted_next, box)
        if d < best_dist:
            best_dist, best_dist_track = d, t
    return best_dist_track


# ══════════════════════════════════════════════════════════════════════════════
# METRICHE DI TRACKING  (CLEAR-MOT + IDF1)
# ══════════════════════════════════════════════════════════════════════════════

class TrackingMetrics:

    def __init__(self, iou_thr=0.5):
        self.iou_thr = float(iou_thr)
        # aggregati globali
        self.gt_total = 0
        self.fp = 0
        self.fn = 0
        self.idsw = 0
        self.matched = 0
        self.iou_sum = 0.0
        self.idtp = 0
        self.idfp = 0
        self.idfn = 0
        self.n_seq = 0
        self._start_seq()

    def _start_seq(self):
        self._prev = {} # gt_id -> pred_id dell'ultimo match (persistente nella sequenza)
        self._cooc = {} # (gt_id, pred_id) -> n. frame in cui sono stati matchati
        self._seq_gt = 0 # detection GT totali nella sequenza
        self._seq_pred = 0 # detection predette totali nella sequenza
        self._seq_used = False

    def new_sequence(self):
        """hiude la sequenza corrente e ne apre una nuova."""
        self.finalize()
        self._start_seq()

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        """
        Un frame. boxes: array (N, 4) cxcywh normalizzate; ids: sequenze di interi.
        Matching per-frame ottimo (Hungarian su IoU) con soglia iou_thr.
        """
        n_gt, n_pr = len(gt_ids), len(pred_ids)
        self._seq_used = True
        self.gt_total += n_gt
        self._seq_gt += n_gt
        self._seq_pred += n_pr

        if n_gt == 0 or n_pr == 0:
            self.fp += n_pr
            self.fn += n_gt
            return

        iou = np.zeros((n_gt, n_pr), dtype=np.float64)
        for i in range(n_gt):
            for j in range(n_pr):
                iou[i, j] = box_iou_cxcywh(gt_boxes[i], pred_boxes[j])

        rows, cols = linear_sum_assignment(-iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= self.iou_thr]

        for r, c in pairs:
            g, p = int(gt_ids[r]), int(pred_ids[c])
            self.matched += 1
            self.iou_sum += float(iou[r, c])
            self._cooc[(g, p)] = self._cooc.get((g, p), 0) + 1
            # ID switch: questo GT era associato a un'altra identità predetta
            if g in self._prev and self._prev[g] != p:
                self.idsw += 1
            self._prev[g] = p

        self.fp += n_pr - len(pairs)
        self.fn += n_gt - len(pairs)

    def finalize(self):
        if not self._seq_used:
            return
        self.n_seq += 1
        if self._cooc:
            gts = sorted({g for g, _ in self._cooc})
            prs = sorted({p for _, p in self._cooc})
            gi = {g: i for i, g in enumerate(gts)}
            pi = {p: i for i, p in enumerate(prs)}
            C = np.zeros((len(gts), len(prs)), dtype=np.float64)
            for (g, p), n in self._cooc.items():
                C[gi[g], pi[p]] = n
            r, c = linear_sum_assignment(-C)
            idtp = int(C[r, c].sum())
        else:
            idtp = 0
        self.idtp += idtp
        self.idfn += self._seq_gt - idtp
        self.idfp += self._seq_pred - idtp
        self._seq_used = False   # idempotenza

    def compute(self):
        self.finalize()
        gt = max(self.gt_total, 1)
        den_idf1 = 2 * self.idtp + self.idfp + self.idfn
        return {
            'mota':        1.0 - (self.fn + self.fp + self.idsw) / gt,
            'motp_iou':    self.iou_sum / max(self.matched, 1),
            'idf1':        (2.0 * self.idtp / den_idf1) if den_idf1 > 0 else 0.0,
            'id_switches': self.idsw,
            'track_tp':    self.matched,
            'track_fp':    self.fp,
            'track_fn':    self.fn,
            'track_precision': self.matched / max(self.matched + self.fp, 1),
            'track_recall':    self.matched / gt,
            'gt_total':    self.gt_total,
            'num_sequences': self.n_seq,
        }


def collate_autoregressive(batch, image_processor):
    """
    Collate batch=1 per l'eval autoregressivo. Restituisce
        (frames_dict, present_labels, present_ids, future_gt, future_ids)
    dove present_ids sono gli ID di traccia GT dei droni presenti (servono per le
    metriche di tracking); allineati posizionalmente a present_labels[0]['boxes'].
    con tutti i droni presenti come GT. future_gt/future_ids sono liste di T tensori
    (n_t, 4) cxcywh norm / (n_t,) id — servono alla viz per disegnare il GT futuro;
    sono None se il dataset non ha annotazioni future (num_future_annotations=0).
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

    # ID di traccia GT dei droni presenti (per MOTA/IDF1/ID-switch). Stesso ordine dei
    # coco_anns passati al processor → allineati a present_labels[0]['boxes'].
    present_ids = target.get('ids', torch.zeros(0, dtype=torch.int64))

    # GT futuro (per la viz autoregressiva): None se num_future_annotations=0
    future_gt  = target.get('future_boxes', None)
    future_ids = target.get('future_ids', None)

    return frames_dict, present_labels, present_ids, future_gt, future_ids


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

    # Build future_gt (B, max_objects, T, 4) + future_mask (B, max_objects, T) — forecasting
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

    return {
        'frames': frames_dict,
        'labels': present_labels,
        'past_boxes': past_boxes,
        'past_mask': past_mask,
        'new_labels': new_labels,   # <-- aggiungi
        'future_gt': future_gt,     # (B, max_objects, T, 4) — forecasting
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
        vis_score_thr = getattr(self.config, 'vis_score_threshold', 0.3)   # soglia SOLO per la visualizzazione
        query_mode    = self.config.query_mode
        phase         = getattr(self.config, 'phase', 1)


        # temporaneo per diagnosi
        diag_best_iou = int(getattr(self.config, 'diag_best_iou', 0) or 0)
        diag_gt_count = 0 # n. GT totali visti
        diag_iou_sum = 0.0 # Σ best-IoU per GT
        diag_score_sum = 0.0 # Σ score@best-IoU per GT
        diag_loc_count = 0 # n. GT con best-IoU ≥ iou_thr (localizzati)
        diag_loc_score_sum = 0.0 # Σ score@best-IoU sui soli GT localizzati
                             
        print("\nStarting evaluation...")
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Inference")):
            
            if batch is None:
                continue
            
            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            past_boxes = batch['past_boxes'].to(self.device)
            past_mask  = batch['past_mask'].to(self.device)

            if query_mode == 'past_only' and past_boxes.shape[1] == 0:
                continue
            
            # Forward pass
            outputs = self.model(
                event_frames=frames_dict,
                past_boxes=past_boxes,
                past_mask=past_mask,
                query_mode=query_mode,
            )

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
                # ── Diagnostico soglia-free: best-IoU per GT su TUTTE le pred (nessun filtro score) ──
                if diag_best_iou:
                    gtb = gt['boxes']
                    pbx = pred['boxes']
                    if gtb.shape[0] > 0:
                        if pbx.shape[0] == 0:
                            diag_gt_count += int(gtb.shape[0])   # nessuna pred → best-IoU=0 per ogni GT
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

        # ── Diagnostico soglia-free: localizzazione vs calibrazione del ramo attivo ──
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
            fp = outputs.future_boxes[:, :M]                        # (B, M, T, 4) — query-passato

            # ── Visualizzazione: present (giallo) + futuro GT (verde) + futuro predetto (rosso) ──
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

        # ── ADE/FDE per orizzonte temporale (full + short 0.4s + mid 0.8s) ──────────
        step_dt = float(getattr(self.config, 'forecast_step_dt_s', 1.0 / 30.0))   # s/step
        horizons = {                                   # nome_metrica → orizzonte (s)
            'short_0_4s': float(getattr(self.config, 'ade_short_horizon_s', 0.4)),
            'mid_0_8s':   float(getattr(self.config, 'ade_mid_horizon_s',   0.8)),
        }

        def _horizon_metrics(H_s):
            n = min(T, max(1, int(round(H_s / step_dt))))        # n. step entro H (cap a T)
            disp = sum(step_disp[:n]); cnt = sum(step_cnt[:n])
            ade_h = disp / max(cnt, 1)
            fde_h = step_disp[n - 1] / max(step_cnt[n - 1], 1)   # errore all'ultimo step ≤ H
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
        """Forward AR per-frame. BASE: una singola passata in 'both' (comportamento
        originale, invariato). La sottoclasse 2-pass la sovrascrive con passate separate
        standard_only + past_only (fix dell'interferenza both). Vedi
        evaluator_rtdetrv2_past_conditioned_2pass.py."""
        return self.model(
            event_frames=frames_dict,
            past_boxes=past_boxes,
            past_mask=past_mask,
            query_mode='both',
        )

    @torch.no_grad()
    def evaluate_autoregressive(self):
        from torchvision.ops import box_iou
        """
        Eval closed-loop: il passato NON viene dalla GT ma dalle predizioni del modello.
        Stateful, sequenziale, batch=1. I track persistono tra frame e si resettano al
        cambio di sequenza (hdf5_path). Metriche identiche a evaluate() (mAP + TP/FP/FN),
        calcolate sulle current_dets (output reale del sistema) vs droni presenti.
        """
        self.model.eval()
        self.metric = MeanAveragePrecision(iou_type='bbox')

        # ── Parametri ────────────────────────────────────────────────────────────
        iou_thr_eval = getattr(self.config, 'eval_iou_threshold', 0.5)
        tp_score_thr = getattr(self.config, 'tp_score_threshold', 0.5)
        W = getattr(self.config, 'img_width', 1280)
        H = getattr(self.config, 'img_height', 720)
        P = getattr(self.config, 'num_past_steps',
                    getattr(self.config, 'num_past_annotations', 10))
        conf_thr  = getattr(self.config, 'ar_conf_thr', 0.35)
        coher_thr = getattr(self.config, 'ar_coher_thr', 0.10)
        iou_thr   = getattr(self.config, 'ar_iou_thr', 0.2)   # FIX: era 0.0 → match sempre (vedi best_match_track)
        dist_thr  = getattr(self.config, 'ar_dist_thr', 0.08)
        max_missed = getattr(self.config, 'ar_max_missed', 6)
        # Le query-passato hanno priorita. Una detection standard che ricalca una box già
        # piazzata da una query-passato viene scartata → niente track duplicati sullo stesso
        # drone.
        std_past_iou_thr = getattr(self.config, 'ar_std_past_iou_thr', 0.3)
        # Usa il forecast appreso t+1 come reference
        # point della query-passato del frame successivo (rimpiazza la box più recente).
        use_fc_ref = bool(getattr(self.config, 'ar_use_forecast_refpoint', 0))

        # ── Loader batch=1 sequenziale ───────────────────────────────────────────
        dataset = self.test_dataset
        # C2: il tracker accumula la storia da frame CONSECUTIVI. Con subsample>1 due frame
        # distano subsample×33ms invece di 33ms → il passato closed-loop ha passo temporale
        # diverso dal training (velocità/ancora fuori scala, is_valid rigetta i track) e le
        # metriche AR sarebbero sbagliate IN SILENZIO. Il tracker originale forza subsample=1.
        ar_subsample = getattr(self.config, 'subsample', 1)
        if ar_subsample != 1:
            raise RuntimeError(
                f"evaluate_autoregressive richiede subsample=1 (attuale: {ar_subsample}). "
            )
        windows = getattr(dataset, 'windows', None)
        if windows is None:
            raise RuntimeError("Il dataset non espone .windows: impossibile rilevare i confini di sequenza.")

        # ── Subsample per SEQUENZA (per sweep) ──────────────────────────
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
            collate_fn=lambda b: collate_autoregressive(b, self.image_processor),
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
        # Metriche di tracking sulle identità del tracker vs ID di traccia GT
        track_metrics = TrackingMetrics(iou_thr=iou_thr_eval)

        
        diag = {'frames': 0, 'with_valid': 0, 'no_tracks': 0, 'tracks_but_none_valid': 0,
                'valid_sum': 0, 'fail_len': 0, 'fail_conf': 0, 'fail_coher': 0}

        
        diag_fp_source = bool(getattr(self.config, 'diag_fp_source', 0))
        emit_src = {'past': 0, 'standard': 0}   # det emesse (score>=tp_score_thr) per ramo
        tp_src   = {'past': 0, 'standard': 0}   # di quelle, matchate a un GT
        fp_src   = {'past': 0, 'standard': 0}   # di quelle, NON matchate (= FP) → il numero chiave

        # ── Export MOTChallenge per eval_tracker.py (motmetrics) ──
        export_mot = bool(getattr(self.config, 'ar_export_mot', 0))
        mot_pred, mot_gt, mot_frame = {}, {}, {}
        # seq_id → nome-file UNIVOCO. Il basename dell'hdf5 collide fra scene diverse (su FRED
        # sono tutti uguali) → senza questo tutte le sequenze finirebbero in un file solo.
        seq_name_map = {}
        if export_mot:
            mot_pred_dir = os.path.join(self.output_dir, 'mot_tracks', 'pred')
            mot_gt_dir   = os.path.join(self.output_dir, 'mot_tracks', 'gt')
            os.makedirs(mot_pred_dir, exist_ok=True)
            os.makedirs(mot_gt_dir, exist_ok=True)

        def _why_invalid(t):
            """Prima condizione di is_valid che fallisce, per il breakdown diagnostico."""
            if len(t.boxes) < P:
                return 'fail_len'
            if float(np.mean(t.scores)) < conf_thr:
                return 'fail_conf'
            return 'fail_coher'

        print("\nStarting AUTOREGRESSIVE evaluation...")
        for idx, batch in enumerate(tqdm(loader, desc="Autoreg inference")):
            # ── Confine di sequenza → reset tracker (e chiusura sequenza metriche) ──
            seq_id = windows[idx]['hdf5_path']
            if seq_id != prev_seq:
                tracks = []
                track_metrics.new_sequence()
                prev_seq = seq_id

            frames_dict, present_labels, present_ids, future_gt, future_ids = batch
            frames_dict = {d: f.to(self.device) for d, f in frames_dict.items()}

            # ── Passato dai track validi (predizioni del modello) ────────────────
            valid = [t for t in tracks if t.is_valid(P, conf_thr, coher_thr)]

            diag['frames'] += 1
            diag['valid_sum'] += len(valid)
            if valid:
                diag['with_valid'] += 1
            elif not tracks:
                diag['no_tracks'] += 1 # nessun track: problema di detection
            else:
                diag['tracks_but_none_valid'] += 1  # track esistono ma il gate li rifiuta
                for t in tracks:
                    diag[_why_invalid(t)] += 1

            if valid:
                trajs = []
                for t in valid:
                    traj = t.trajectory_recent_first(P)          # (P, 4), indice 0 = più recente
                    if use_fc_ref and t.forecast_next is not None:
                        # reference point = forecast appreso (stima del presente dal frame prima)
                        traj = traj.copy()
                        traj[0] = np.clip(t.forecast_next, 0.0, 1.0)
                    trajs.append(traj)
                pb = np.stack(trajs, axis=0)
                past_boxes = torch.from_numpy(pb).float().unsqueeze(0).to(self.device)
            else:
                past_boxes = torch.zeros(1, 0, P, 4, device=self.device)
            past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=self.device)

            outputs = self._ar_forward(frames_dict, past_boxes, past_mask)

            if outputs.pred_boxes.shape[1] > 0:
                scores = outputs.logits[0].sigmoid().squeeze(-1)   # (Q,)
                pred_boxes = outputs.pred_boxes[0]                 # (Q, 4)
            else:
                scores = torch.zeros(0, device=self.device)
                pred_boxes = torch.zeros(0, 4, device=self.device)

            # ── Update closed-loop ───────────────────────────────────────────────
            N = past_boxes.shape[1]
            past_b = pred_boxes[:N].cpu().numpy(); past_s = scores[:N].cpu().numpy()
            std_b  = pred_boxes[N:].cpu().numpy(); std_s  = scores[N:].cpu().numpy()

            updated = set()
            current_dets = []   # (box cxcywh np, score, track_id) — l'ID serve alle metriche di tracking
            current_dets_src = []  # 'past'|'standard' per ogni det, parallelo a current_dets (diag_fp_source; sola misura)
            past_det_boxes = []  # M4: box piazzate dalle query-passato (priorità sulle standard)

            # track validi aggiornati dalla loro predizione past (1-a-1)
            for t, b, s in zip(valid, past_b, past_s):
                if s > conf_thr:
                    t.update(b, float(s)); updated.add(t)
                    current_dets.append((b, float(s), t.track_id))
                    current_dets_src.append('past')
                    past_det_boxes.append(b)
                else:
                    t.missed += 1

            # detection standard → associa (eredita l'ID) o crea un nuovo track (nuovo ID)
            for i in np.argsort(-std_s):
                if std_s[i] <= conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                # M4: scarto la standard se ricalca una box già piazzata da una query-passato
                if any(box_iou_cxcywh(b, pbx) >= std_past_iou_thr for pbx in past_det_boxes):
                    continue
                m = best_match_track(tracks, b, iou_thr, dist_thr, exclude=updated)
                if m is not None:
                    m.update(b, s); updated.add(m)
                    current_dets.append((b, s, m.track_id))
                    current_dets_src.append('standard')
                else:
                    nt = Track(b, s, P); tracks.append(nt); updated.add(nt)
                    current_dets.append((b, s, nt.track_id))
                    current_dets_src.append('standard')

            # invecchiamento + eliminazione track morti
            for t in tracks:
                if t not in updated:
                    t.missed += 1
            tracks = [t for t in tracks if t.missed < max_missed]

            # ── memorizza il forecast t+1 (appreso) per i track validi ──
            # future_boxes[0, i, 0] è la box t+1 del track valid[i] (stesso ordine delle
            # query-passato). Verrà usata come predicted_next all'associazione del frame dopo.
            # I track NON validi in questo frame azzerano forecast_next (evita di riusare una
            # predizione stantia → predicted_next torna all'estrapolazione lineare).
            if outputs.future_boxes is not None and N > 0:
                fc_next = outputs.future_boxes[0, :N, 0].detach().cpu().numpy()   # (N, 4)
                fc_map = {id(t): fb for t, fb in zip(valid, fc_next)}
            else:
                fc_map = {}
            for t in tracks:
                t.forecast_next = fc_map.get(id(t), None)

            # ── Visualizzazione (ogni vis_every_n_batches frame) ─────────────────
            if idx % getattr(self.config, 'vis_every_n_batches', 200) == 0:
                out_dir = os.path.join(self.output_dir, "visualizations", "test_autoregressive")
                # Predizioni future del forecasting head per i track validi (solo se eval_forecasting=1)
                future_pred = None
                if getattr(self.config, 'eval_forecasting', 0) and outputs.future_boxes is not None and N > 0:
                    future_pred = outputs.future_boxes[0, :N].detach().cpu().numpy()   # (N, T, 4)
                self._save_visualization_autoregressive(
                    frames_dict, present_labels, tracks, current_dets, idx, out_dir,
                    future_pred=future_pred, future_gt=future_gt, future_ids=future_ids,
                )

            # ── Metriche di TRACKING: identità predette vs ID di traccia GT ──────
            # Usa le stesse detection della mAP (current_dets) ma con il loro track_id.
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

            # ── Export MOTChallenge del frame: cxcywh norm → pixel left,top,w,h ──
            # Stesso contatore di frame (fno) per pred e GT → allineati. Un file per sequenza.
            if export_mot:
                # Nome-file univoco per sequenza (il basename collide fra scene → 1 file solo).
                if seq_id not in seq_name_map:
                    seq_name_map[seq_id] = f"seq_{len(seq_name_map):04d}"
                seq_name = seq_name_map[seq_id]
                fno = mot_frame.get(seq_name, 0) + 1
                mot_frame[seq_name] = fno
                pred_l = mot_pred.setdefault(seq_name, [])
                for b, s, tid in current_dets:
                    cx, cy, bw, bh = (float(v) for v in b)
                    pred_l.append(f"{fno},{int(tid)},{(cx-bw/2)*W:.2f},{(cy-bh/2)*H:.2f},"
                                  f"{bw*W:.2f},{bh*H:.2f},{float(s):.4f},-1,-1,-1")
                gt_l = mot_gt.setdefault(seq_name, [])
                for i in range(n_pair):
                    cx, cy, bw, bh = (float(v) for v in gt_boxes_norm[i])
                    gt_l.append(f"{fno},{int(gt_ids_np[i])},{(cx-bw/2)*W:.2f},{(cy-bh/2)*H:.2f},"
                                f"{bw*W:.2f},{bh*H:.2f},1,-1,-1,-1")

            # ── Metrica: current_dets (predizioni) vs droni presenti (GT) ────────
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

            # TP/FP/FN a tp_score_thr (greedy IoU, come in evaluate())
            keep = pred_scores >= tp_score_thr
            pb_keep = pred_boxes_xyxy[keep]
            n_pred, n_gt = len(pb_keep), len(gt_boxes_xyxy)
            # FP-by-source: sorgenti delle SOLE det tenute (>=tp_score_thr), in ordine pb_keep
            src_keep = ([current_dets_src[i] for i, k in enumerate(keep.tolist()) if k]
                        if diag_fp_source else None)
            if n_pred == 0 and n_gt == 0:
                pass
            elif n_pred == 0:
                total_fn += n_gt
            elif n_gt == 0:
                total_fp += n_pred
                if diag_fp_source:
                    for s in src_keep:                       # nessun GT nel frame → tutte FP
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

        # ── Risultati ────────────────────────────────────────────────────────────
        print("\nComputing final mAP score (autoregressive)...")
        final_results = self.metric.compute()
        tm = track_metrics.compute()   # chiude anche l'ultima sequenza

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        print("\n" + "=" * 50)
        print("  MODE                       : AUTOREGRESSIVE (closed-loop)")
        print(f"  IoU threshold per TP/FP/FN : {iou_thr_eval}")
        print(f"  conf_thr / tp_score_thr    : {conf_thr} / {tp_score_thr}")
        print(f"  P / max_missed             : {P} / {max_missed}")
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
        print("  TRACKING (identità del tracker vs ID di traccia GT)")
        print(f"  MOTA          : {tm['mota']:.4f}")
        print(f"  IDF1          : {tm['idf1']:.4f}")
        print(f"  ID switches   : {tm['id_switches']}")
        print(f"  MOTP (IoU)    : {tm['motp_iou']:.4f}")
        print(f"  TP/FP/FN      : {tm['track_tp']} / {tm['track_fp']} / {tm['track_fn']}  (GT: {tm['gt_total']})")
        print(f"  Prec / Rec    : {tm['track_precision']:.4f} / {tm['track_recall']:.4f}")
        print(f"  sequenze      : {tm['num_sequences']}  |  IoU thr: {iou_thr_eval}")
        print("-" * 50)
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
              f"conf<{conf_thr}: {100*diag['fail_conf']/tot_fail:.1f}%  |  "
              f"coerenza IoU<{coher_thr}: {100*diag['fail_coher']/tot_fail:.1f}%")
        print(f"  track creati (ID totali)   : {Track._next_id}")
        print("=" * 50)

        # ── Scrittura dei file MOTChallenge (uno per sequenza) per eval_tracker.py ──
        if export_mot:
            for seq_name, lines in mot_pred.items():
                with open(os.path.join(mot_pred_dir, seq_name + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
            for seq_name, lines in mot_gt.items():
                with open(os.path.join(mot_gt_dir, seq_name + '.txt'), 'w') as f:
                    f.write('\n'.join(lines) + ('\n' if lines else ''))
            print(f"\n[MOT] {len(mot_pred)} sequenze esportate")
            print(f"[MOT]   pred → {mot_pred_dir}")
            print(f"[MOT]   gt   → {mot_gt_dir}")

            _cm = None
            try:
                from evaluation.eval_tracker import compute_tracking_metrics as _cm
            except Exception:
                try:
                    from eval_tracker import compute_tracking_metrics as _cm
                except Exception:
                    _cm = None
            if _cm is not None:
                try:
                    csv_path = os.path.join(self.output_dir, 'tracking_motmetrics.csv')
                    print("\n[MOT] === metriche motmetrics (riferimento — riga OVERALL) ===")
                    _cm(mot_gt_dir, mot_pred_dir, csv_path)   # (gt_dir, tracks_dir, output_file)
                    print(f"[MOT] CSV: {csv_path}")
                except Exception as e:
                    print(f"[MOT] motmetrics fallito ({type(e).__name__}: {e}) — file MOT comunque salvati.")
            else:
                print("[MOT] eval_tracker/motmetrics non importabili (pip install motmetrics?). Calcolo a mano:")
                print(f"[MOT]   cd src/evaluation && python -c \"from eval_tracker import "
                      f"compute_tracking_metrics as f; f('{mot_gt_dir}','{mot_pred_dir}','out.csv')\"")

        import pprint
        pprint.pprint(final_results)

        final_results['tp'] = total_tp
        final_results['fp'] = total_fp
        final_results['fn'] = total_fn
        final_results['precision'] = precision
        final_results['recall'] = recall
        final_results['f1'] = f1
        final_results.update(tm)   # mota, idf1, id_switches, motp_iou, track_*, ...

        # wandb (se una run è attiva: l'eval può girare dentro uno script di training)
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
        Rendering per l'eval autoregressivo:
          - traiettorie dei track (blu tratteggiato, fade)
          - GT presente (verde chiaro)
          - detection correnti con id track + score (arancione)
          - predizioni FUTURE del forecasting head (rosso, fade per step) — se future_pred != None
          - GT FUTURO (verde, tratteggiato, fade per step) — se future_gt != None
        future_pred: np.ndarray (N, T, 4) cxcywh norm, forecast per i track validi correnti.
        future_gt:   lista di T tensori (n_t, 4) cxcywh norm (GT futuro); None se non disponibile.
        future_ids:  lista di T tensori (n_t,) id, per collegare le box GT nello stesso drone.
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

        # 4. Predizioni FUTURE (forecasting head): rosso, dissolvenza per step + traiettoria
        if future_pred is not None and len(future_pred) > 0:
            T = future_pred.shape[1]
            for n in range(future_pred.shape[0]):
                cxs, cys = [], []
                for t in range(T):
                    fb = future_pred[n, t]
                    if float(np.abs(fb).sum()) == 0:      # slot vuoto/padding
                        continue
                    a = 1.0 - 0.6 * (t / max(T - 1, 1))   # vicino pieno → lontano trasparente
                    draw(fb.tolist(), '#FF3333', '-', 1.4, alpha=a)
                    cxs.append(fb[0] * W); cys.append(fb[1] * H)
                if len(cxs) > 1:
                    ax.plot(cxs, cys, '-', color='#FF3333', lw=0.8, alpha=0.6)

        # 5. Future GT (verde medio #00B050, tratteggiato, fade per step): la traiettoria
        #    REALE, per confronto diretto con le predizioni rosse. Con future_ids collego le
        #    box dello stesso drone; altrimenti disegno solo le box per step.
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
        print("  Setting up test dataset and dataloader...")

        task_modality = getattr(self.config, 'task_modality', 'detection')
        if task_modality not in ['detection', 'tracking']:
            task_modality = 'detection'

        print(f"    Loading test dataset: {self.config.dataset_name} | Split: test")
        print(f"    Using index file: {self.config.index_path}")
        print(f"    Task modality: {task_modality}")

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

        print(f"   Test samples: {len(self.test_dataset)} | Test batches: {len(self.test_loader)}")

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
            if score < vis_score_thr:        # filtro SOLO per la visualizzazione (non tocca la mAP)
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
        """Frame di forecasting: present (giallo) + futuro GT (verde tratteggiato) +
        futuro PREDETTO (rosso), con dissolvenza per step (vicino pieno → lontano trasparente)
        e traiettorie dei centri (verde = GT, rosso = predetto)."""
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
            # present = passato più recente (indice 0) — giallo
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
                a = 1.0 - 0.6 * (t / max(T - 1, 1))          # vicino pieno → lontano trasparente
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
