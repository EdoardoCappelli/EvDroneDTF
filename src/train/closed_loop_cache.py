"""
Training curriculum con il passato prodotto dal modello

Ogni tanto il trainer (PastConditionedTrainer._refresh_self_past) fa girare il modello corrente in
eval + no_grad, in CLOSED LOOP con lo stesso tracker dell'inferenza (evaluation/closed_loop.py), su
un sottoinsieme di sequenze di training. Per ogni frame di training salva:
  - le finestre che il tracker darebbe come query-passato in quel frame (N, P, 4), come in inferenza
    (track confermati, frame persi, fantasmi, doppioni compresi);
  - per ogni finestra il drone GT corrispondente (matching ungherese fra l'ANCORA, cioè la box più
    recente, e i droni presenti, IoU ≥ match_iou) oppure -1 = sfondo.
Il collate del training, con probabilità p, usa queste finestre al posto del passato GT. I TARGET
restano sempre GT (box presente + futuro del drone abbinato; sfondo per le finestre -1).
"""
import functools
import time
from collections import deque

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset

from data.datasets.event_dataset_forecasting import frame_key
from evaluation.closed_loop import LegacyTracker, box_iou_cxcywh


def match_windows(anchors, gt_boxes, iou_thr):
    """Matching ungherese ancora (box più recente della finestra) ↔ droni presenti.
    anchors (N, 4), gt_boxes (M, 4) cxcywh normalizzate. Ritorna, per ogni finestra, l'indice del
    drone abbinato (IoU ≥ iou_thr) oppure -1."""
    out = [-1] * len(anchors)
    if len(anchors) == 0 or len(gt_boxes) == 0:
        return out
    iou = np.array([[box_iou_cxcywh(a, g) for g in gt_boxes] for a in anchors])
    rows, cols = linear_sum_assignment(-iou)
    for i, j in zip(rows, cols):
        if iou[i, j] >= iou_thr:
            out[i] = int(j)
    return out


def sequences_in_order(windows):
    """{hdf5_path: [indici delle finestre in ordine temporale]} (ordine di prima apparizione)."""
    seqs = {}
    for i, w in enumerate(windows):
        seqs.setdefault(w['hdf5_path'], []).append(i)
    for path, idx in seqs.items():
        idx.sort(key=lambda i: windows[i]['window_end_time'])
    return seqs


def _make_tracker(config, P):
    """Lo stesso tracker dell'eval closed-loop (ar_tracker: legacy | tracktrack)."""
    if str(getattr(config, 'ar_tracker', 'legacy')).lower() == 'tracktrack':
        from evaluation.tracktrack_ar import TrackTrackAR, TTConfig
        return TrackTrackAR(TTConfig.from_config(config), P), True
    return LegacyTracker.from_config(config, P), False


@torch.no_grad()
def rollout_closed_loop(model, image_processor, config, device, dataset, seqs, keep_keys=None,
                        match_iou=0.1, max_frames=0, num_workers=0, desc="rollout"):
    """Closed loop del modello sulle sequenze `seqs` ({hdf5_path: [indici in ordine]} di `dataset`,
    che deve avere frame consecutivi: subsample=1).

    keep_keys: frame_key da salvare in cache (None = tutti; set() = nessuno, solo statistiche).
    Ritorna (cache, stats):
      cache[frame_key] = (finestre float32 (N, P, 4) indice 0 = più recente, target_id int64 (N,), -1 = sfondo)
      stats: frame, finestre, abbinate, fantasmi (non abbinate), q (IoU medio box della finestra ↔ GT
             del drone abbinato negli stessi frame; None se nessuna finestra abbinata), anchor_iou,
             coverage (droni presenti con finestra), map / map50 del closed loop (box emesse dal tracker,
             come 'map'/'map_50' di evaluate_autoregressive, frame vuoti compresi)."""
    from evaluation.evaluators.evaluator_rtdetrv2_past_conditioned import collate_autoregressive
    from torchmetrics.detection import MeanAveragePrecision

    P = int(getattr(config, 'num_past_annotations', 12))
    tracker, use_tt = _make_tracker(config, P)
    order, owner = [], []
    for path, idx in seqs.items():
        idx = idx[:max_frames] if max_frames and max_frames > 0 else idx
        order += idx
        owner += [path] * len(idx)
    loader = DataLoader(
        Subset(dataset, order), batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=functools.partial(collate_autoregressive, image_processor=image_processor, num_past=P),
    )

    W = float(getattr(config, 'img_width', 1280))
    H = float(getattr(config, 'img_height', 720))

    def to_xyxy(b):   # (N, 4) cxcywh normalizzate → tensor (N, 4) xyxy pixel
        b = torch.as_tensor(np.asarray(b, dtype=np.float32)).reshape(-1, 4)
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1)

    metric = MeanAveragePrecision(iou_type='bbox')   # mAP del sistema in closed loop (box emesse dal tracker)
    was_training = model.training
    model.eval()
    cache = {}
    st = dict(frames=0, windows=0, matched=0, ghosts=0, gt=0, gt_covered=0,
              q_sum=0.0, q_n=0, anchor_sum=0.0)
    hist = deque(maxlen=P)        # GT dei P frame precedenti: hist[k] = {id: box} al frame t-1-k
    prev = None
    t0 = time.time()
    try:
        for n, batch in enumerate(loader):
            seq = owner[n]
            if seq != prev:
                tracker.reset()
                hist.clear()
                prev = seq
            frames_dict, present_labels, present_ids = batch[0], batch[1], batch[2]
            gt_boxes = present_labels[0]['boxes'].float().numpy() if present_labels else np.zeros((0, 4), np.float32)
            gt_all = gt_boxes                       # tutti i droni presenti: GT della mAP (come l'eval)
            gt_ids = np.asarray(present_ids, dtype=np.int64)
            n_pair = min(len(gt_ids), len(gt_boxes))
            gt_boxes, gt_ids = gt_boxes[:n_pair], gt_ids[:n_pair]

            # ── query-passato del frame: esattamente quelle dell'inferenza ──
            valid = tracker.query_tracks()
            arr = np.asarray(tracker.query_array(valid), dtype=np.float32)          # (N, P, 4)

            # ── target: ancora ↔ droni presenti; statistiche sulla qualità del passato ──
            m = match_windows(arr[:, 0], gt_boxes, match_iou) if len(arr) else []
            tids = np.array([int(gt_ids[j]) if j >= 0 else -1 for j in m], dtype=np.int64)
            st['frames'] += 1
            st['windows'] += len(arr)
            st['ghosts'] += int((tids < 0).sum())
            st['gt'] += n_pair
            st['gt_covered'] += int((tids >= 0).sum())
            for i, j in enumerate(m):
                if j < 0:
                    continue
                st['matched'] += 1
                st['anchor_sum'] += box_iou_cxcywh(arr[i, 0], gt_boxes[j])
                tid = int(tids[i])
                ious = [box_iou_cxcywh(arr[i, k], hist[k][tid])
                        for k in range(min(P, len(hist))) if tid in hist[k]]
                if ious:
                    st['q_sum'] += float(np.mean(ious))
                    st['q_n'] += 1
            key = frame_key(dataset.windows[order[n]])
            if keep_keys is None or key in keep_keys:
                cache[key] = (arr.copy(), tids)

            # ── forward + update del tracker (come evaluate_autoregressive) ──
            fd = {d: f.to(device) for d, f in frames_dict.items()}
            past_boxes = torch.from_numpy(arr).float().unsqueeze(0).to(device)
            past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=device)
            out = model(event_frames=fd, past_boxes=past_boxes, past_mask=past_mask, query_mode='both')
            if out.pred_boxes.shape[1] > 0:
                scores = out.logits[0].sigmoid().squeeze(-1).float().cpu().numpy()
                boxes = out.pred_boxes[0].float().cpu().numpy()
            else:
                scores, boxes = np.zeros(0, np.float32), np.zeros((0, 4), np.float32)
            N = past_boxes.shape[1]
            past_b, past_s, std_b, std_s = boxes[:N], scores[:N], boxes[N:], scores[N:]
            if use_tt:
                fut = (out.future_boxes[0, :N].float().cpu().numpy()
                       if out.future_boxes is not None and N > 0 else None)
                emb_p = emb_s = None
                if tracker.cfg.w_app > 0 and getattr(out, 'hidden_states', None) is not None:
                    hs = out.hidden_states[0].detach().float().cpu().numpy()
                    emb_p, emb_s = hs[:N], hs[N:]
                dets, _ = tracker.step(past_b, past_s, std_b, std_s, valid, fut, emb_p, emb_s)
            else:
                fc_next = (out.future_boxes[0, :N, 0].float().cpu().numpy()
                           if out.future_boxes is not None and N > 0 else None)
                dets, _, _ = tracker.step(past_b, past_s, std_b, std_s, valid, fc_next)
            # mAP del sistema: le box emesse dal tracker (score già > ar_conf_thr), come in evaluate_autoregressive
            metric.update(
                [{'boxes': to_xyxy([d[0] for d in dets]), 'scores': torch.tensor([float(d[1]) for d in dets]),
                  'labels': torch.zeros(len(dets), dtype=torch.long)}],
                [{'boxes': to_xyxy(gt_all), 'labels': torch.zeros(len(gt_all), dtype=torch.long)}],
            )

            hist.appendleft({int(i): b for i, b in zip(gt_ids, gt_boxes)})
            if (n + 1) % 2000 == 0:
                print(f"  [{desc}] {n + 1}/{len(order)} frame ({time.time() - t0:.0f}s)", flush=True)
    finally:
        model.train(was_training)

    mres = metric.compute() if st['frames'] else {}
    # q / anchor_iou = None se non c'è nessuna finestra abbinata (es. tracker che non conferma track
    # sulla sequenza): "non misurabile", non "pessimo" → il trainer ripiega sul q del training.
    stats = dict(
        map=float(mres['map']) if 'map' in mres else None,
        map50=float(mres['map_50']) if 'map_50' in mres else None,
        frames=st['frames'], windows=st['windows'], matched=st['matched'], ghosts=st['ghosts'],
        q=st['q_sum'] / st['q_n'] if st['q_n'] else None,
        anchor_iou=float(st['anchor_sum'] / st['matched']) if st['matched'] else None,
        coverage=st['gt_covered'] / st['gt'] if st['gt'] else 0.0,
        ghosts_per_frame=st['ghosts'] / max(st['frames'], 1),
        cached=len(cache), seconds=time.time() - t0,
    )
    return cache, stats


SCHEDULES = ('ramp', 'product_iou', 'product_map')


def _fmt(v):
    return 'n/a' if v is None else f"{v:.3f}"


def schedule_p(schedule, r, q_ref, map_ref, epoch, epochs, q_min=0.6, step=0.05, p_max=0.8, floor=0.0):
    """Probabilità p che un frame di training usi il passato closed-loop (il resto usa il passato GT).
    Ritorna (p, r aggiornato, spiegazione per il log).
      ramp:        r += step se q_ref ≥ q_min, altrimenti r resta; p = min(p_max, r)
      product_iou: p = min(p_max, max(q_ref, floor) · (epoch+1)/epochs)
      product_map: p = min(p_max, max(map_ref, floor) · (epoch+1)/epochs)
    q_ref: IoU medio finestra closed-loop ↔ GT; map_ref: mAP50 del closed loop (box del tracker), entrambi
    dall'ultimo rollout (val, o train se il val non li misura). None (non misurabile) vale 0, + floor.
    floor = 0.5 riproduce la formula max(map_val, 0.5)·epoch/epochs."""
    if schedule == 'ramp':
        if q_ref is not None and q_ref >= q_min:
            r = min(1.0, r + step)
        return min(p_max, r), r, f"ramp: q_ref {_fmt(q_ref)} (soglia {q_min:.2f}) → r {r:.2f}"
    if schedule in ('product_iou', 'product_map'):
        ref, name = (q_ref, 'q') if schedule == 'product_iou' else (map_ref, 'mAP50')
        progress = min(1.0, (epoch + 1) / max(1, epochs))
        base = max(ref if ref is not None else 0.0, floor)
        return (min(p_max, base * progress), r,
                f"{schedule}: max({name}_ref {_fmt(ref)}, floor {floor:.2f}) × progresso {progress:.2f}")
    raise ValueError(f"self_past_schedule sconosciuto: {schedule!r} ({' | '.join(SCHEDULES)})")
