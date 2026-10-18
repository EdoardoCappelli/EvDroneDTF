"""
Passato closed-loop per il training (use_self_past).

Il modello gira in closed loop col tracker dell'eval su alcune sequenze di training; per ogni frame si
salvano le finestre date come query-passato e il drone GT abbinato a ciascuna (-1 = sfondo).
"""
import functools
import time
from collections import deque

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from data.datasets.event_dataset_forecasting import frame_key
from evaluation.closed_loop import LegacyTracker, box_iou_cxcywh


def match_windows(anchors, gt_boxes, iou_thr):
    """Hungarian fra la box più recente di ogni finestra e i droni presenti: indice del drone o -1."""
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
    """{hdf5_path: indici delle finestre in ordine temporale}."""
    seqs = {}
    for i, w in enumerate(windows):
        seqs.setdefault(w['hdf5_path'], []).append(i)
    for path, idx in seqs.items():
        idx.sort(key=lambda i: windows[i]['window_end_time'])
    return seqs


def _make_tracker(config, P):
    """Tracker dell'eval closed-loop (legacy o tracktrack)."""
    if str(getattr(config, 'ar_tracker', 'legacy')).lower() == 'tracktrack':
        from evaluation.tracktrack_ar import TrackTrackAR, TTConfig
        return TrackTrackAR(TTConfig.from_config(config), P), True
    return LegacyTracker.from_config(config, P), False


def _lane_schedule(seq_list, n_lanes):
    """Passi del rollout: ogni corsia segue una sequenza alla volta, un frame per corsia a ogni passo."""
    queue = [k for k, (_, idx) in enumerate(seq_list) if len(idx) > 0]
    lanes = [None] * n_lanes                         # (k sequenza, prossima posizione) o None
    steps = []
    while True:
        for l in range(n_lanes):
            if lanes[l] is not None and lanes[l][1] >= len(seq_list[lanes[l][0]][1]):
                lanes[l] = None                      # sequenza finita
            if lanes[l] is None and queue:
                lanes[l] = (queue.pop(0), 0)         # prossima sequenza della coda
        step = [(l, k, p, seq_list[k][1][p]) for l, lane in enumerate(lanes) if lane is not None
                for k, p in (lane,)]
        if not step:
            return steps
        steps.append(step)
        lanes = [None if lane is None else (lane[0], lane[1] + 1) for lane in lanes]


def _collate_step(batch, image_processor, num_past):
    """Frame di un passo impilati per durata, con la GT di ciascuno (preprocessing dell'eval AR)."""
    from evaluation.evaluators.evaluator_rtdetrv2_past_conditioned import collate_autoregressive
    outs = [collate_autoregressive([item], image_processor=image_processor, num_past=num_past) for item in batch]
    frames = {d: torch.cat([o[0][d] for o in outs], dim=0) for d in outs[0][0]}
    return frames, [(o[1], o[2]) for o in outs]


@torch.no_grad()
def rollout_closed_loop(model, image_processor, config, device, dataset, seqs, keep_keys=None,
                        match_iou=0.1, max_frames=0, num_workers=0, desc="rollout", batch_seqs=1):
    """Closed loop del modello sulle sequenze `seqs` di `dataset` (subsample=1).

    batch_seqs: sequenze avanzate in parallelo, ognuna col suo tracker. keep_keys: frame da salvare
    (None = tutti, set() = solo statistiche). Ritorna (cache, stats), con
    cache[frame_key] = (finestre (N, P, 4), id GT di ogni finestra o -1).
    """
    from torchmetrics.detection import MeanAveragePrecision

    P = int(getattr(config, 'num_past_annotations', 12))
    seq_list = [(path, idx[:max_frames] if max_frames and max_frames > 0 else idx) for path, idx in seqs.items()]
    n_frames = sum(len(idx) for _, idx in seq_list)
    n_lanes = max(1, min(int(batch_seqs), len(seq_list)))
    steps = _lane_schedule(seq_list, n_lanes)
    loader = DataLoader(
        dataset, batch_sampler=[[s[3] for s in step] for step in steps], num_workers=num_workers,
        collate_fn=functools.partial(_collate_step, image_processor=image_processor, num_past=P),
    )
    lanes = []
    for _ in range(n_lanes):
        tracker, use_tt = _make_tracker(config, P)
        lanes.append({'tracker': tracker, 'hist': deque(maxlen=P)})   # hist[k] = {id: box} GT al frame t-1-k

    W = float(getattr(config, 'img_width', 1280))
    H = float(getattr(config, 'img_height', 720))

    def to_xyxy(b):   # cxcywh normalizzate -> xyxy pixel
        b = torch.as_tensor(np.asarray(b, dtype=np.float32)).reshape(-1, 4)
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1)

    metric = MeanAveragePrecision(iou_type='bbox')   # mAP del sistema (box emesse dal tracker)
    pending = {}      # (k, posizione) -> (pred, gt), aggiunti alla mAP alla fine
    was_training = model.training
    model.eval()
    cache = {}
    st = dict(frames=0, windows=0, matched=0, ghosts=0, gt=0, gt_covered=0,
              q_sum=0.0, q_n=0, anchor_sum=0.0)
    t0 = time.time()
    try:
        for step, (frames_dict, gts) in zip(steps, loader):
            rows = []
            for (l, k, p, didx), (present_labels, present_ids) in zip(step, gts):
                lane = lanes[l]
                if p == 0:                           # la corsia comincia una sequenza nuova
                    lane['tracker'].reset()
                    lane['hist'].clear()
                gt_boxes = present_labels[0]['boxes'].float().numpy() if present_labels else np.zeros((0, 4), np.float32)
                gt_all = gt_boxes                    # GT della mAP: tutti i droni presenti
                gt_ids = np.asarray(present_ids, dtype=np.int64)
                n_pair = min(len(gt_ids), len(gt_boxes))
                gt_boxes, gt_ids = gt_boxes[:n_pair], gt_ids[:n_pair]

                # query-passato del frame, come in inferenza
                valid = lane['tracker'].query_tracks()
                arr = np.asarray(lane['tracker'].query_array(valid), dtype=np.float32)      # (N, P, 4)

                # target delle finestre e qualità del passato
                m = match_windows(arr[:, 0], gt_boxes, match_iou) if len(arr) else []
                tids = np.array([int(gt_ids[j]) if j >= 0 else -1 for j in m], dtype=np.int64)
                st['frames'] += 1
                st['windows'] += len(arr)
                st['ghosts'] += int((tids < 0).sum())
                st['gt'] += n_pair
                st['gt_covered'] += int((tids >= 0).sum())
                hist = lane['hist']
                for i, j in enumerate(m):
                    if j < 0:
                        continue
                    st['matched'] += 1
                    st['anchor_sum'] += box_iou_cxcywh(arr[i, 0], gt_boxes[j])
                    tid = int(tids[i])
                    ious = [box_iou_cxcywh(arr[i, kk], hist[kk][tid])
                            for kk in range(min(P, len(hist))) if tid in hist[kk]]
                    if ious:
                        st['q_sum'] += float(np.mean(ious))
                        st['q_n'] += 1
                key = frame_key(dataset.windows[didx])
                if keep_keys is None or key in keep_keys:
                    cache[key] = (arr.copy(), tids)
                rows.append((lane, k, p, valid, arr, gt_all, gt_ids, gt_boxes))

            # forward per gruppi con lo stesso N: niente padding, la forecasting head non maschera gli slot
            groups = {}
            for j, row in enumerate(rows):
                groups.setdefault(len(row[4]), []).append(j)
            for N, js in groups.items():
                sel = torch.as_tensor(js, dtype=torch.long)
                fd = {d: f[sel].to(device) for d, f in frames_dict.items()}
                past_boxes = torch.from_numpy(np.stack([rows[j][4] for j in js])).float().to(device)  # (G, N, P, 4)
                past_mask = torch.zeros(len(js), N, dtype=torch.bool, device=device)
                out = model(event_frames=fd, past_boxes=past_boxes, past_mask=past_mask, query_mode='both')
                for g, j in enumerate(js):
                    lane, k, p, valid, arr, gt_all, gt_ids, gt_boxes = rows[j]
                    tracker = lane['tracker']
                    if out.pred_boxes.shape[1] > 0:
                        scores = out.logits[g].sigmoid().squeeze(-1).float().cpu().numpy()
                        boxes = out.pred_boxes[g].float().cpu().numpy()
                    else:
                        scores, boxes = np.zeros(0, np.float32), np.zeros((0, 4), np.float32)
                    past_b, past_s, std_b, std_s = boxes[:N], scores[:N], boxes[N:], scores[N:]
                    if use_tt:
                        fut = (out.future_boxes[g, :N].float().cpu().numpy()
                               if out.future_boxes is not None and N > 0 else None)
                        emb_p = emb_s = None
                        if tracker.cfg.w_app > 0 and getattr(out, 'hidden_states', None) is not None:
                            hs = out.hidden_states[g].detach().float().cpu().numpy()
                            emb_p, emb_s = hs[:N], hs[N:]
                        dets, _ = tracker.step(past_b, past_s, std_b, std_s, valid, fut, emb_p, emb_s)
                    else:
                        fc_next = (out.future_boxes[g, :N, 0].float().cpu().numpy()
                                   if out.future_boxes is not None and N > 0 else None)
                        dets, _, _ = tracker.step(past_b, past_s, std_b, std_s, valid, fc_next)
                    pending[(k, p)] = (
                        {'boxes': to_xyxy([d[0] for d in dets]), 'scores': torch.tensor([float(d[1]) for d in dets]),
                         'labels': torch.zeros(len(dets), dtype=torch.long)},
                        {'boxes': to_xyxy(gt_all), 'labels': torch.zeros(len(gt_all), dtype=torch.long)},
                    )
                    lane['hist'].appendleft({int(i): b for i, b in zip(gt_ids, gt_boxes)})
            if st['frames'] // 2000 > (st['frames'] - len(step)) // 2000:
                print(f"  [{desc}] {st['frames']}/{n_frames} frame ({time.time() - t0:.0f}s)", flush=True)
    finally:
        model.train(was_training)

    for key in sorted(pending):          # ordine sequenziale
        metric.update([pending[key][0]], [pending[key][1]])
    mres = metric.compute() if st['frames'] else {}
    # q e anchor_iou restano None se nessuna finestra è abbinata
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
    """Probabilità p di usare il passato closed-loop. Ritorna (p, r aggiornato, testo per il log).
    ramp: r += step se q_ref >= q_min. product_iou / product_map: p = max(ref, floor) * (epoch+1) / epochs."""
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
