import argparse
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import cv2
import matplotlib.patches as patches
from transformers import RTDetrImageProcessor

from models.Multiscale_ERT_Detr import MultiDurationRTDetrAttention
from data.datasets.event_dataset_forecasting import FREDMultiDurationTensorDatasetTracking
from create_detection_video import (
    _make_multi_fig, _draw_box, _fig_to_bgr,
    Track, best_match_track, box_iou_cxcywh,
    scan_segments, build_collate,
    C_PRESENT_GT, C_PRED, C_TRACK_USED, C_PAST_GT,
)


# ══════════════════════════════════════════════════════════════════════════════
# COLLATE — riusa il collate detection per il passato, aggiunge il futuro GT
# ══════════════════════════════════════════════════════════════════════════════
def build_collate_forecast(image_processor, num_past: int, num_future: int):
    """Come build_collate (frames/present/past) + future_gt/future_mask (1, n_present, T, 4).
    future_gt è indicizzato per drone PRESENTE (target['ids']); per ogni step futuro trova la box
    dello stesso id (mask=False dove il drone non c'è in quello step)."""
    base = build_collate(image_processor, num_past)

    def collate(batch):
        frames_dict, present_labels, past_boxes, past_mask = base(batch)
        _, target = batch[0]
        present_ids = target.get('ids', torch.zeros(0, dtype=torch.int64)).tolist()
        fut_ids   = target.get('future_ids', [])    # lista di T tensori (id per step)
        fut_boxes = target.get('future_boxes', [])  # lista di T tensori (box per step)

        n, T = len(present_ids), num_future
        future_gt   = torch.zeros(1, n, T, 4)
        future_mask = torch.zeros(1, n, T, dtype=torch.bool)
        for t in range(min(T, len(fut_ids))):
            ids_t = fut_ids[t].tolist()
            id_to_box = {int(pid): fut_boxes[t][k] for k, pid in enumerate(ids_t)}
            for di, pid in enumerate(present_ids):
                if pid in id_to_box:
                    future_gt[0, di, t]   = id_to_box[pid]
                    future_mask[0, di, t] = True
        return frames_dict, present_labels, past_boxes, past_mask, future_gt, future_mask

    return collate


# ══════════════════════════════════════════════════════════════════════════════
# RENDER — presente (GT verde / det arancione) + futuro (GT verde-tratteg / pred arancione-tratteg)
# ══════════════════════════════════════════════════════════════════════════════
def render_forecast(
    frames_dict, present_labels, pred_boxes, scores, future_gt, future_mask, pred_future,
    score_thr: float, mode_label: str, frame_idx: int, img_w: int, img_h: int,
    tracks=None, valid_set=None, past_boxes=None,
) -> np.ndarray:
    fig, ax = _make_multi_fig(frames_dict)

    # Passato: GIALLO = track VALIDO (usato per detection+forecasting), BLU = track non ancora valido.
    if tracks is not None:                       # AR: storia di ogni track (giallo se valido)
        vset = valid_set or set()
        for t in tracks:
            color = C_TRACK_USED if t in vset else C_PAST_GT
            bs = list(t.boxes)
            n = max(len(bs) - 1, 1)
            for j, box in enumerate(bs):
                a = max(0.2, 0.3 + 0.6 * j / n)
                _draw_box(ax, box, color, img_w, img_h, '--', 1.0, a)
    elif past_boxes is not None:                 # oracle: passato GT (tutto usato) → giallo
        pb = past_boxes[0].detach().cpu().numpy()   # (N, P, 4)
        for i in range(pb.shape[0]):
            for p in range(pb.shape[1]):
                box = pb[i, p]
                if float(box.sum()) == 0.0:
                    continue
                a = max(0.2, 1.0 - p / max(pb.shape[1], 1) * 0.8)
                _draw_box(ax, box, C_TRACK_USED, img_w, img_h, '--', 1.0, a)

    # Presente: GT (verde solido) + detection del modello (arancione solido)
    if present_labels is not None:
        for box in present_labels[0]['boxes'].cpu().numpy():
            _draw_box(ax, box, C_PRESENT_GT, img_w, img_h, lw=2.0, label='GT')
    if pred_boxes is not None and pred_boxes.shape[0] > 0:
        for box, sc in zip(pred_boxes.cpu().numpy(), scores.cpu().numpy()):
            if sc >= score_thr:
                _draw_box(ax, box, C_PRED, img_w, img_h, lw=2.0)

    # Futuro GT (verde tratteggiato, fade con l'orizzonte)
    if future_gt is not None:
        fg = future_gt[0].cpu().numpy()      # (n, T, 4)
        fm = future_mask[0].cpu().numpy()    # (n, T)
        for i in range(fm.shape[0]):
            for t in range(fm.shape[1]):
                if fm[i, t]:
                    a = max(0.15, 1.0 - t / max(fm.shape[1], 1) * 0.85)
                    _draw_box(ax, fg[i, t], C_PRESENT_GT, img_w, img_h, '--', 1.0, a)

    # Futuro PREDETTO (arancione tratteggiato)
    if pred_future is not None and pred_future.shape[1] > 0:
        pf = pred_future[0].cpu().numpy()    # (N, T, 4)
        for i in range(pf.shape[0]):
            for t in range(pf.shape[1]):
                a = max(0.15, 1.0 - t / max(pf.shape[1], 1) * 0.85)
                _draw_box(ax, pf[i, t], C_PRED, img_w, img_h, '--', 1.0, a)

    fig.suptitle(
        f'Frame {frame_idx:05d}  |  FORECAST  |  past={mode_label}',
        fontsize=9, color='white',
    )
    ax.legend(
        handles=[
            patches.Patch(facecolor='none', edgecolor=C_PRESENT_GT, linestyle='-',  label='GT presente'),
            patches.Patch(facecolor='none', edgecolor=C_PRESENT_GT, linestyle='--', label='GT futuro'),
            patches.Patch(facecolor='none', edgecolor=C_PRED,       linestyle='-',  label='Detection presente'),
            patches.Patch(facecolor='none', edgecolor=C_PRED,       linestyle='--', label='Forecast futuro'),
            patches.Patch(facecolor='none', edgecolor=C_TRACK_USED, linestyle='--', label='Past usato (valido)'),
            patches.Patch(facecolor='none', edgecolor=C_PAST_GT,    linestyle='--', label='Track non valido'),
        ],
        loc='upper right', fontsize=6,
        facecolor='#222222', edgecolor='#555555', labelcolor='white', framealpha=0.75,
    )
    return _fig_to_bgr(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  past={"AR" if args.autoregressive else "oracle"}  |  T={args.num_future_steps}')

    image_processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_v2_r18vd')
    if args.use_custom_normalization:
        print("Using custom normalization")
        image_processor.image_mean = [0.007950919680297375, 0.010960247367620468, 0.013969575986266136]
        image_processor.image_std  = [0.048023562878370285, 0.054145995527505875, 0.066846564412117]

    # Modello CON testa forecasting
    model = MultiDurationRTDetrAttention(
        checkpoint='PekingU/rtdetr_v2_r18vd', num_labels=1,
        num_past_steps=args.num_past_steps, durations_ms=args.durations,
        phase=2, num_standard_queries=args.num_standard_queries,
        num_future_steps=args.num_future_steps,
        forecast_head_type=args.forecast_head_type,
        use_present_refine=bool(args.use_present_refine),
        use_cv_anchor=bool(args.use_cv_anchor),
        vel_avg_k=args.vel_avg_k,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Chiavi mancanti: {len(missing)}  |  inattese: {len(unexpected)}')
    if len(missing) > 50:
        print("  ⚠️  Molte chiavi mancanti: forse il checkpoint NON ha la testa forecasting "
              "o P/durate non combaciano.")
    model.to(device).eval()

    dataset = FREDMultiDurationTensorDatasetTracking(
        index_file=args.index, width=args.img_width, height=args.img_height,
        durations_ms=args.durations, render_mode=args.render_mode,
        subsample=1, num_past_annotations=args.num_past_steps,
        num_future_annotations=args.num_future_steps,
    )

    if args.scan:
        min_len = args.clip_seconds * args.fps
        wants = ['single', 'multi'] if args.drone_filter == 'any' else [args.drone_filter]
        for want in wants:
            segs = sorted(scan_segments(dataset.windows, want, min_len), key=lambda s: -s[2])
            print(f"\n=== Segmenti '{want}' >= {min_len} frame ===")
            for seq, s, length, avg in segs[:20]:
                print(f"  --start_frame {s}  len={length}  avg_drones={avg:.2f}  seq={os.path.basename(seq)}")
        return

    collate = build_collate_forecast(image_processor, args.num_past_steps, args.num_future_steps)
    start = args.start_frame
    n_clip = args.clip_seconds * args.fps if args.clip_seconds > 0 else args.max_frames
    end = min(start + n_clip, len(dataset))
    print(f"Clip: frame [{start}, {end}) = {end - start} frame (~{(end - start)/args.fps:.0f}s @ {args.fps}fps)")

    clip_loader = DataLoader(
        Subset(dataset, list(range(start, end))),
        batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
    )

    mode_label = 'AR' if args.autoregressive else 'oracle'
    out_path = args.output.replace('.mp4', f'_{mode_label}.mp4')
    raw_path = out_path.replace('.mp4', '_raw.avi')
    writer, n_written = None, 0
    tracks: list = []
    prev_seq = None
    P = args.num_past_steps

    for local_idx, batch in enumerate(tqdm(clip_loader, desc='Forecast video', total=end - start)):
        idx = start + local_idx
        frames_dict, present_labels, gt_past, gt_mask, future_gt, future_mask = batch
        frames_dict = {d: f.to(device) for d, f in frames_dict.items()}

        if args.autoregressive:
            seq_id = dataset.windows[idx]['hdf5_path']
            if seq_id != prev_seq:
                tracks = []
                prev_seq = seq_id

        # ── Passato: oracle (GT) o AR (tracker) ─────────────────────────────────
        if args.autoregressive:
            valid = [t for t in tracks if t.is_valid(P, args.conf_thr, args.coher_thr)]
            if valid:
                pb = np.stack([t.trajectory_recent_first(P) for t in valid], axis=0)
                past_boxes = torch.from_numpy(pb).float().unsqueeze(0).to(device)
            else:
                past_boxes = torch.zeros(1, 0, P, 4, device=device)
            past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=device)
        else:
            valid = None
            past_boxes = gt_past.to(device)
            past_mask = gt_mask.to(device)

        with torch.no_grad():
            outputs = model(event_frames=frames_dict, past_boxes=past_boxes,
                            past_mask=past_mask, query_mode='both')

        if outputs.pred_boxes.shape[1] > 0:
            scores = outputs.logits[0].sigmoid().squeeze(-1)
            pred_boxes = outputs.pred_boxes[0]
        else:
            scores = torch.zeros(0, device=device)
            pred_boxes = torch.zeros(0, 4, device=device)
        pred_future = outputs.future_boxes   # (1, N, T, 4) o None

        # ── AR: aggiorna il tracker (per costruire il passato del frame dopo) ────
        if args.autoregressive:
            N = past_boxes.shape[1]
            past_b, past_s = pred_boxes[:N].cpu().numpy(), scores[:N].cpu().numpy()
            std_b,  std_s  = pred_boxes[N:].cpu().numpy(), scores[N:].cpu().numpy()
            updated, past_det_boxes = set(), []
            for t, b, s in zip(valid, past_b, past_s):
                if s > args.conf_thr:
                    t.update(b, float(s)); updated.add(t); past_det_boxes.append(b)
                else:
                    t.missed += 1
            for i in np.argsort(-std_s):
                if std_s[i] <= args.conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                if any(box_iou_cxcywh(b, pbx) >= args.std_past_iou_thr for pbx in past_det_boxes):
                    continue
                m = best_match_track(tracks, b, args.iou_thr, args.dist_thr, exclude=updated)
                if m is not None:
                    m.update(b, s); updated.add(m)
                else:
                    tracks.append(Track(b, s, P))
            for t in tracks:
                if t not in updated:
                    t.missed += 1
            tracks = [t for t in tracks if t.missed < args.max_missed]

        bgr = render_forecast(
            frames_dict, present_labels, pred_boxes, scores, future_gt, future_mask, pred_future,
            args.score_thr, mode_label, idx, model_img_w(frames_dict), model_img_h(frames_dict),
            tracks=(tracks if args.autoregressive else None),
            valid_set=(set(valid) if (args.autoregressive and valid is not None) else None),
            past_boxes=(None if args.autoregressive else past_boxes),
        )
        if writer is None:
            h_px, w_px = bgr.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (w_px, h_px))
        writer.write(bgr)
        n_written += 1

    if writer:
        writer.release()
    print(f"Scritti {n_written} frame in {raw_path}")

    # Re-encode in mp4 (se ffmpeg c'è; altrimenti resta l'avi)
    try:
        subprocess.run(
            [args.ffmpeg, '-y', '-i', raw_path, '-c:v', 'libx264',
             '-crf', str(args.crf), '-pix_fmt', 'yuv420p', out_path],
            check=True, capture_output=True,
        )
        os.remove(raw_path)
        print(f"✅ Video: {out_path}")
    except Exception as e:
        print(f"⚠️  ffmpeg non riuscito ({e}); resta il raw: {raw_path}")


def model_img_w(frames_dict):
    d0 = list(frames_dict.keys())[0]
    return frames_dict[d0].shape[3]


def model_img_h(frames_dict):
    d0 = list(frames_dict.keys())[0]
    return frames_dict[d0].shape[2]


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--index', default='/seidenas/datasets/FRED/preprocessed/test_windows_33ms.json')
    p.add_argument('--output', default='forecast.mp4')
    p.add_argument('--query_mode', choices=['both', 'past_only', 'standard_only'], default='both')
    p.add_argument('--num_standard_queries', type=int, default=50)
    p.add_argument('--durations', nargs='+', type=int, default=[33, 165, 330])
    p.add_argument('--render_mode', default='metavision')
    p.add_argument('--img_width', type=int, default=1280)
    p.add_argument('--img_height', type=int, default=720)
    p.add_argument('--num_past_steps', type=int, default=12)
    p.add_argument('--use_custom_normalization', type=int, default=1)
    # ── forecasting head ──
    p.add_argument('--num_future_steps', type=int, default=24)
    p.add_argument('--forecast_head_type', default='transformer', choices=['transformer', 'mlp'])
    p.add_argument('--use_present_refine', type=int, default=1)
    p.add_argument('--use_cv_anchor', type=int, default=0)
    p.add_argument('--vel_avg_k', type=int, default=3)
    # ── clip / scan ──
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--max_frames', type=int, default=100000)
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--clip_seconds', type=int, default=20)
    p.add_argument('--scan', type=int, default=0)
    p.add_argument('--drone_filter', choices=['single', 'multi', 'any'], default='any')
    p.add_argument('--score_thr', type=float, default=0.3)
    p.add_argument('--crf', type=int, default=23)
    p.add_argument('--ffmpeg', default='ffmpeg')
    # ── tracker AR ──
    p.add_argument('--autoregressive', type=int, default=0)
    p.add_argument('--conf_thr', type=float, default=0.35)
    p.add_argument('--coher_thr', type=float, default=0.05)
    p.add_argument('--iou_thr', type=float, default=0.2)
    p.add_argument('--dist_thr', type=float, default=0.08)
    p.add_argument('--std_past_iou_thr', type=float, default=0.3)
    p.add_argument('--max_missed', type=int, default=6)
    p.add_argument('--num_workers', type=int, default=4)
    main(p.parse_args())
