import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HDF5_PLUGIN_PATH'] = '/seidenas/datasets/FRED/plugins'

import argparse
import subprocess
import sys
from collections import deque

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms as tv_nms
from tqdm import tqdm
from transformers import RTDetrImageProcessor

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.datasets.event_dataset_forecasting import FREDMultiDurationTensorDatasetTracking
from models.Multiscale_ERT_Detr import MultiDurationRTDetrAttention


# ── Palette colori ─────────────────────────────────────────────────────────────
C_PAST_GT = '#4169E1'   # blu       — traiettoria predetta (track NON usato come past)
C_TRACK_USED = '#FFD700'   # giallo    — traccia valida, effettivamente usata come past input
C_PRESENT_GT = '#00FF00'   # verde     — box GT presente (verità)
C_PRED = '#FF8C00'   # arancione — detection predette


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILS
# ══════════════════════════════════════════════════════════════════════════════

def box_iou_cxcywh(a: np.ndarray, b: np.ndarray) -> float:
    """IoU tra due box (4,) cxcywh normalizzate."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def box_center_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distanza euclidea normalizzata tra i centri di due box cxcywh."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def cxcywh_to_xyxy(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """(N,4) cxcywh norm → (N,4) xyxy pixel (per l'NMS)."""
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack([
        (cx - bw / 2) * img_w,
        (cy - bh / 2) * img_h,
        (cx + bw / 2) * img_w,
        (cy + bh / 2) * img_h,
    ], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class Track:
    """
    Finestra scorrevole delle ultime P box predette dal modello.

    Le box nel deque sono ordinate dalla più vecchia (indice 0) alla più
    recente (indice -1), in modo da poter calcolare la velocità come
    differenza tra gli ultimi due elementi.
    """
    def __init__(self, box: np.ndarray, score: float, P: int):
        self.boxes = deque([box], maxlen=P)
        self.scores = deque([score], maxlen=P)
        self.missed = 0

    # ── Stato ──────────────────────────────────────────────────────────────────

    def update(self, box: np.ndarray, score: float) -> None:
        self.boxes.append(box)
        self.scores.append(score)
        self.missed = 0

    @property
    def last(self) -> np.ndarray:
        """Box più recente."""
        return self.boxes[-1]

    @property
    def predicted_next(self) -> np.ndarray:
        """
        Posizione stimata al frame successivo per estrapolazione lineare.
        Con un solo punto disponibile restituisce semplicemente l'ultimo box.
        """
        if len(self.boxes) < 2:
            return self.last
        velocity = self.boxes[-1] - self.boxes[-2]          # (4,)
        return np.clip(self.last + velocity, 0.0, 1.0)

    # ── Validità ───────────────────────────────────────────────────────────────

    def is_valid(self, P: int, conf_thr: float, coher_thr: float) -> bool:
        """
        Il track è valido (e può essere inviato al modello come past) se:
          1. La finestra è piena (P box disponibili).
          2. La confidenza media supera conf_thr.
          3. Ogni coppia di box consecutive ha IoU >= coher_thr.
        """
        if len(self.boxes) < P:
            return False
        if float(np.mean(self.scores)) < conf_thr:
            return False
        bs = list(self.boxes)
        return all(
            box_iou_cxcywh(a, b) >= coher_thr
            for a, b in zip(bs[:-1], bs[1:])
        )

    # ── Esportazione ───────────────────────────────────────────────────────────

    def trajectory_recent_first(self, P: int) -> np.ndarray:
        """(P, 4) con indice 0 = più recente — formato atteso dal modello."""
        return np.stack(list(self.boxes)[::-1], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def best_match_track(
    tracks:   list,
    box:      np.ndarray,
    iou_thr:  float,
    dist_thr: float,
    exclude:  set,
) -> 'Track | None':
    """
    Cerca il track esistente migliore per una detection, in due passaggi:

    1. IoU tra la posizione PREDETTA del track e la nuova box.
       Usa la proiezione con velocità, quindi funziona anche quando il drone
       si è spostato molto tra un frame e l'altro.

    2. Fallback su distanza euclidea tra centri (normalizzata).
       Utile quando il drone è piccolo o si muove tanto e le box non si
       sovrappongono affatto (IoU = 0 ovunque).

    Restituisce None se nessun track supera entrambe le soglie.
    """
    # ── Passo 1: IoU sulla posizione predetta ───────────────────────────────
    best_iou, best_iou_track = iou_thr, None
    for t in tracks:
        if t in exclude:
            continue
        v = box_iou_cxcywh(t.predicted_next, box)
        if v >= best_iou:
            best_iou, best_iou_track = v, t
    if best_iou_track is not None:
        return best_iou_track

    # ── Passo 2: distanza euclidea tra centri (fallback) ────────────────────
    best_dist, best_dist_track = dist_thr, None
    for t in tracks:
        if t in exclude:
            continue
        d = box_center_dist(t.predicted_next, box)
        if d < best_dist:
            best_dist, best_dist_track = d, t
    return best_dist_track


# ══════════════════════════════════════════════════════════════════════════════
# COLLATE  (batch_size = 1)
# ══════════════════════════════════════════════════════════════════════════════

def build_collate(image_processor, num_past: int):
    """
    Collate per la generazione video del detector past-conditioned.

    Output (batch=1):
        frames_dict     Dict[int, Tensor(1, C, H, W)]   frame normalizzati
        present_labels  List[Dict]                       GT droni presenti (cxcywh norm)
        past_boxes      Tensor(1, max_obj, P, 4)         traiettorie passate
        past_mask       Tensor(1, max_obj) bool          True = padding
    """
    def collate(batch):
        assert len(batch) == 1, "batch_size deve essere 1 per la generazione video"
        frames_raw, target = batch[0]

        present_ids = target.get('ids',        torch.zeros(0, dtype=torch.int64))
        past_ids_list = target.get('past_ids',   [])
        past_box_list = target.get('past_boxes', [])

        # valid_ids = droni presenti ∩ presenti in TUTTI gli step passati
        if len(past_ids_list) == 0 or len(present_ids) == 0:
            valid_ids = []
        else:
            valid_ids = set(present_ids.tolist())
            for pid in past_ids_list:
                valid_ids &= set(pid.tolist())
            valid_ids = sorted(valid_ids)

        # Frame normalizzati + GT presente
        frames_dict, present_labels = {}, None
        for d, img_tensor in frames_raw.items():
            img_np = img_tensor.permute(1, 2, 0).numpy()
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

        # past_boxes (1, max_obj, P, 4) + past_mask
        max_obj = len(valid_ids)
        past_boxes = torch.zeros(1, max_obj, num_past, 4)
        past_mask = torch.ones(1, max_obj, dtype=torch.bool)
        if max_obj > 0:
            past_mask[0, :max_obj] = False
            id_to_past = {}
            for p_idx, (pid_t, pb_t) in enumerate(zip(past_ids_list, past_box_list)):
                for k, pid in enumerate(pid_t.tolist()):
                    id_to_past.setdefault(pid, {})[p_idx] = pb_t[k]
            for di, vid in enumerate(valid_ids):
                for p_idx in range(num_past):
                    past_boxes[0, di, p_idx] = id_to_past[vid][p_idx]

        return frames_dict, present_labels, past_boxes, past_mask

    return collate


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _base_frame(frames_dict: dict) -> np.ndarray:
    """Estrae il frame a durata massima come array HWC float [0,1]."""
    img = frames_dict[max(frames_dict.keys())][0].cpu().float().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    else:
        img = img[0]
    return np.clip(img, 0.0, 1.0)


def _make_fig(img_np: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#111111')
    if img_np.ndim == 2:
        ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(img_np, vmin=0, vmax=1)
    ax.axis('off')
    return fig, ax


def _frame_np(frames_dict: dict, d) -> np.ndarray:
    """Frame di UNA durata come array HWC (o HW) float [0,1]."""
    img = frames_dict[d][0].cpu().float().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    else:
        img = img[0]
    return np.clip(img, 0.0, 1.0)


def _make_multi_fig(frames_dict: dict):
    """
    Figura multi-pannello: un pannello per durata (crescente), ciascuno mostra il
    proprio frame accumulato. Ritorna (fig, main_ax), dove main_ax è il pannello
    della durata MASSIMA — l'unico su cui si disegnano le box.
    """
    durs = sorted(frames_dict.keys())
    n = len(durs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    fig.patch.set_facecolor('#111111')
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, durs):
        ax.set_facecolor('#111111')
        img = _frame_np(frames_dict, d)
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(f'{d} ms', color='white', fontsize=9)
        ax.axis('off')
    main_ax = axes[-1]   # durata massima (durs è crescente)
    return fig, main_ax


def _fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buf.reshape(h_px, w_px, 4)[:, :, :3].copy()
    plt.close(fig)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_box(ax, box: np.ndarray, color: str, img_w: int, img_h: int,
              linestyle='-', lw=1.5, alpha=1.0, label=None) -> None:
    if not isinstance(box, np.ndarray) or box.shape != (4,):
        return
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    ax.add_patch(patches.Rectangle(
        (x1, y1), bw * img_w, bh * img_h,
        linewidth=lw, edgecolor=color, facecolor='none',
        linestyle=linestyle, alpha=alpha,
    ))
    if label:
        ax.text(x1, max(y1 - 2, 0), label,
                fontsize=6, color=color, fontfamily='monospace',
                bbox=dict(facecolor='black', alpha=0.5, pad=0.1, linewidth=0))


def render(
    frames_dict, present_labels, past_boxes, pred_boxes, scores,
    frame_idx: int, query_mode: str, score_thr: float, img_w: int, img_h: int,
    draw: str = 'both',
) -> np.ndarray:
    """
    Rendering modalità standard (passato GT):
      - traiettoria GT passata (blu tratteggiato, fade con l'età)   [GT   → draw in gt/both]
      - GT presente (verde)                                          [GT   → draw in gt/both]
      - detection predette sopra soglia (arancione)                 [pred → draw in pred/both]
    `draw`: 'gt' = solo ground-truth | 'pred' = solo predizioni/oracle | 'both' = sovrapposti.
    I frame di TUTTE le accumulazioni sono mostrati affiancati; le box vengono
    disegnate solo sul pannello della durata massima.
    """
    fig, ax = _make_multi_fig(frames_dict)

    # 1. Traiettoria GT passata (elemento GT)
    if draw in ('gt', 'both') and past_boxes is not None and past_boxes.shape[1] > 0:
        pb = past_boxes[0].cpu().numpy()    # (max_obj, P, 4)
        P = pb.shape[1]
        for drone in range(pb.shape[0]):
            for p_idx in range(P):
                box = pb[drone, p_idx]
                if box.sum() == 0:
                    continue
                alpha = max(0.2, 1.0 - p_idx / max(P, 1) * 0.8)
                _draw_box(ax, box, C_PAST_GT, img_w, img_h, '--', 1.2, alpha)

    # 2. GT presente (elemento GT)
    if draw in ('gt', 'both') and present_labels is not None:
        for box in present_labels[0]['boxes'].cpu().numpy():
            _draw_box(ax, box, C_PRESENT_GT, img_w, img_h, lw=2.0, label='GT')

    # 3. Detection predette (elemento predetto)
    n_det = 0
    if draw in ('pred', 'both') and pred_boxes is not None and pred_boxes.shape[0] > 0:
        for box, sc in zip(pred_boxes.cpu().numpy(), scores.cpu().numpy()):
            if sc >= score_thr:
                _draw_box(ax, box, C_PRED, img_w, img_h, lw=2.0, label=f'{sc:.2f}')
                n_det += 1

    fig.suptitle(
        f'Frame {frame_idx:05d}  |  mode={query_mode}  |  draw={draw}  |  thr={score_thr:.2f}  |  det={n_det}',
        fontsize=9, color='white',
    )
    _handles = []
    if draw in ('gt', 'both'):
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PAST_GT,    linestyle='--', label='Past GT'))
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PRESENT_GT, linestyle='-',  label='Present GT'))
    if draw in ('pred', 'both'):
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PRED,       linestyle='-',  label='Detection'))
    ax.legend(
        handles=_handles,
        loc='upper right', fontsize=6,
        facecolor='#222222', edgecolor='#555555', labelcolor='white', framealpha=0.75,
    )
    return _fig_to_bgr(fig)


def render_autoregressive(
    frames_dict, present_labels, tracks: list, current_dets: list,
    used_tracks: set, frame_idx: int, img_w: int, img_h: int,
    draw: str = 'both',
) -> np.ndarray:
    """
    Rendering modalità autoregressiva:
      - traiettorie dei track predetti (blu=non usato, giallo=usato come past) [pred → draw in pred/both]
      - GT presente (verde)                                                     [GT   → draw in gt/both]
      - detection correnti con score (arancione)                               [pred → draw in pred/both]
    `draw`: 'gt' = solo ground-truth | 'pred' = solo predizioni | 'both' = sovrapposti.
    I frame di TUTTE le accumulazioni sono mostrati affiancati; le box vengono
    disegnate solo sul pannello della durata massima.
    """
    fig, ax = _make_multi_fig(frames_dict)

    # 1. Traiettorie dei track (predette → giallo=usato come past, blu=non usato)
    for t in (tracks if draw in ('pred', 'both') else []):
        color = C_TRACK_USED if t in used_tracks else C_PAST_GT
        bs = list(t.boxes)
        n = max(len(bs) - 1, 1)
        for j, box in enumerate(bs):
            alpha = max(0.2, 0.3 + 0.6 * j / n)
            _draw_box(ax, box, color, img_w, img_h, '--', 1.0, alpha)

    # 2. GT presente (elemento GT)
    if draw in ('gt', 'both') and present_labels is not None:
        for box in present_labels[0]['boxes'].cpu().numpy():
            _draw_box(ax, box, C_PRESENT_GT, img_w, img_h, lw=2.0, label='GT')

    # 3. Detection correnti (elemento predetto)
    for box, score in (current_dets if draw in ('pred', 'both') else []):
        _draw_box(ax, box, C_PRED, img_w, img_h, lw=2.0, label=f'{score:.2f}')

    fig.suptitle(
        f'Frame {frame_idx:05d}  |  AUTOREG  |  draw={draw}  |  tracks={len(tracks)}  |  det={len(current_dets)}',
        fontsize=9, color='white',
    )
    _handles = []
    if draw in ('pred', 'both'):
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PAST_GT,    linestyle='--', label='Track predetto (non usato)'))
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_TRACK_USED, linestyle='--', label='Track usato (past input)'))
    if draw in ('gt', 'both'):
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PRESENT_GT, linestyle='-',  label='Present GT'))
    if draw in ('pred', 'both'):
        _handles.append(patches.Patch(facecolor='none', edgecolor=C_PRED,       linestyle='-',  label='Detection'))
    ax.legend(
        handles=_handles,
        loc='upper right', fontsize=6,
        facecolor='#222222', edgecolor='#555555', labelcolor='white', framealpha=0.75,
    )
    return _fig_to_bgr(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN — trova segmenti contigui single/multi drone
# ══════════════════════════════════════════════════════════════════════════════

def scan_segments(windows, want, min_len):
    """Run contigui (stessa sequenza) di frame single/multi-drone >= min_len.
    want: 'single' (1 drone) o 'multi' (>=2). Ritorna (seq, start, len, avg_drones)."""
    def count(w):
        return len(w.get('annotations', [])) if w.get('has_annotations', False) else 0
    def ok(w):
        c = count(w)
        return (c == 1) if want == 'single' else (c >= 2)

    segments, i, n = [], 0, len(windows)
    while i < n:
        if not ok(windows[i]):
            i += 1
            continue
        seq = windows[i]['hdf5_path']
        j = i
        while j < n and windows[j]['hdf5_path'] == seq and ok(windows[j]):
            j += 1
        if j - i >= min_len:
            avg = float(np.mean([count(windows[k]) for k in range(i, j)]))
            segments.append((seq, i, j - i, avg))
        i = j
    return segments


def _run_scan(args):
    """Elenca i segmenti single/multi abbastanza lunghi e termina."""
    dataset = FREDMultiDurationTensorDatasetTracking(
        index_file=args.index, width=args.img_width, height=args.img_height,
        durations_ms=args.durations, render_mode=args.render_mode,
        subsample=1, num_past_annotations=args.num_past_steps, num_future_annotations=0,
    )
    windows = dataset.windows
    min_len = args.clip_seconds * args.fps
    wants = ['single', 'multi'] if args.drone_filter == 'any' else [args.drone_filter]
    for want in wants:
        segs = sorted(scan_segments(windows, want, min_len), key=lambda s: -s[2])
        print(f"\n=== Segmenti '{want}' >= {min_len} frame ({args.clip_seconds}s @ {args.fps}fps) ===")
        if not segs:
            print("  (nessun segmento abbastanza lungo)")
        for seq, start, length, avg in segs[:20]:
            print(f"  --start_frame {start}  len={length} (~{length/args.fps:.0f}s)  "
                  f"avg_drones={avg:.2f}  seq={os.path.basename(seq)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    if args.scan:
        _run_scan(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  query_mode: {args.query_mode}')

    # ── Processor ──────────────────────────────────────────────────────────────
    image_processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_v2_r18vd')
    print("Using custom normalization")
    image_processor.image_mean = [0.007950919680297375, 0.010960247367620468, 0.013969575986266136]
    image_processor.image_std = [0.048023562878370285, 0.054145995527505875, 0.066846564412117]

    # ── Modello ────────────────────────────────────────────────────────────────
    model = MultiDurationRTDetrAttention(
        checkpoint = 'PekingU/rtdetr_v2_r18vd',
        num_labels = 1,
        num_past_steps = args.num_past_steps,
        durations_ms = args.durations,
        phase = 2,
        num_standard_queries = args.num_standard_queries,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'  Checkpoint:      {args.checkpoint}')
    print(f'  Chiavi mancanti: {len(missing)}  |  inattese: {len(unexpected)}')
    if 'epoch' in ckpt:
        print(f'  Epoca:           {ckpt["epoch"]}')

    model.to(device).eval()
    print(f'Modello: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M param')

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = FREDMultiDurationTensorDatasetTracking(
        index_file = args.index,
        width = args.img_width,
        height = args.img_height,
        durations_ms = args.durations,
        render_mode = args.render_mode,
        subsample = 1,
        num_past_annotations = args.num_past_steps,
        num_future_annotations = 0,
    )
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.num_workers,
        collate_fn = build_collate(image_processor, args.num_past_steps),
        pin_memory = torch.cuda.is_available(),
    )
    print(f'Campioni nel test set: {len(dataset)}')

    # Dimensioni del frame dopo il processor
    sample_frames, *_ = next(iter(loader))
    d0 = list(sample_frames.keys())[0]
    img_h = sample_frames[d0].shape[2]
    img_w = sample_frames[d0].shape[3]
    print(f'Dimensioni frame processor: {img_w}×{img_h}')
    del sample_frames

    # ── Video writer ───────────────────────────────────────────────────────────
    suffix = 'autoreg' if args.autoregressive else args.query_mode
    out_path = args.output.replace('.mp4', f'_{suffix}.mp4')
    raw_path = out_path.replace('.mp4', '_raw.avi')
    writer = None
    n_written = 0

    tracks: list[Track] = []
    prev_seq = None
    P = args.num_past_steps

    start = args.start_frame
    n_clip = args.clip_seconds * args.fps if args.clip_seconds > 0 else args.max_frames
    end = min(start + n_clip, len(dataset))
    print(f"Clip: frame [{start}, {end}) = {end - start} frame (~{(end - start)/args.fps:.0f}s @ {args.fps}fps)")

    # Carica SOLO i frame della clip (evita di scorrere tutto il dataset fino a start)
    from torch.utils.data import Subset
    clip_loader = DataLoader(
        Subset(dataset, list(range(start, end))),
        batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=build_collate(image_processor, args.num_past_steps),
        pin_memory=torch.cuda.is_available(),
    )

    for local_idx, batch in enumerate(tqdm(clip_loader, desc='Generazione video', total=end - start)):
        idx = start + local_idx   # indice globale nel dataset

        frames_dict, present_labels, gt_past, gt_mask = batch
        frames_dict = {d: f.to(device) for d, f in frames_dict.items()}

        # reset tracker al cambio sequenza (robustezza se la clip tocca un confine)
        if args.autoregressive:
            seq_id = dataset.windows[idx]['hdf5_path']
            if seq_id != prev_seq:
                tracks = []
                prev_seq = seq_id

        # ── Costruzione del passato ────────────────────────────────────────────
        if args.autoregressive:
            valid = [t for t in tracks if t.is_valid(P, args.conf_thr, args.coher_thr)]
            if valid:
                pb = np.stack([t.trajectory_recent_first(P) for t in valid], axis=0)
                past_boxes = torch.from_numpy(pb).float().unsqueeze(0).to(device)
            else:
                past_boxes = torch.zeros(1, 0, P, 4, device=device)
            past_mask = torch.zeros(1, past_boxes.shape[1], dtype=torch.bool, device=device)
            qmode = 'both'
        else:
            valid = None
            past_boxes = gt_past.to(device)
            past_mask = gt_mask.to(device)
            qmode = args.query_mode

        # ── Inferenza ─────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(
                event_frames = frames_dict,
                past_boxes = past_boxes,
                past_mask = past_mask,
                query_mode = qmode,
            )

        if outputs.pred_boxes.shape[1] > 0:
            scores = outputs.logits[0].sigmoid().squeeze(-1)
            pred_boxes = outputs.pred_boxes[0]
        else:
            scores = torch.zeros(0, device=device)
            pred_boxes = torch.zeros(0, 4, device=device)

        # ── Branch autoregressivo ──────────────────────────────────────────────
        if args.autoregressive:
            N = past_boxes.shape[1]
            past_b = pred_boxes[:N].cpu().numpy()
            past_s = scores[:N].cpu().numpy()
            std_b = pred_boxes[N:].cpu().numpy()
            std_s = scores[N:].cpu().numpy()

            updated:      set = set()
            current_dets: list = []
            past_det_boxes: list = []   # box piazzate dalle past query (hanno priorità)

            # Aggiorna i track validi con le predizioni past (mapping 1-a-1)
            for t, b, s in zip(valid, past_b, past_s):
                if s > args.conf_thr:
                    t.update(b, float(s))
                    updated.add(t)
                    current_dets.append((b, float(s)))
                    past_det_boxes.append(b)
                else:
                    t.missed += 1

            # Associa le detection standard a track esistenti o crea nuovi
            for i in np.argsort(-std_s):
                if std_s[i] <= args.conf_thr:
                    break
                b, s = std_b[i], float(std_s[i])
                # Le past query hanno priorità: scarto la standard se si sovrappone a una box del passato
                if any(box_iou_cxcywh(b, pbx) >= args.std_past_iou_thr for pbx in past_det_boxes):
                    continue
                m = best_match_track(tracks, b, args.iou_thr, args.dist_thr, exclude=updated)
                if m is not None:
                    m.update(b, s)
                    updated.add(m)
                    current_dets.append((b, s))
                else:
                    nt = Track(b, s, P)
                    tracks.append(nt)
                    updated.add(nt)
                    current_dets.append((b, s))

            # Invecchiamento ed eliminazione track morti
            for t in tracks:
                if t not in updated:
                    t.missed += 1
            tracks = [t for t in tracks if t.missed < args.max_missed]

            bgr = render_autoregressive(
                frames_dict, present_labels, tracks, current_dets, set(valid), idx, img_w, img_h,
                draw=args.draw,
            )

        # ── Branch standard (GT past) ──────────────────────────────────────────
        else:
            if args.use_nms and pred_boxes.shape[0] > 0:
                keep = tv_nms(
                    cxcywh_to_xyxy(pred_boxes, img_w, img_h).float(),
                    scores.float(),
                    args.nms_threshold,
                )
                pred_boxes = pred_boxes[keep]
                scores = scores[keep]

            bgr = render(
                frames_dict, present_labels, past_boxes, pred_boxes, scores,
                idx, qmode, args.score_thr, img_w, img_h,
                draw=args.draw,
            )

        # ── Scrittura frame ────────────────────────────────────────────────────
        if writer is None:
            h_px, w_px = bgr.shape[:2]
            writer = cv2.VideoWriter(
                raw_path, cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (w_px, h_px),
            )
        writer.write(bgr)
        n_written += 1

    if writer:
        writer.release()

    print(f'\nFrame scritti: {n_written}  ({n_written / args.fps:.1f}s @ {args.fps}fps)')
    print(f'Video grezzo:  {raw_path}')

    # ── Compressione H.264 via ffmpeg ──────────────────────────────────────────
    ffmpeg = args.ffmpeg
    has_ffmpeg = os.path.isfile(ffmpeg) or __import__('shutil').which(ffmpeg)
    if has_ffmpeg:
        cmd = [
            ffmpeg, '-y', '-i', raw_path,
            '-c:v', 'libx264', '-crf', str(args.crf),
            '-preset', 'medium', '-pix_fmt', 'yuv420p', out_path,
        ]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            os.remove(raw_path)
            print(f'Video compresso: {out_path}')
            return
        print('  ffmpeg fallito, mantengo il file grezzo.')
    else:
        print(f'  ffmpeg non trovato, mantengo il file grezzo: {raw_path}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument('--checkpoint',
                   default='/andromeda/personal/ecappelli/github_/FRED_OFA/second_architecture/src/runs/rtdetrv2_multiscale_past_conditioned_phase1_test_finetuning_rtdetrv2_only_trainable_mlp_20260623_130501/checkpoints/best_model.pt')
    p.add_argument('--index',  default='/seidenas/datasets/FRED/preprocessed/test_windows_33ms.json')
    p.add_argument('--output', default='detection.mp4')

    # Modello
    p.add_argument('--query_mode', choices=['both', 'past_only', 'standard_only'], default='both')
    p.add_argument('--draw', choices=['gt', 'pred', 'both'], default='both',
                   help="cosa disegnare: 'gt' = solo ground-truth (verde/blu) | "
                        "'pred' = solo predizioni/oracle (arancione) | 'both' = sovrapposti")
    p.add_argument('--num_standard_queries', type=int, default=50)

    # Dataset
    p.add_argument('--durations',    nargs='+', type=int, default=[33, 165, 330])
    p.add_argument('--render_mode',  default='metavision')
    p.add_argument('--img_width',    type=int, default=1280)
    p.add_argument('--img_height',   type=int, default=720)
    p.add_argument('--num_past_steps', type=int, default=10)
    p.add_argument('--use_custom_normalization', type=int, default=1)

    # Video
    p.add_argument('--fps',       type=int,   default=30)
    p.add_argument('--max_frames', type=int,  default=100000)
    p.add_argument('--start_frame',  type=int, default=0, help='primo frame della clip')
    p.add_argument('--clip_seconds', type=int, default=30, help='durata clip in secondi (0 = usa max_frames)')
    p.add_argument('--scan', type=int, default=0, help='1 = elenca i segmenti single/multi >= clip e termina')
    p.add_argument('--drone_filter', choices=['single', 'multi', 'any'], default='any',
                   help='tipo di segmento da elencare nello scan')
    p.add_argument('--score_thr', type=float, default=0.3)
    p.add_argument('--crf',       type=int,   default=23)
    p.add_argument('--ffmpeg',    default='ffmpeg')

    # NMS
    p.add_argument('--use_nms',       type=int,   default=1)
    p.add_argument('--nms_threshold', type=float, default=0.45)

    # Modalità autoregressiva
    p.add_argument('--autoregressive', type=int, default=1,
                   help='1 = closed-loop: il passato viene dai track (predizioni), non dalla GT')
    p.add_argument('--conf_thr',  type=float, default=0.35,
                   help='Confidenza minima per aggiornare/validare un track')
    p.add_argument('--coher_thr', type=float, default=0.05,
                   help='IoU minima tra box consecutive per considerare un track coerente')
    p.add_argument('--iou_thr',   type=float, default=0.0,
                   help='IoU minima (su predicted_next) per associare una detection a un track')
    p.add_argument('--dist_thr',  type=float, default=0.08,
                   help='Distanza euclidea massima tra centri (norm.) — fallback se IoU=0')
    p.add_argument('--std_past_iou_thr', type=float, default=0.3,
                   help='IoU oltre cui una detection da standard query viene scartata perché '
                        'si sovrappone a una box del passato (le past query hanno priorità)')
    p.add_argument('--max_missed', type=int,  default=6,
                   help='Frame consecutivi senza match dopo cui un track viene eliminato')

    # DataLoader
    p.add_argument('--num_workers', type=int, default=4)

    main(p.parse_args())
