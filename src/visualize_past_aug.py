#!/usr/bin/env python3
"""Visualizza la past-box augmentation (DRIFT + JUMP) sui DATI VERI del train set, PRIMA di allenare.
Costruisce il passato ORACLE con la stessa collate del training, applica `augment_past_boxes`, e
salva dei PNG: per ogni traiettoria-drone mostra oracle (verde) vs augmented (arancione), con le
box "sfasate" dal JUMP evidenziate in ROSSO, sul frame eventi + GT presente (lime).

Le box saltate sono identificate confrontando drift-solo vs drift+jump con la STESSA seed
(la differenza è esattamente il jump). NON carica il modello.

Lancialo con run_viz_past_aug.sh (imposta HDF5_PLUGIN_PATH e i path)."""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from transformers import RTDetrImageProcessor
from configs.run_configs.rtdetrv2_multiscale_past_conditioned_config import DefaultArgs
from data.dataset_factory import get_dataset
from train.trainers.train_rtdetrv2_past_conditioned import (
    collate_fn_detection_rtdetrv2_past_conditioned as collate_fn,
)

_CUSTOM_MEAN = [0.007950919680297375, 0.010960247367620468, 0.013969575986266136]
_CUSTOM_STD  = [0.048023562878370285, 0.054145995527505875, 0.066846564412117]


def augment_past_boxes(past_boxes, valid_mask, p, std, max_off,
                       jump_p=0.0, jump_std=0.0, jump_max=0.5):
    """Copia self-contained (identica al trainer) per NON dipendere dal sync del trainer.
    DRIFT (random walk correlato) + JUMP (outlier sparsi per-box → box 'sfasate')."""
    if std <= 0.0 and jump_std <= 0.0:
        return past_boxes
    B, O, P, _ = past_boxes.shape
    dev = past_boxes.device
    sel = valid_mask & (torch.rand(B, O, device=dev) < p)
    if not bool(sel.any()):
        return past_boxes
    scale = torch.rand(B, O, 1, 1, device=dev) * std
    offset = torch.cumsum(torch.randn(B, O, P, 4, device=dev) * scale, dim=2)
    offset = offset.clamp(-max_off, max_off)
    if jump_p > 0.0 and jump_std > 0.0:
        jmask = (torch.rand(B, O, P, 1, device=dev) < jump_p).float()
        jump = (torch.randn(B, O, P, 4, device=dev) * jump_std).clamp(-jump_max, jump_max)
        offset = offset + jump * jmask
    p_cx, p_cy, p_w, p_h = past_boxes.unbind(-1)
    ox, oy, ow, oh = offset.unbind(-1)
    cx = (p_cx + ox * p_w).clamp(0.0, 1.0)
    cy = (p_cy + oy * p_h).clamp(0.0, 1.0)
    w  = (p_w * (1.0 + ow)).clamp(1e-3, 1.0)
    h  = (p_h * (1.0 + oh)).clamp(1e-3, 1.0)
    aug = torch.stack([cx, cy, w, h], dim=-1)
    return torch.where(sel.unsqueeze(-1).unsqueeze(-1), aug, past_boxes)


def draw_traj(ax, traj, W, H, jumped=None, oracle=False):
    """traj (P,4) cxcywh. Se oracle=True: verde tratteggiato. Altrimenti arancione, ROSSO dove jumped."""
    P = traj.shape[0]
    cxs, cys = [], []
    for k in range(P):
        cx, cy, bw, bh = [float(v) for v in traj[k]]
        if bw <= 0 or bh <= 0:
            continue
        a = 0.28 + 0.62 * (k / max(P - 1, 1))
        if oracle:
            color, lw, style, op = '#22c55e', 1.1, '--', a
        elif jumped is not None and bool(jumped[k]):
            color, lw, style, op = '#ef4444', 1.4, '-', 1.0          # box sfasata (outlier), sottile
        else:
            color, lw, style, op = '#fb923c', 1.6, '-', a
        ax.add_patch(patches.Rectangle(((cx - bw / 2) * W, (cy - bh / 2) * H), bw * W, bh * H,
                     fill=False, edgecolor=color, lw=lw, linestyle=style, alpha=op))
        cxs.append(cx * W); cys.append(cy * H)
    line_c = '#22c55e' if oracle else '#fb923c'
    if cxs:
        ax.plot(cxs, cys, '-', color=line_c, lw=1.2, alpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_path', required=True)
    ap.add_argument('--durations', default='33,165,330')
    ap.add_argument('--num_past', type=int, default=12)
    ap.add_argument('--render_mode', default='metavision')
    ap.add_argument('--use_custom_normalization', type=int, default=0)
    ap.add_argument('--subsample', type=int, default=50)
    ap.add_argument('--n_samples', type=int, default=8)
    ap.add_argument('--aug_p', type=float, default=1.0, help='1.0 = perturba ogni track (per la viz)')
    ap.add_argument('--aug_std', type=float, default=0.05, help='drift')
    ap.add_argument('--aug_max', type=float, default=0.2)
    ap.add_argument('--jump_p', type=float, default=0.2, help='prob per-box di outlier "sfasato"')
    ap.add_argument('--jump_std', type=float, default=0.3)
    ap.add_argument('--jump_max', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = DefaultArgs()
    cfg.index_path = args.index_path if args.index_path.endswith('/') else args.index_path + '/'
    cfg.durations = [int(d) for d in args.durations.split(',')]
    cfg.render_mode = args.render_mode
    cfg.subsample = args.subsample
    cfg.num_past_annotations = args.num_past
    cfg.num_future_annotations = 0
    cfg.num_future_steps = 0
    cfg.phase = 2
    cfg.use_custom_normalization = args.use_custom_normalization

    ip = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    if args.use_custom_normalization:
        ip.image_mean = _CUSTOM_MEAN; ip.image_std = _CUSTOM_STD
    mean = np.array(ip.image_mean); std_n = np.array(ip.image_std)

    ds = get_dataset("FRED", split='train', modality='tracking', config=cfg)
    print(f"[viz] {len(ds)} finestre | render={args.render_mode} | jump_p={args.jump_p} jump_std={args.jump_std}")

    saved, i, N = 0, 0, len(ds)
    while saved < args.n_samples and i < N:
        chunk = []
        for _ in range(4):
            if i >= N:
                break
            try:
                chunk.append(ds[i])
            except Exception as e:
                print(f"[viz] skip {i}: {e}")
            i += 1
        if not chunk:
            continue
        out = collate_fn(chunk, ip, cfg, augment=False)
        if out is None:
            continue
        past = out['past_boxes']; pmask = out['past_mask']; gts = out['labels']; frames = out['frames']
        B, O, P, _ = past.shape
        # stessa seed → il DRIFT è identico; la differenza drift+jump vs drift-solo È il jump
        torch.manual_seed(args.seed)
        drift_only = augment_past_boxes(past.clone(), ~pmask, args.aug_p, args.aug_std, args.aug_max, 0.0, 0.0, args.jump_max)
        torch.manual_seed(args.seed)
        drift_jump = augment_past_boxes(past.clone(), ~pmask, args.aug_p, args.aug_std, args.aug_max, args.jump_p, args.jump_std, args.jump_max)
        jumped = (drift_jump - drift_only).abs().sum(-1) > 1e-6      # (B, O, P) True dove una box è saltata

        for b in range(B):
            main_dur = max(frames.keys())
            fr = frames[main_dur][b].numpy().transpose(1, 2, 0)
            img = np.clip(fr * std_n + mean, 0, None); img = img / max(float(img.max()), 1e-8); img = np.clip(img, 0, 1)
            Hh, Ww = img.shape[:2]
            gt_boxes = gts[b]['boxes'].numpy() if gts[b]['boxes'].numel() else np.zeros((0, 4))
            for o in [o for o in range(O) if not bool(pmask[b, o])]:
                if saved >= args.n_samples:
                    break
                njump = int(jumped[b, o].sum())
                fig, ax = plt.subplots(figsize=(7.5, 4.6))
                ax.imshow(img, extent=[0, Ww, Hh, 0])
                draw_traj(ax, past[b, o].numpy(), Ww, Hh, oracle=True)                      # oracle verde
                draw_traj(ax, drift_jump[b, o].numpy(), Ww, Hh, jumped=jumped[b, o].numpy())  # augmented + jump
                if o < len(gt_boxes):
                    g = gt_boxes[o]
                    ax.add_patch(patches.Rectangle(((g[0] - g[2] / 2) * Ww, (g[1] - g[3] / 2) * Hh),
                                 g[2] * Ww, g[3] * Hh, fill=False, edgecolor='#69f0ae', lw=2.5))
                ax.set_title(f"past-aug DRIFT+JUMP | b{b} drone{o} | P={P} | jump_p={args.jump_p:g} jump_std={args.jump_std:g} | "
                             f"box saltate: {njump}", fontsize=9)
                ax.set_xlim(0, Ww); ax.set_ylim(Hh, 0); ax.axis('off')
                out_png = os.path.join(args.output_dir, f"jump_{saved:02d}_b{b}_o{o}.png")
                fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
                print(f"[viz] {out_png}  (box saltate: {njump})")
                saved += 1

    print(f"[viz] DONE: {saved} PNG in {args.output_dir}  (verde=oracle, arancio=augmented, ROSSO=sfasata, lime=GT)")


if __name__ == '__main__':
    main()
