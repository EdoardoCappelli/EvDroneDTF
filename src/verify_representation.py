"""
verify_representation.py — verifica che le rappresentazioni dell'input siano calcolate bene.

Per gli STESSI window (stesso index/subsample → stesso idx = stessa scena) renderizza ogni
render_mode e:
  1) salva un confronto FIANCO A FIANCO (righe = modalita, colonne = durate) con le box GT,
     cosi vedi se i droni compaiono e se la codifica temporale ha senso;
  2) stampa statistiche numeriche per canale (range, media, % pixel non-nulli).

Gira sul CLUSTER (serve numpy/numba/h5py + i dati). Esempio:
  python verify_representation.py \
      --index /seidenas/datasets/FRED/preprocessed/test_windows_33ms.json \
      --render_modes metavision time_surface tencode \
      --durations 33 165 330 --num_samples 6 --subsample 50 \
      --output_dir ./repr_check

Suggerimento: lancialo nello stesso env del training (conda activate FRED++) cosi gli export
HDF5 sono a posto.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless: salva PNG senza display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.datasets.event_dataset_forecasting import FREDMultiDurationTensorDatasetTracking


def frame_stats(t):
    """Statistiche di un frame tensor (C,H,W): per-canale + % pixel non-nulli."""
    a = t.numpy()
    nz = float((a.max(axis=0) > 0).mean()) * 100.0       # pixel con almeno un canale > 0
    per_ch = [(float(a[c].min()), float(a[c].max()), float(a[c].mean())) for c in range(a.shape[0])]
    return nz, per_ch


def draw(ax, frame, boxes, title):
    img = np.clip(frame.permute(1, 2, 0).numpy(), 0, 1)   # (C,H,W)->(H,W,C)
    H, W = img.shape[0], img.shape[1]
    ax.imshow(img)
    for box in boxes:
        cx, cy, w, h = box.tolist()
        ax.add_patch(patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H),
                                       w * W, h * H, linewidth=2, edgecolor='lime', facecolor='none'))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# mean/std custom (tarati sul metavision) — IDENTICI al trainer (train_rtdetrv2_past_conditioned.py)
CUSTOM_MEAN = np.array([0.007950919680297375, 0.010960247367620468, 0.013969575986266136], dtype=np.float32)
CUSTOM_STD  = np.array([0.048023562878370285, 0.054145995527505875, 0.066846564412117], dtype=np.float32)


def normalize(frame, mean, std):
    """(frame - mean)/std per canale — come il processor con do_rescale=False, do_normalize=True.
    frame = tensor torch (C,H,W); ritorna numpy (C,H,W)."""
    a = frame.numpy()
    return (a - mean[:, None, None]) / std[:, None, None]


def norm_channel_stats(nf):
    """Per-canale (mean, std, min, max) del frame normalizzato. Norm buona -> mean~0, std~1."""
    return [(float(nf[c].mean()), float(nf[c].std()), float(nf[c].min()), float(nf[c].max()))
            for c in range(nf.shape[0])]


def draw_normalized(ax, nframe, boxes, title):
    """Frame normalizzato riscalato min-max in [0,1] SOLO per display (vedi se la struttura resta)."""
    img = np.transpose(nframe, (1, 2, 0))                   # (C,H,W)->(H,W,C)
    lo, hi = float(img.min()), float(img.max())
    disp = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    H, W = disp.shape[0], disp.shape[1]
    ax.imshow(np.clip(disp, 0, 1))
    for box in boxes:
        cx, cy, w, h = box.tolist()
        ax.add_patch(patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H),
                                       w * W, h * H, linewidth=2, edgecolor='lime', facecolor='none'))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='/seidenas/datasets/FRED/preprocessed/test_windows_33ms.json')
    ap.add_argument('--render_modes', nargs='+', default=['metavision', 'time_surface', 'tencode'])
    ap.add_argument('--durations', nargs='+', type=int, default=[33, 165, 330])
    ap.add_argument('--num_samples', type=int, default=6, help="quanti campioni ANNOTATI verificare")
    ap.add_argument('--subsample', type=int, default=50)
    ap.add_argument('--img_width', type=int, default=1280)
    ap.add_argument('--img_height', type=int, default=720)
    ap.add_argument('--num_past_annotations', type=int, default=12)
    ap.add_argument('--num_future_annotations', type=int, default=0)
    ap.add_argument('--output_dir', default='./repr_check')
    ap.add_argument('--show_normalized', action='store_true',
                    help="mostra anche il frame COME LO VEDE IL MODELLO: (frame-mean)/std con le "
                         "stats custom (metavision). Norm buona -> mean~0 / std~1 per canale.")
    a = ap.parse_args()

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    durations = sorted(a.durations, reverse=True)

    # Una dataset per modalita: STESSO index/subsample -> stessi windows -> stesso idx = stessa scena.
    print("Costruzione dataset per ogni render_mode...")
    ds = {}
    for m in a.render_modes:
        ds[m] = FREDMultiDurationTensorDatasetTracking(
            index_file=a.index, width=a.img_width, height=a.img_height,
            durations_ms=a.durations, render_mode=m, subsample=a.subsample,
            num_past_annotations=a.num_past_annotations,
            num_future_annotations=a.num_future_annotations,
        )
    ref = ds[a.render_modes[0]]
    print(f"Dataset: {len(ref)} campioni | modalita: {a.render_modes} | durate: {durations}")

    # Scegli idx con annotazioni (droni visibili).
    picked, idx = [], 0
    while len(picked) < a.num_samples and idx < len(ref):
        _, tgt = ref[idx]
        if len(tgt['boxes']) > 0:
            picked.append(idx)
        idx += 1
    if not picked:
        print("Nessun campione annotato trovato: abbasso --subsample o cambia --index."); return
    print(f"Campioni annotati scelti: {picked}\n")

    for si in picked:
        # Render dello STESSO idx in tutte le modalita.
        rendered = {m: ds[m][si] for m in a.render_modes}
        boxes = rendered[a.render_modes[0]][1]['boxes']

        # ── stampa numerica: RAW (+ normalizzato se richiesto) ──
        print(f"[sample {si}]  box GT: {len(boxes)}")
        for dur in durations:
            print(f"  durata {dur}ms:")
            for m in a.render_modes:
                fr = rendered[m][0][dur]
                nz, per_ch = frame_stats(fr)
                ch = "  ".join(f"c{c}[{lo:.2f},{hi:.2f}]µ{mu:.3f}" for c, (lo, hi, mu) in enumerate(per_ch))
                print(f"    {m:13s} shape{tuple(fr.shape)} nz={nz:5.2f}%  {ch}")
            if a.show_normalized:
                print("    -- normalizzato (model input, mean/std metavision) | ideale: µ~0  σ~1 --")
                for m in a.render_modes:
                    st = norm_channel_stats(normalize(rendered[m][0][dur], CUSTOM_MEAN, CUSTOM_STD))
                    ch = "  ".join(f"c{c}:µ{mu:+.2f}σ{sd:.2f}[{lo:+.1f},{hi:+.1f}]"
                                   for c, (mu, sd, lo, hi) in enumerate(st))
                    print(f"    {m:13s} {ch}")

        # ── figura: righe = modalita (+ riga '(norm)' per modalita se --show_normalized), col = durate ──
        blocks = ([x for m in a.render_modes for x in (m, m + ' (norm)')]
                  if a.show_normalized else list(a.render_modes))
        nrow, ncol = len(blocks), len(durations)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
        for r, label in enumerate(blocks):
            is_norm = label.endswith(' (norm)')
            m = label[:-7] if is_norm else label
            for c, dur in enumerate(durations):
                if is_norm:
                    nf = normalize(rendered[m][0][dur], CUSTOM_MEAN, CUSTOM_STD)
                    draw_normalized(axes[r][c], nf, boxes, f"{label}  {dur}ms")
                else:
                    draw(axes[r][c], rendered[m][0][dur], boxes, f"{label}  {dur}ms")
        fig.suptitle(f"Sample {si} | box GT: {len(boxes)}", fontsize=13)
        plt.tight_layout()
        p = out / f"sample_{si:05d}.png"
        plt.savefig(p, dpi=110, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"  -> {p}\n")

    print(f"Fatto. Immagini in {out}")


if __name__ == '__main__':
    main()
