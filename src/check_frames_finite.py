#!/usr/bin/env python3
"""Verifica se il RENDERER (tencode/time_surface/metavision) produce frame con NaN/Inf o valori
fuori scala — causa probabile del CUDA 'unspecified launch failure' (frame corrotto → NaN nel
modello → indice GPU fuori range → illegal access). Itera il dataset e controlla i frame GREZZI
(prima della normalizzazione). NON allena, non carica il modello.

Uso (dentro la dir che ha il renderer da testare, es. EvDroneDTF/src):
    export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
    export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH
    python3 check_frames_finite.py --index_path /seidenas/datasets/FRED/preprocessed/ \
        --render_mode tencode --durations 330 --subsample 1
"""
import argparse
import numpy as np
import torch
from configs.run_configs.rtdetrv2_multiscale_past_conditioned_config import DefaultArgs
from data.dataset_factory import get_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_path', required=True, help="dir preprocessed (finisce con /)")
    ap.add_argument('--render_mode', default='tencode', help="tencode | time_surface | metavision")
    ap.add_argument('--durations', default='330')
    ap.add_argument('--num_past', type=int, default=12)
    ap.add_argument('--subsample', type=int, default=1, help="1 = TUTTE le finestre")
    ap.add_argument('--max_windows', type=int, default=0, help="0 = nessun limite")
    ap.add_argument('--abs_max', type=float, default=1e4, help="segnala anche |val| oltre questa soglia")
    args = ap.parse_args()

    cfg = DefaultArgs()
    cfg.index_path = args.index_path if args.index_path.endswith('/') else args.index_path + '/'
    cfg.render_mode = args.render_mode
    cfg.durations = [int(d) for d in args.durations.split(',')]
    cfg.num_past_annotations = args.num_past
    cfg.num_future_annotations = 0

    ds = get_dataset("FRED", split='train', modality='tracking', config=cfg)
    N = len(ds) if args.max_windows == 0 else min(len(ds), args.max_windows)
    print(f"[check] {N} finestre | render={args.render_mode} | durations={cfg.durations}")

    bad_windows, big_windows, exc, checked = 0, 0, 0, 0
    for i in range(N):
        try:
            frames, _ = ds[i]
        except Exception as e:
            print(f"[EXC ] idx={i}: {type(e).__name__}: {e}")
            exc += 1
            continue
        hp = ds.windows[i].get('hdf5_path', '?') if i < len(ds.windows) else '?'
        for dur, fr in frames.items():
            t = fr if torch.is_tensor(fr) else torch.as_tensor(np.asarray(fr, dtype=np.float32))
            t = t.float()
            checked += 1
            finite = torch.isfinite(t)
            if not bool(finite.all()):
                nan = int(torch.isnan(t).sum()); inf = int(torch.isinf(t).sum())
                safe = torch.nan_to_num(t)
                print(f"[NaN ] idx={i} dur={dur}ms  nan={nan} inf={inf}  "
                      f"range=[{float(safe.min()):.3g},{float(safe.max()):.3g}]  hdf5={hp}")
                bad_windows += 1
            else:
                amax = float(t.abs().max())
                if amax > args.abs_max:
                    print(f"[BIG ] idx={i} dur={dur}ms  |max|={amax:.3g}  hdf5={hp}")
                    big_windows += 1
        if i % 200 == 0:
            print(f"   ...{i}/{N}  (NaN/Inf: {bad_windows} | fuori-scala: {big_windows} | exc: {exc})")

    print("=" * 60)
    print(f"[check] DONE: {checked} frame | NaN/Inf: {bad_windows} | fuori-scala(>{args.abs_max:g}): "
          f"{big_windows} | eccezioni: {exc}")
    if bad_windows or exc:
        print("[check] => il RENDERER produce frame corrotti: e' la causa del CUDA launch failure.")
    elif big_windows:
        print("[check] => frame finiti ma con valori enormi: instabilita' probabile (rinormalizzare).")
    else:
        print("[check] => frame tutti finiti e in scala: il crash NON viene dal renderer a questo livello.")


if __name__ == '__main__':
    main()
