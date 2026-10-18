#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zip portabile del test set FRED (intero o segmenti contigui di alcune sequenze), con lo stesso
formato di indice e HDF5 del preprocessed; eventi ricompressi con gzip.

Server:
    python utils/make_testset_zip.py pack --index /seidenas/datasets/FRED/preprocessed/test_windows_33ms.json \\
        --out fred_test_subset.zip --n-seq 6 --seconds 30 [--dry-run]
Destinazione:
    python utils/make_testset_zip.py unpack fred_test_subset.zip --dest /percorso/dati
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile

import numpy as np

ROOT = "fred_test_subset"
INDEX_NAME = "test_windows_33ms.json"
SEQ_DIR = "sequences"
STEP_US = 33333                       # passo delle finestre


def folder_of(path):
    """.../{folder}/Event/events.hdf5 -> {folder}"""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def group_sequences(windows):
    seqs = {}
    for w in windows:
        seqs.setdefault(w["hdf5_path"], []).append(w)
    for ws in seqs.values():
        ws.sort(key=lambda w: w["window_idx"])
    return seqs


def stats(ws):
    ann = np.array([bool(w.get("has_annotations")) for w in ws])
    multi = np.array([len(w.get("annotations") or []) >= 2 for w in ws])
    return {"frames": len(ws), "annotated": int(ann.sum()), "multi": int(multi.sum()),
            "seconds": len(ws) * STEP_US / 1e6}


def choose_segment(ws, n, mode):
    """Segmento [i0, i1) di n finestre: il più annotato (multi-drone peso 3) o l'inizio."""
    if n <= 0 or n >= len(ws):
        return 0, len(ws)
    if mode == "start":
        return 0, n
    w8 = np.array([bool(w.get("has_annotations")) + 2 * (len(w.get("annotations") or []) >= 2)
                   for w in ws], dtype=np.int64)
    c = np.concatenate([[0], np.cumsum(w8)])
    i0 = int(np.argmax(c[n:] - c[:-n]))
    return i0, i0 + n


def select(seqs, args):
    names = {folder_of(p): p for p in seqs}
    order = sorted(seqs, key=lambda p: (len(folder_of(p)), folder_of(p)))
    if args.seqs:
        want = [s.strip().strip("/") for s in args.seqs.split(",") if s.strip()]
        missing = [s for s in want if s not in names]
        if missing:
            sys.exit(f"sequenze non trovate nell'indice: {missing}")
        return [names[s] for s in want]
    n = len(order) if args.n_seq <= 0 else min(args.n_seq, len(order))
    if args.pick == "random":
        chosen = random.Random(args.seed).sample(order, n)
    elif args.pick == "densest":
        chosen = sorted(order, key=lambda p: -stats(seqs[p])["annotated"] / max(len(seqs[p]), 1))[:n]
    else:                                                     # spread: equidistanti nell'ordine
        idx = np.linspace(0, len(order) - 1, n).round().astype(int) if n > 1 else [0]
        chosen = [order[i] for i in dict.fromkeys(idx)]
    if args.min_multi > 0:
        have = sum(stats(seqs[p])["multi"] > 0 for p in chosen)
        extra = [p for p in order if p not in chosen and stats(seqs[p])["multi"] > 0]
        while have < args.min_multi and extra:
            # sostituisce la sequenza single-drone meno annotata
            single = [p for p in chosen if stats(seqs[p])["multi"] == 0]
            if not single:
                break
            worst = min(single, key=lambda p: stats(seqs[p])["annotated"])
            chosen[chosen.index(worst)] = extra.pop(0)
            have += 1
    return chosen


def plan(seqs, chosen, args):
    n = int(round(args.seconds * 1e6 / STEP_US)) if args.seconds > 0 else 0
    out = []
    for p in chosen:
        ws = seqs[p]
        i0, i1 = choose_segment(ws, n, args.segment)
        seg = ws[i0:i1]
        t_lo = seg[0]["window_end_time"] - int(args.max_dur_ms * 1000) - 1000
        t_hi = seg[-1]["window_end_time"] + 1000
        # stima eventi: segmento + max_dur prima dell'inizio
        look = int(np.ceil(args.max_dur_ms * 1000 / STEP_US))
        n_ev = sum(w.get("num_events", 0) for w in ws[max(0, i0 - look):i1])
        out.append({"src": p, "folder": folder_of(p), "windows": seg, "t_lo": int(t_lo), "t_hi": int(t_hi),
                    "est_events": int(n_ev), **stats(seg)})
    return out


def print_plan(items, seqs):
    tot_f = sum(i["frames"] for i in items)
    tot_e = sum(i["est_events"] for i in items)
    print(f"{'sequenza':>10} {'frame':>7} {'secondi':>8} {'annotati':>9} {'multi':>6} {'eventi (stima)':>15}")
    for i in items:
        print(f"{i['folder']:>10} {i['frames']:>7} {i['seconds']:>8.1f} {i['annotated']:>9} {i['multi']:>6} "
              f"{i['est_events']:>15,}")
    all_f = sum(len(v) for v in seqs.values())
    print(f"{'TOTALE':>10} {tot_f:>7} {tot_f * STEP_US / 1e6:>8.1f}   ({100 * tot_f / max(all_f, 1):.1f}% "
          f"dei {all_f} frame di test, {len(items)}/{len(seqs)} sequenze)")
    # 16 byte/evento, gzip circa dimezza
    print(f"dimensione stimata: ~{fmt_bytes(tot_e * 16)} grezzi, ~{fmt_bytes(tot_e * 8)} nello zip")


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _h5():
    os.environ.setdefault("HDF5_PLUGIN_PATH", "/seidenas/datasets/FRED/plugins")
    import hdf5plugin  # noqa: F401
    import h5py
    return h5py


def load_events(f, t_start, t_end):
    """Come HDF5IndexCache.load_events_by_time."""
    idx = f["CD/indexes"][:]
    off = int(f["CD/indexes"].attrs.get("offset", 0))
    a, b = np.searchsorted(idx["ts"], np.array([t_start, t_end]) + off, side="left") - 1
    a = max(0, a)
    b = min(len(idx) - 1, b + 1)
    ev = f["CD/events"][int(idx[a][0]):int(idx[b][0])]
    if len(ev) == 0:
        return ev
    s, e = np.searchsorted(ev["t"], [t_start, t_end], side="left")
    return ev[s:e]


def crop(h5py, src, dst, t_lo, t_hi):
    """Copia gli eventi in [t_lo, t_hi] con i loro CD/indexes. Ritorna (n eventi, primo indice originale)."""
    with h5py.File(src, "r") as fs:
        idx = fs["CD/indexes"][:]
        off = int(fs["CD/indexes"].attrs.get("offset", 0))
        ts = idx["ts"]
        i0 = max(0, int(np.searchsorted(ts, t_lo + off, side="left")) - 1)
        i1 = min(len(idx) - 1, int(np.searchsorted(ts, t_hi + off, side="left")) + 1)
        e0, e1 = int(idx[i0][0]), int(idx[i1][0])
        ev = fs["CD/events"][e0:e1]
        new_idx = idx[i0:i1 + 1].copy()
        first = new_idx.dtype.names[0]                        # campo "id" dell'evento
        new_idx[first] = (new_idx[first].astype(np.int64) - e0).astype(new_idx.dtype[first])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # v110: necessario per aprire con swmr=True
        with h5py.File(dst, "w", libver=("v110", "latest")) as fd:
            g = fd.create_group("CD")
            g.create_dataset("events", data=ev, compression="gzip", compression_opts=4, shuffle=True,
                             chunks=(min(max(len(ev), 1), 1 << 15),))
            di = g.create_dataset("indexes", data=new_idx)
            for k, v in fs["CD/indexes"].attrs.items():
                di.attrs[k] = v
            fd.attrs["source"] = src
            fd.attrs["t_lo_us"], fd.attrs["t_hi_us"] = t_lo, t_hi
    return len(ev), e0


def pack(args):
    with open(args.index, encoding="utf-8") as f:
        data = json.load(f)
    seqs = group_sequences(data["windows"])
    chosen = select(seqs, args)
    items = plan(seqs, chosen, args)
    print_plan(items, seqs)
    if args.dry_run:
        return

    h5py = _h5()
    out = os.path.abspath(args.out)
    stage = tempfile.mkdtemp(prefix=".fred_subset_", dir=os.path.dirname(out) or ".")
    root = os.path.join(stage, ROOT)
    try:
        windows, video_stats = [], []
        for k, it in enumerate(items):
            # stessa struttura dell'originale, il nome della sequenza viene dalla cartella
            rel = f"{SEQ_DIR}/{it['folder']}/Event/events.hdf5"
            it["rel"] = rel
            n, e0 = crop(h5py, it["src"], os.path.join(root, rel), it["t_lo"], it["t_hi"])
            for w in it["windows"]:
                w = dict(w)
                w["hdf5_path"] = rel                          # relativo: unpack lo rende assoluto
                for key in ("start_event_idx", "end_event_idx"):
                    if key in w:
                        w[key] = int(w[key]) - e0             # indici evento del file ritagliato
                windows.append(w)
            video_stats.append({"video_id": it["windows"][0].get("video_id", it["folder"]),
                                "total_windows": it["frames"], "windows_with_annotations": it["annotated"],
                                "total_events": sum(int(w.get("num_events", 0)) for w in it["windows"])})
            print(f"[{k + 1}/{len(items)}] {it['folder']}: {it['frames']} frame, {n:,} eventi")
        meta = {kk: v for kk, v in data.items() if kk != "windows"}
        meta.update({"split": data.get("split", "test"), "subset_of": os.path.abspath(args.index),
                     "total_videos": len(items), "video_stats": video_stats,
                     "total_windows": len(windows),
                     "windows_with_annotations": sum(bool(w.get("has_annotations")) for w in windows),
                     "selection": [{kk: it[kk] for kk in ("folder", "frames", "seconds", "annotated", "multi",
                                                          "t_lo", "t_hi")} for it in items],
                     "windows": windows})
        with open(os.path.join(root, INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
            f.write(README.format(n_seq=len(items), frames=len(windows), cmd=" ".join(sys.argv)))
        if args.verify:
            verify(h5py, items, root, args.verify, args.max_dur_ms)
        with zipfile.ZipFile(out, "w", allowZip64=True) as z:
            for dirpath, _, files in os.walk(root):
                for fn in sorted(files):
                    full = os.path.join(dirpath, fn)
                    arc = os.path.relpath(full, stage).replace(os.sep, "/")
                    comp = zipfile.ZIP_STORED if fn.endswith(".hdf5") else zipfile.ZIP_DEFLATED
                    z.write(full, arc, compress_type=comp)
        print(f"\nzip: {out}  ({fmt_bytes(os.path.getsize(out))})")
    finally:
        if args.keep_dir:
            print(f"cartella non compressa: {root}")
        else:
            shutil.rmtree(stage, ignore_errors=True)


def verify(h5py, items, root, n_check, max_dur_ms):
    """Confronta gli eventi di alcune finestre tra originale e ritaglio."""
    rng = random.Random(0)
    bad = 0
    for it in items:
        # prima e ultima finestra + altre a caso
        inner = it["windows"][1:-1]
        ws = it["windows"][:1] + it["windows"][-1:] + rng.sample(inner, max(0, min(n_check - 2, len(inner))))
        with h5py.File(it["src"], "r") as fs, \
                h5py.File(os.path.join(root, it["rel"]), "r", swmr=True) as fd:
            for w in ws:
                t1 = w["window_end_time"]
                t0 = t1 - int(max_dur_ms * 1000)
                a, b = load_events(fs, t0, t1), load_events(fd, t0, t1)
                if len(a) != len(b) or not all(np.array_equal(a[n], b[n]) for n in a.dtype.names):
                    bad += 1
    print(f"verifica: {sum(min(n_check, len(i['windows'])) for i in items)} finestre confrontate, "
          f"{'tutte identiche ✓' if bad == 0 else f'{bad} DIVERSE ✗'}")
    if bad:
        sys.exit("verifica fallita: lo zip non viene scritto")


def unpack(args):
    dest = os.path.abspath(args.dest)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(dest)
    root = os.path.join(dest, ROOT)
    p = os.path.join(root, INDEX_NAME)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    for w in data["windows"]:
        if not os.path.isabs(w["hdf5_path"]):
            w["hdf5_path"] = os.path.join(root, w["hdf5_path"]).replace("\\", "/")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    r = root.replace(os.sep, "/")
    print(f"estratto in {root}\n"
          f"  main.py --mode eval:  --index_path {r}/\n"
          f"  script video:         --index {r}/{INDEX_NAME}")


README = """FRED test set, sottoinsieme portabile ({n_seq} sequenze, {frames} frame da 33 ms).

- test_windows_33ms.json: indice nello stesso formato del preprocessed FRED (finestre, annotazioni).
  I percorsi hdf5_path sono relativi: 'make_testset_zip.py unpack' li rende assoluti.
- sequences/<cartella>/Event/events.hdf5: eventi ritagliati sui segmenti (CD/events + CD/indexes),
  compressi con gzip: si leggono con h5py senza il plugin ECF.
- Ogni segmento è contiguo; i primi P frame non hanno passato e gli ultimi T non hanno futuro
  completo, esattamente come all'inizio e alla fine di una sequenza.

Creato con: {cmd}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="crea lo zip (sul server)")
    p.add_argument("--index", required=True, help="test_windows_33ms.json del preprocessed")
    p.add_argument("--out", default="fred_test_subset.zip")
    p.add_argument("--n-seq", type=int, default=6, help="sequenze da includere (0 = tutte)")
    p.add_argument("--seqs", default="", help="sequenze esplicite, es. 9,230,41")
    p.add_argument("--pick", choices=["spread", "random", "densest"], default="spread")
    p.add_argument("--min-multi", type=int, default=1, help="almeno K sequenze con più droni")
    p.add_argument("--seconds", type=float, default=30.0, help="durata del segmento (0 = tutta la sequenza)")
    p.add_argument("--segment", choices=["densest", "start"], default="densest")
    p.add_argument("--max-dur-ms", type=float, default=330.0, help="accumulazione più lunga del modello")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=20, help="finestre per sequenza da confrontare (0 = no)")
    p.add_argument("--dry-run", action="store_true", help="solo selezione e stima di dimensione")
    p.add_argument("--keep-dir", action="store_true", help="tieni anche la cartella non compressa")
    u = sub.add_parser("unpack", help="estrae lo zip e rende assoluti i percorsi dell'indice")
    u.add_argument("zip")
    u.add_argument("--dest", default=".")
    args = ap.parse_args()
    pack(args) if args.cmd == "pack" else unpack(args)


if __name__ == "__main__":
    main()
