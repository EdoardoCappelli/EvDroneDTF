#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Come make_testset_zip.py, ma con segmenti di sequenze di training (escluse quelle di validazione,
data/seq_split.py).

Server (da src/):
    python utils/make_trainset_zip.py pack --index /seidenas/datasets/FRED/preprocessed/train_windows_33ms.json [--dry-run]
Locale:
    python utils/make_trainset_zip.py unpack fred_train_subset.zip --dest <cartella>
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]            # utils/ (make_testset_zip) e src/ (data.seq_split)
import make_testset_zip as M                            # noqa: E402
from data.seq_split import sequence_holdout             # noqa: E402

M.ROOT = "fred_train_subset"
M.INDEX_NAME = "train_windows_33ms.json"
M.README = """FRED training set, segmenti portabili ({n_seq} sequenze di training, {frames} frame da 33 ms).

- train_windows_33ms.json: indice nello stesso formato del preprocessed FRED (finestre, annotazioni),
  solo sequenze dello split di training (nessuna di validazione). I percorsi hdf5_path sono
  relativi: 'make_trainset_zip.py unpack' li rende assoluti.
- sequences/<cartella>/Event/events.hdf5: eventi ritagliati sui segmenti (CD/events + CD/indexes),
  compressi con gzip: si leggono con h5py senza il plugin ECF.
- Ogni segmento è contiguo; i primi P frame non hanno passato e gli ultimi T non hanno futuro
  completo, esattamente come all'inizio e alla fine di una sequenza.

Creato con: {cmd}
"""


def train_only(group, args):
    """group_sequences ristretto alle sequenze di training."""
    def group_train(windows):
        seqs = group(windows)
        keep = {w["hdf5_path"] for w in sequence_holdout([{"hdf5_path": p} for p in seqs], "train",
                                                         val_frac=args.val_frac, seed=args.val_seed)}
        val = sorted((M.folder_of(p) for p in seqs if p not in keep), key=lambda s: (len(s), s))
        print(f"split del training (val_frac={args.val_frac}, seed={args.val_seed}): {len(keep)} sequenze "
              f"di training; {len(val)} di validazione escluse: {', '.join(val)}")
        asked_val = [s for s in (x.strip().strip("/") for x in args.seqs.split(",")) if s in val]
        if asked_val:
            sys.exit(f"sequenze di validazione, non di training: {asked_val}")
        return {p: ws for p, ws in seqs.items() if p in keep}
    return group_train


def print_plan(items, seqs):
    tot_f = sum(i["frames"] for i in items)
    tot_e = sum(i["est_events"] for i in items)
    print(f"{'sequenza':>10} {'frame':>7} {'secondi':>8} {'annotati':>9} {'multi':>6} {'eventi (stima)':>15}")
    for i in items:
        print(f"{i['folder']:>10} {i['frames']:>7} {i['seconds']:>8.1f} {i['annotated']:>9} {i['multi']:>6} "
              f"{i['est_events']:>15,}")
    all_f = sum(len(v) for v in seqs.values())
    print(f"{'TOTALE':>10} {tot_f:>7} {tot_f * M.STEP_US / 1e6:>8.1f}   ({100 * tot_f / max(all_f, 1):.1f}% "
          f"dei {all_f} frame di training, {len(items)}/{len(seqs)} sequenze)")
    # 16 byte/evento, gzip circa dimezza
    print(f"dimensione stimata: ~{M.fmt_bytes(tot_e * 16)} grezzi, ~{M.fmt_bytes(tot_e * 8)} nello zip")


def unpack(args):
    dest = os.path.abspath(args.dest)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(dest)
    root = os.path.join(dest, M.ROOT)
    p = os.path.join(root, M.INDEX_NAME)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    for w in data["windows"]:
        if not os.path.isabs(w["hdf5_path"]):
            w["hdf5_path"] = os.path.join(root, w["hdf5_path"]).replace("\\", "/")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"estratto in {root}\n  indice: {p.replace(os.sep, '/')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="crea lo zip (sul server)")
    p.add_argument("--index", required=True, help="train_windows_33ms.json del preprocessed")
    p.add_argument("--out", default="fred_train_subset.zip")
    p.add_argument("--n-seq", type=int, default=4, help="sequenze da includere (0 = tutte quelle di training)")
    p.add_argument("--seqs", default="", help="sequenze esplicite, es. 12,57 (solo di training)")
    p.add_argument("--pick", choices=["spread", "random", "densest"], default="spread")
    p.add_argument("--min-multi", type=int, default=2, help="almeno K sequenze con più droni")
    p.add_argument("--seconds", type=float, default=20.0, help="durata del segmento (0 = tutta la sequenza)")
    p.add_argument("--segment", choices=["densest", "start"], default="densest")
    p.add_argument("--max-dur-ms", type=float, default=330.0, help="accumulazione più lunga del modello")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=20, help="finestre per sequenza da confrontare (0 = no)")
    p.add_argument("--dry-run", action="store_true", help="solo selezione e stima di dimensione")
    p.add_argument("--keep-dir", action="store_true", help="tieni anche la cartella non compressa")
    p.add_argument("--val-frac", type=float, default=0.15, help="val_seq_frac del training (default del codice)")
    p.add_argument("--val-seed", type=int, default=42, help="val_seq_seed del training (default del codice)")
    u = sub.add_parser("unpack", help="estrae lo zip e rende assoluti i percorsi dell'indice")
    u.add_argument("zip")
    u.add_argument("--dest", default=".")
    args = ap.parse_args()
    if args.cmd == "pack":
        M.group_sequences = train_only(M.group_sequences, args)
        M.print_plan = print_plan
        M.pack(args)
    else:
        unpack(args)


if __name__ == "__main__":
    main()
