#!/usr/bin/env python3
"""
Ri-affetta il preprocessed FRED su uno split diverso (es. challenging) SENZA ri-preprocessare
gli HDF5.

Il preprocessed canonical (`train_windows_33ms.json` + `test_windows_33ms.json`) copre TUTTE le
sequenze del dataset. Uno "split" (canonical / challenging) è solo una diversa PARTIZIONE delle
stesse sequenze in train/test, definita da due file `.txt` che elencano le cartelle-sequenza
(es. `9/`, `8/`, `230/`). Qui si ri-partizionano le stesse finestre secondo quei `.txt`.

La cartella-sequenza di ogni finestra si ricava dal suo path:
    hdf5_path = .../{folder}/Event/events.hdf5   →   folder = basename(dirname(dirname(hdf5_path)))
Il path resta ASSOLUTO e invariato (il file HDF5 sta dov'è fisicamente, a prescindere dallo split
label): cambia solo in quale JSON finisce la finestra.

Output: `train_windows_<acc>ms.json` e `test_windows_<acc>ms.json` con i NOMI CANONICI dentro
`--out`, così si usa direttamente con `INDEX_PATH=--out` (i prefissi `train_`/`test_` che il
dataset_factory si aspetta combaciano).

Uso (sul cluster, dopo aver copiato i due .txt del challenging):
  python make_split_index.py \
      --preprocessed /seidenas/datasets/FRED/preprocessed \
      --train_split  challenging_train_split.txt \
      --test_split   challenging_test_split.txt \
      --out /seidenas/datasets/FRED/preprocessed_challenging
"""
import argparse
import copy
import json
import os


def load_folders(txt_path):
    """Insieme delle cartelle-sequenza da un .txt (una per riga, es. '9/' → '9')."""
    folders = set()
    with open(txt_path) as f:
        for line in f:
            s = line.strip().strip('/').strip()
            if s:
                folders.add(s)
    return folders


def folder_of(window):
    """Cartella-sequenza di una finestra: .../{folder}/Event/events.hdf5 → {folder}."""
    return os.path.basename(os.path.dirname(os.path.dirname(window['hdf5_path'])))


def main():
    ap = argparse.ArgumentParser(description="Ri-affetta il preprocessed FRED su un altro split.")
    ap.add_argument('--preprocessed', required=True,
                    help="dir col preprocessed canonical (train_/test_windows_<acc>ms.json)")
    ap.add_argument('--train_split', required=True, help=".txt con le cartelle dello split TRAIN")
    ap.add_argument('--test_split', required=True, help=".txt con le cartelle dello split TEST")
    ap.add_argument('--out', required=True, help="dir di output (nomi canonici train_/test_)")
    ap.add_argument('--acc_ms', type=int, default=33, help="accumulazione (default 33)")
    a = ap.parse_args()

    # 1) Unione di TUTTE le finestre (train + test canonical = tutte le sequenze) ──────────
    pool, template = [], None
    for name in ('train', 'test'):
        p = os.path.join(a.preprocessed, f'{name}_windows_{a.acc_ms}ms.json')
        with open(p) as f:
            d = json.load(f)
        if template is None:
            template = {k: v for k, v in d.items() if k != 'windows'}   # metadati (accumulation_time_ms, ...)
        pool.extend(d['windows'])

    by_folder = {}
    for w in pool:
        by_folder.setdefault(folder_of(w), []).append(w)
    print(f"[pool] {len(pool)} finestre | {len(by_folder)} sequenze uniche")
    _sample = pool[0]['hdf5_path'] if pool else '(vuoto)'
    print(f"[check] esempio hdf5_path: {_sample}  ->  folder={folder_of(pool[0]) if pool else '?'}")

    # 2) Ri-partiziona secondo i .txt e scrivi coi NOMI CANONICI ──────────────────────────
    os.makedirs(a.out, exist_ok=True)
    for split_name, txt in (('train', a.train_split), ('test', a.test_split)):
        want = load_folders(txt)
        got = want & set(by_folder)
        missing = want - set(by_folder)                 # cartelle richieste ma senza finestre
        wins = [w for fld in sorted(got) for w in by_folder[fld]]

        out_obj = copy.deepcopy(template)
        out_obj['split'] = split_name
        out_obj['windows'] = wins
        outp = os.path.join(a.out, f'{split_name}_windows_{a.acc_ms}ms.json')
        with open(outp, 'w') as f:
            json.dump(out_obj, f)

        warn = f"  [!] MANCANTI {sorted(missing)}" if missing else ""
        print(f"[{split_name}] richieste {len(want)} seq | trovate {len(got)} | "
              f"{len(wins)} finestre -> {outp}{warn}")

    print("\nFatto. Ora:")
    print(f"  INDEX_PATH={a.out}/ SPLIT_TAG=challenging ./run_joint_p12_scratch_pastdrop_fakepast.sh")


if __name__ == '__main__':
    main()
