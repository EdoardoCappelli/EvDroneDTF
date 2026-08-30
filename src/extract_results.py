"""
Estrae detection / tracking / forecasting dai test log.
"""
import os
import re
import json
from pathlib import Path

# valori numerici robusti: notazione scientifica (1.97e-05) e segno negativo (MOTA)
_NUM = r"(-?[\d.]+(?:[eE][+-]?\d+)?)"

# ── Detection (dal pprint dict + dal riepilogo di evaluate()) ──
DET_PATTERNS = {
    "map":       rf"(?m)^\s*\{{?\s*'?map'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "map_50":    rf"(?m)^\s*\{{?\s*'?map_50'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "map_75":    rf"(?m)^\s*\{{?\s*'?map_75'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "mar_100":   rf"(?m)^\s*\{{?\s*'?mar_100'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "precision": rf"(?m)^\s*Precision\s*:\s*{_NUM}",
    "recall":    rf"(?m)^\s*Recall\s*:\s*{_NUM}",
    "f1":        rf"(?m)^\s*F1\s*:\s*{_NUM}",
}
# ── Forecasting (formato reale: "ADE full      (px)  :"; orizzonti short_0_4s / mid_0_8s) ──
FORECAST_PATTERNS = {
    "ADE_full": rf"ADE full\s*\(px\)\s*:\s*{_NUM}",
    "FDE_full": rf"FDE full\s*\(px\)\s*:\s*{_NUM}",
    "ADE_0.4s": rf"ADE short_0_4s\s*\(px\)\s*:\s*{_NUM}",
    "FDE_0.4s": rf"FDE short_0_4s\s*\(px\)\s*:\s*{_NUM}",
    "ADE_0.8s": rf"ADE mid_0_8s\s*\(px\)\s*:\s*{_NUM}",
    "FDE_0.8s": rf"FDE mid_0_8s\s*\(px\)\s*:\s*{_NUM}",
    "mean_iou": rf"mean future IoU\s*:\s*{_NUM}",
}
# ── Tracking (dal blocco AUTOREGRESSIVE) ──
TRACK_PATTERNS = {
    "MOTA":        rf"MOTA\s*:\s*{_NUM}",
    "IDF1":        rf"IDF1\s*:\s*{_NUM}",
    "ID_switches": r"ID switches\s*:\s*(\d+)",
    "MOTP":        rf"MOTP \(IoU\)\s*:\s*{_NUM}",
}

# ── Epoca del checkpoint testato: riga "Checkpoint from epoch: N" (o "Checkpoints ...")
#    stampata dall'evaluator al load. Robusta a singolare/plurale e maiuscole. ──
CKPT_EPOCH_PAT = r"(?i)Checkpoints?\s+from\s+epoch\s*:\s*(\d+)"


def extract(text, patterns):
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = m.group(1).strip() if m else "N/A"
    return out


def extract_per_step_ade(text):
    """ADE per-step dalla riga 'per-step ADE px : t1=.. t2=..'."""
    out = {}
    m = re.search(r"per-step ADE px\s*:\s*(.*)", text)
    if m:
        for t_str, v in re.findall(r"t(\d+)\s*=\s*([\d.]+)", m.group(1)):
            out[f"ADE_t{int(t_str)}"] = v
    return out


def split_sections(content):
    """Un solo test.log contiene più pass concatenati → li separo in sezioni,
    ancorando ai marker STAMPATI DALL'EVALUATOR (non ai nomi file)."""
    anchors = []
    # detection: riepilogo di evaluate()  →  "  Query mode                 : both"
    for m in re.finditer(r"Query mode\s*:\s*(both|standard_only|past_only)", content):
        anchors.append((m.start(), f"detection_{m.group(1)}"))
    # forecasting: header  "  FORECASTING"
    for m in re.finditer(r"(?m)^\s*FORECASTING\s*$", content):
        anchors.append((m.start(), "forecasting"))
    # autoregressive:  "  MODE : AUTOREGRESSIVE (closed-loop)"
    for m in re.finditer(r"MODE\s*:\s*AUTOREGRESSIVE", content):
        anchors.append((m.start(), "autoregressive"))
    anchors.sort(key=lambda a: a[0])
    sections = []
    for i, (pos, label) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(content)
        sections.append((label, content[pos:end]))
    return sections


def parse_test_log(path):
    """Ritorna {label -> metriche} per un test.log (che può contenere tutti i pass)."""
    evals = {}
    content = path.read_text(errors="ignore")
    for label, text in split_sections(content):
        if label.startswith("detection"):
            metrics = extract(text, DET_PATTERNS)
        elif label == "forecasting":
            metrics = extract(text, FORECAST_PATTERNS)
            metrics.update(extract_per_step_ade(text))
        elif label == "autoregressive":
            # AR: mAP + prec/rec/f1 closed-loop  +  tracking
            metrics = extract(text, DET_PATTERNS)
            metrics.update(extract(text, TRACK_PATTERNS))
        else:
            continue
        if label in evals:                      # stessa label due volte → disambigua
            label = f"{label}__{path.stem}"
        evals[label] = metrics
    return evals


def run_after(run_name, since):
    """True se il timestamp nel nome run (..._YYYYMMDD_HHMMSS) è >= since (inclusivo).
    Confronto su cifre: YYYYMMDD[HHMMSS] ordina lessicograficamente = cronologicamente."""
    if not since:
        return True
    matches = re.findall(r"(\d{8})_(\d{6})", run_name)   # tutti i YYYYMMDD_HHMMSS
    if not matches:
        return False                              # SINCE attivo ma niente data → escludo
    d, t = matches[-1]                            # l'ultimo = il timestamp finale della run
    run_digits = d + t                            # YYYYMMDDHHMMSS
    since_digits = re.sub(r"\D", "", since)        # tengo solo le cifre di SINCE
    return run_digits[:len(since_digits)] >= since_digits


def parse_durations(train_log):
    if not train_log:
        return "N/A"
    m = re.search(r"(?i)durations[\s=:]+\[?\s*([\d,\s]+?)\s*\]?\s*(?:ms|$)",
                  train_log.read_text(errors="ignore"))
    return m.group(1).strip() if m else "N/A"


def parse_checkpoint_epoch(test_logs):
    """Epoca del checkpoint usato in test — riga 'Checkpoint from epoch: N' stampata
    dall'evaluator al load. Tutti i pass usano lo STESSO best_model.pt → prima occorrenza."""
    for tlog in test_logs:
        m = re.search(CKPT_EPOCH_PAT, tlog.read_text(errors="ignore"))
        if m:
            return m.group(1)
    return "N/A"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Estrae detection/tracking/forecasting dai test log.")
    ap.add_argument("--runs-dir",   default=os.environ.get("RUNS_DIR", "/equilibrium/ecappelli/runs"),
                    help="cartella delle run (o env RUNS_DIR)")
    ap.add_argument("--since",      default=os.environ.get("SINCE", ""),
                    help="tiene solo run col timestamp nel nome >= SINCE, inclusivo "
                         "(es. 20260802). O env SINCE.")
    ap.add_argument("--run-filter", default=os.environ.get("RUN_FILTER", ""),
                    help="tiene solo run che contengono questa sottostringa (o env RUN_FILTER)")
    args = ap.parse_args()

    base_dir   = Path(args.runs_dir)
    since      = args.since.strip()
    run_filter = args.run_filter.strip()
    # Config attiva SEMPRE stampata → si vede subito se il filtro è impostato o no.
    print(f"[config] runs_dir={base_dir} | since={since or '(nessuno)'} | "
          f"run_filter={run_filter or '(nessuno)'}\n")

    if not base_dir.exists():
        print(f"Errore: {base_dir} non esiste. Usa --runs-dir o RUNS_DIR.")
        return

    all_runs, n_skip_date = [], 0
    for run_path in sorted(base_dir.iterdir()):
        if not run_path.is_dir():
            continue
        if run_filter and run_filter.lower() not in run_path.name.lower():
            continue
        if not run_after(run_path.name, since):
            n_skip_date += 1
            continue

        test_logs = sorted(run_path.glob("test_*.log"))
        if not test_logs:
            continue

        evaluations = {}
        for tlog in test_logs:
            evaluations.update(parse_test_log(tlog))
        if not evaluations:
            continue

        all_runs.append({
            "nome_run":    run_path.name,
            "durations":   parse_durations(next(run_path.glob("train_*.log"), None)),
            "checkpoint_epoch": parse_checkpoint_epoch(test_logs),
            "evaluations": evaluations,
        })

    # ── Stampa leggibile ──
    for run in all_runs:
        print("=" * 72)
        print(f"RUN: {run['nome_run']}   (durations: {run['durations']}  |  ckpt epoch: {run['checkpoint_epoch']})")
        for label in sorted(run["evaluations"]):
            ev = run["evaluations"][label]
            base = {k: v for k, v in ev.items() if not k.startswith("ADE_t")}
            print(f"  [{label}]  " + "  ".join(f"{k}={v}" for k, v in base.items()))
            steps = sorted((k for k in ev if k.startswith("ADE_t")), key=lambda k: int(k[5:]))
            if steps:
                print("      per-step ADE: " + "  ".join(f"{k}={ev[k]}" for k in steps))

    with open("summary_results.json", "w") as f:
        json.dump(all_runs, f, indent=4)
    print(f"\n-> {len(all_runs)} run incluse"
          + (f", {n_skip_date} escluse dal filtro data (--since {since})" if since else "")
          + "  |  summary_results.json")


if __name__ == "__main__":
    main()
