"""
Estrae detection / tracking / forecasting dai test log.
"""
import os
import re
import json
from pathlib import Path

# numeri anche negativi o in notazione scientifica
_NUM = r"(-?[\d.]+(?:[eE][+-]?\d+)?)"

# detection
DET_PATTERNS = {
    "map":       rf"(?m)^\s*\{{?\s*'?map'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "map_50":    rf"(?m)^\s*\{{?\s*'?map_50'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "map_75":    rf"(?m)^\s*\{{?\s*'?map_75'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "mar_100":   rf"(?m)^\s*\{{?\s*'?mar_100'?\s*[:=]\s*(?:tensor\()?\s*{_NUM}",
    "precision": rf"(?m)^\s*Precision\s*:\s*{_NUM}",
    "recall":    rf"(?m)^\s*Recall\s*:\s*{_NUM}",
    "f1":        rf"(?m)^\s*F1\s*:\s*{_NUM}",
}
# forecasting
FORECAST_PATTERNS = {
    "ADE_full": rf"ADE full\s*\(px\)\s*:\s*{_NUM}",
    "FDE_full": rf"FDE full\s*\(px\)\s*:\s*{_NUM}",
    "ADE_0.4s": rf"ADE short_0_4s\s*\(px\)\s*:\s*{_NUM}",
    "FDE_0.4s": rf"FDE short_0_4s\s*\(px\)\s*:\s*{_NUM}",
    "ADE_0.8s": rf"ADE mid_0_8s\s*\(px\)\s*:\s*{_NUM}",
    "FDE_0.8s": rf"FDE mid_0_8s\s*\(px\)\s*:\s*{_NUM}",
    "mean_iou": rf"mean future IoU\s*:\s*{_NUM}",
}
# tracking (sezione autoregressiva)
TRACK_PATTERNS = {
    "MOTA":        rf"MOTA\s*:\s*{_NUM}",
    "IDF1":        rf"IDF1\s*:\s*{_NUM}",
    "ID_switches": r"ID switches\s*:\s*(\d+)",
    "MOTP":        rf"MOTP \(IoU\)\s*:\s*{_NUM}",
}
# AR: mAP full-PR (tutti gli output), confrontabile con Both/Standard
FULLPR_PATTERNS = {
    "map_fullpr":   rf"'map_fullpr'\s*:\s*{_NUM}",
    "map50_fullpr": rf"'map50_fullpr'\s*:\s*{_NUM}",
    "map75_fullpr": rf"'map75_fullpr'\s*:\s*{_NUM}",
}
# intestazione del full-PR: dice se è calcolata con NMS
FULLPR_HEADER_PAT = r"mAP AR FULL-PR([^\n]*)"

# epoca del checkpoint testato
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


def extract_fullpr(text, before):
    """mAP full-PR di una sezione AR e fullpr_nms ('1' con NMS, '0' senza, 'N/A' se manca)."""
    out = {k: v if v == "N/A" else f"{float(v):.4f}" for k, v in extract(text, FULLPR_PATTERNS).items()}
    heads = re.findall(FULLPR_HEADER_PAT, before)
    out["fullpr_nms"] = ("1" if "NMS" in heads[-1] else "0") if heads else "N/A"
    return out


def split_sections(content):
    """Divide un test log nei suoi pass: [(label, testo, inizio)]."""
    anchors = []
    # detection: riga "Query mode : ..."
    for m in re.finditer(r"Query mode\s*:\s*(both|standard_only|past_only)", content):
        anchors.append((m.start(), f"detection_{m.group(1)}"))
    # forecasting: header  "  FORECASTING"
    for m in re.finditer(r"(?m)^\s*FORECASTING\s*$", content):
        anchors.append((m.start(), "forecasting"))
    # autoregressive: "MODE : AUTOREGRESSIVE (...)", la variante distingue AR, oracle e tracktrack
    for m in re.finditer(r"MODE\s*:\s*AUTOREGRESSIVE(?:\s*\(([^)]*)\))?", content):
        variant = (m.group(1) or "").lower()
        anchors.append((m.start(), "autoregressive_oracle" if "oracle" in variant else
                        "autoregressive_tracktrack" if "tracktrack" in variant else "autoregressive"))
    anchors.sort(key=lambda a: a[0])
    sections = []
    for i, (pos, label) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(content)
        sections.append((label, content[pos:end], pos))
    return sections


def parse_test_log(path):
    """Ritorna {label -> metriche} per un test.log (che può contenere tutti i pass)."""
    evals = {}
    content = path.read_text(errors="ignore")
    prev = 0
    for label, text, start in split_sections(content):
        before, prev = content[prev:start], start
        if label.startswith("detection"):
            metrics = extract(text, DET_PATTERNS)
        elif label == "forecasting":
            metrics = extract(text, FORECAST_PATTERNS)
            metrics.update(extract_per_step_ade(text))
        elif label.startswith("autoregressive"):
            metrics = extract(text, DET_PATTERNS)
            metrics.update(extract_fullpr(text, before))
            metrics.update(extract(text, TRACK_PATTERNS))
        else:
            continue
        if label in evals:                      # stessa label due volte
            label = f"{label}__{path.stem}"
        evals[label] = metrics
    return evals


def run_after(run_name, since):
    """True se il timestamp in coda al nome della run (YYYYMMDD_HHMMSS) è >= since."""
    if not since:
        return True
    parts = run_name.split('_')
    d, t = (parts[-2], parts[-1]) if len(parts) >= 2 else ('', '')
    if not (len(d) == 8 and d.isdigit() and len(t) == 6 and t.isdigit()):
        return False                              # nessun timestamp valido
    since_digits = re.sub(r"\D", "", since)        # tengo solo le cifre di SINCE
    return (d + t)[:len(since_digits)] >= since_digits


def parse_durations(train_log):
    if not train_log:
        return "N/A"
    m = re.search(r"(?i)durations[\s=:]+\[?\s*([\d,\s]+?)\s*\]?\s*(?:ms|$)",
                  train_log.read_text(errors="ignore"))
    return m.group(1).strip() if m else "N/A"


def parse_checkpoint_epoch(test_logs):
    """Epoca del checkpoint testato (prima riga 'Checkpoint from epoch')."""
    for tlog in test_logs:
        m = re.search(CKPT_EPOCH_PAT, tlog.read_text(errors="ignore"))
        if m:
            return m.group(1)
    return "N/A"


# flag della run dal config.json (prima quello di training, in checkpoints/)
def find_config_json(run_path):
    """config.json della run, dando priorità a quello di training (checkpoints/)."""
    prefer = run_path / "checkpoints" / "config.json"
    if prefer.is_file():
        return prefer
    ckpt_hits = [p for p in run_path.rglob("config.json") if p.parent.name == "checkpoints"]
    if ckpt_hits:
        return ckpt_hits[0]
    top = run_path / "config.json"
    if top.is_file():
        return top
    return next(iter(run_path.rglob("config.json")), None)


def _find_in_config(obj, key):
    """Cerca `key` a qualunque livello del config (top-level ha priorità sui *_args)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_in_config(v, key)
            if r is not None:
                return r
    return None


# flag stampato -> chiave del config.json
CONFIG_FLAGS = {
    "query_mixed": "use_mixed_query_mode",
    "shared":      "use_shared_weights",
    "cv_anchor":   "use_cv_anchor",
    "fakepast":    "use_fake_past",
    "pastdrop":    "use_past_dropout",
    "pastnoise":   "use_past_noise",
    "vanish":      "use_vanish_past",
    "selfpast":    "use_self_past",
}
# schedule del curriculum closed-loop, conta solo con selfpast=1
SCHEDULE_KEY = "self_past_schedule"


def parse_config_flags(run_path):
    """Flag di CONFIG_FLAGS ('1' / '0' / 'N/A') e schedule ('-' se selfpast non è attivo)."""
    na = {k: "N/A" for k in (*CONFIG_FLAGS, "schedule")}   # run senza config.json leggibile
    cfg_path = find_config_json(run_path)
    if cfg_path is None:
        return na
    try:
        cfg = json.loads(cfg_path.read_text(errors="ignore"))
    except Exception:
        return na

    def _fmt(key):
        v = _find_in_config(cfg, key)
        if v is None:
            return "N/A"
        truthy = (v is True) or (isinstance(v, (int, float)) and v != 0) \
                 or str(v).strip().lower() in ('1', 'true', 'yes', 'on')
        return "1" if truthy else "0"

    out = {k: _fmt(key) for k, key in CONFIG_FLAGS.items()}
    sched = _find_in_config(cfg, SCHEDULE_KEY)
    out["schedule"] = ("-" if out["selfpast"] != "1" else
                       str(sched).strip().lower() if sched is not None else "N/A")
    return out


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
    # config attiva
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

        cfg_flags = parse_config_flags(run_path)
        all_runs.append({
            "nome_run":    run_path.name,
            "durations":   parse_durations(next(run_path.glob("train_*.log"), None)),
            "checkpoint_epoch": parse_checkpoint_epoch(test_logs),
            **cfg_flags,
            "evaluations": evaluations,
        })

    # stampa
    for run in all_runs:
        print("=" * 72)
        print(f"RUN: {run['nome_run']}   (durations: {run['durations']}  |  ckpt epoch: {run['checkpoint_epoch']}"
              f"  |  query_mixed: {run['query_mixed']}  |  shared: {run['shared']}  |  cv_anchor: {run['cv_anchor']}) | "
              f"fakepast: {run['fakepast']}  |  pastdrop: {run['pastdrop']}  |  pastnoise: {run['pastnoise']}"
              f"  |  vanish: {run['vanish']}  |  selfpast: {run['selfpast']}  |  schedule: {run['schedule']}")
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
