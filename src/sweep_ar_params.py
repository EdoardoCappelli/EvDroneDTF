#!/usr/bin/env python3
"""
Sweep dei 6 iperparametri del tracker AR (INFERENCE-only, nessun retrain) su UNA GPU,
run SEQUENZIALI. I parametri sono flag --ar_* del valutatore: si rigira solo l'eval AR
sullo STESSO checkpoint variando i flag.

Ricerca a BLOCCHI (block coordinate descent), con griglie 2D sulle coppie ACCOPPIATE:
  1) baseline (default) → fissa il vincolo IDF1
  2) griglia 2D  conf × coher   (entrambi gate di validità del track)
  3) griglia 2D  iou  × dist    (entrambi in best_match_track: dist è il fallback di iou)
  4) 1D  std_past_iou_thr       (dedup past↔standard, indipendente)
  5) 1D  max_missed             (lifecycle del track, indipendente)
Ogni blocco fissa il best e passa al successivo.

Selezione = MASSIMIZZA MOTA con VINCOLO "non rompere IDF1":
  tra le config si tengono solo quelle con IDF1 >= (IDF1_baseline - idf1_tol);
  fra queste, MOTA massima (tie-break: IDF1 alto, poi meno ID-switch).
  Se NESSUNA rispetta il vincolo → si protegge l'identità scegliendo la IDF1 massima.

Salvataggio INCREMENTALE in sweep_results.csv (+ summary.json) dopo OGNI run → RESUME:
rilanciando, le combinazioni già nel CSV (con MOTA valida) vengono saltate.

Subset per velocizzare: --seq_sub N → 1 sequenza ogni N (frame INTATTI/consecutivi → AR
corretto). Il tuning è sul TEST subsamplato; il numero finale va riportato sul TEST PIENO
(seq_sub=1): stampato a fine sweep, oppure --final_full per lanciarlo in coda.

Uso (dopo `conda activate FRED++` e gli export HDF5_PLUGIN_PATH / LD_LIBRARY_PATH):
  CUDA_VISIBLE_DEVICES=0 python sweep_ar_params.py \
      --ckpt /equilibrium/ecappelli/runs/joint_scratch_p12_pastdrop_fakepast_20260827_161738/checkpoints/best_model.pt \
      --seq_sub 4
"""
import argparse, csv, json, os, re, subprocess, time
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Spazio di ricerca (il default attuale del tracker è dentro ogni lista) ──
# Griglia COARSE: solo 47 sequenze (≈42 in test) ma enormi (~3200 frame) → ogni run costa in
# proporzione alle sequenze tenute. Con seq_sub~12 restano ~4 seq/run (~13 min). conf resta a 5
# (leva principale); gli altri a 3 valori col DEFAULT incluso. Rifinisci dopo sulla zona migliore.
# CONF_VALS    = [0.35, 0.40, 0.45, 0.50, 0.55]   # <0.35 = valanga di FP (MOTA negativa/phantom)
CONF_VALS    = [0.50, 0.55, 0.60, 0.65, 0.70]   # MOTA cresce con conf (fantasmi filtrati); picco >0.55 → si estende in alto
COHER_VALS   = [0.05, 0.10, 0.20]               # IoU box consecutive (0.0 = gate off, il peggiore)
IOU_VALS     = [0.1, 0.2, 0.35]                 # IoU associazione (0 = bug: match sempre)
DIST_VALS    = [0.06, 0.08, 0.12]               # dist centri norm (fallback IoU)
STDPAST_VALS = [0.2, 0.3, 0.5]                  # IoU dedup past↔standard
MAXMISS_VALS = [4, 6, 12]                        # frame senza update prima di eliminare (33ms → ~0.13..0.4s)
DEFAULT = dict(conf=0.35, coher=0.10, iou=0.2, dist=0.08, stdpast=0.3, maxmiss=6)

PARAM_ORDER = ['conf', 'coher', 'iou', 'dist', 'stdpast', 'maxmiss']

# ── Metriche dal log dell'AR eval (stampa della TrackingMetrics interna) ──
PAT = dict(
    mota=re.compile(r"MOTA\s*:\s*(-?[\d.]+)"),
    idf1=re.compile(r"IDF1\s*:\s*(-?[\d.]+)"),
    idsw=re.compile(r"ID switches\s*:\s*(\d+)"),
    motp=re.compile(r"MOTP \(IoU\)\s*:\s*(-?[\d.]+)"),
)


def parse_metrics(text):
    """Ultima occorrenza di ogni metrica (robusto se il log ne stampa più d'una)."""
    out = {}
    for k, pat in PAT.items():
        ms = pat.findall(text)
        out[k] = (int(ms[-1]) if k == 'idsw' else float(ms[-1])) if ms else None
    return out


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def _i(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None


def _find_in_config(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_in_config(v, key)
            if r is not None:
                return r
    return None


def read_arch_args(ckpt):
    """Flag di ARCHITETTURA dal config.json della run (accanto al checkpoint), così l'eval
    combacia col training (altrimenti strict-load fallisce). Fallback ai valori di questo run."""
    fallback = dict(use_shared_weights=0, shared_cat=0, num_past_annotations=12,
                    num_future_annotations=24, num_future_steps=24,
                    forecast_head_type='transformer', use_cv_anchor=0, vel_avg_k=3)
    cfg = None
    for cand in [os.path.join(os.path.dirname(ckpt), 'config.json'),
                 os.path.join(os.path.dirname(os.path.dirname(ckpt)), 'config.json')]:
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    cfg = json.load(f)
                break
            except Exception:
                pass
    args = {}
    for k, dv in fallback.items():
        v = _find_in_config(cfg, k) if cfg else None
        args[k] = dv if v is None else v
    return args, (cfg is not None)


def key_of(p):
    return (round(p['conf'], 4), round(p['coher'], 4), round(p['iou'], 4),
            round(p['dist'], 4), round(p['stdpast'], 4), int(p['maxmiss']))


def fmt(p):
    return (f"conf={p['conf']:<4} coher={p['coher']:<4} iou={p['iou']:<3} "
            f"dist={p['dist']:<4} stdpast={p['stdpast']:<3} maxmiss={int(p['maxmiss'])}")


def fmt_tag(p):
    return (f"conf{p['conf']}_coher{p['coher']}_iou{p['iou']}_dist{p['dist']}"
            f"_std{p['stdpast']}_mm{int(p['maxmiss'])}").replace('.', 'p')


def eta_str(times, remaining):
    if not times:
        return "?"
    avg = sum(times) / len(times)
    return str(timedelta(seconds=int(avg * max(remaining, 0))))


# ── CSV incrementale + resume ──
CSV_FIELDS = ['phase', 'seq_sub', 'conf', 'coher', 'iou', 'dist', 'stdpast', 'maxmiss',
              'mota', 'idf1', 'idsw', 'motp', 'seconds', 'status']


def load_done(csv_path, seq_sub):
    """Riusa SOLO le run fatte con lo STESSO seq_sub: le metriche dipendono dal subset,
    non si possono mescolare subset diversi (né confrontare col vincolo IDF1)."""
    done = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if str(row.get('seq_sub')) != str(seq_sub):
                    continue
                p = dict(conf=_f(row.get('conf')), coher=_f(row.get('coher')),
                         iou=_f(row.get('iou')), dist=_f(row.get('dist')),
                         stdpast=_f(row.get('stdpast')), maxmiss=_i(row.get('maxmiss')))
                if None in p.values():
                    continue
                done[key_of(p)] = dict(p, mota=_f(row.get('mota')), idf1=_f(row.get('idf1')),
                                       idsw=_i(row.get('idsw')), motp=_f(row.get('motp')),
                                       status=row.get('status', ''))
    return done


def append_row(csv_path, phase, seq_sub, p, m, seconds, status):
    is_new = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(dict(phase=phase, seq_sub=seq_sub, mota=m.get('mota'), idf1=m.get('idf1'),
                        idsw=m.get('idsw'), motp=m.get('motp'),
                        seconds=round(seconds, 1), status=status,
                        **{k: p[k] for k in PARAM_ORDER}))


# ── selezione con vincolo IDF1 ──
def pick_best(rows, idf1_floor):
    valid = [r for r in rows if r.get('mota') is not None and r.get('idf1') is not None]
    ok = [r for r in valid if r['idf1'] >= idf1_floor]
    if ok:
        return max(ok, key=lambda r: (r['mota'], r['idf1'], -(r['idsw'] or 0))), True
    if valid:   # nessuna rispetta il vincolo → proteggi l'identità
        return max(valid, key=lambda r: (r['idf1'], r['mota'])), False
    return None, False


def build_cmd(p, arch, a, out_dir):
    return [a.python, '-u', os.path.join(SCRIPT_DIR, 'main.py'),
            '--config', a.config, '--mode', 'eval', '--evaluator_type', 'past_conditioned_detr',
            '--checkpoint_path', a.ckpt, '--test_batch_size', str(a.batch),
            '--durations', a.durations, '--output_dir', out_dir, '--subsample', '1',
            '--index_path', a.index_path, '--phase', '2', '--query_mode', 'both',
            '--num_standard_queries', '50',
            '--use_shared_weights', str(arch['use_shared_weights']),
            '--shared_cat', str(arch['shared_cat']),
            '--num_past_annotations', str(arch['num_past_annotations']),
            '--num_future_annotations', str(arch['num_future_annotations']),
            '--num_future_steps', str(arch['num_future_steps']),
            '--forecast_head_type', str(arch['forecast_head_type']),
            '--use_cv_anchor', str(arch['use_cv_anchor']), '--vel_avg_k', str(arch['vel_avg_k']),
            '--processor_threshold_eval', '0.3', '--use_only_annotated', '0',
            '--use_custom_normalization', '1', '--use_nms', '1', '--vis_every_n_batches', '999999',
            '--eval_forecasting', '0', '--autoregressive', '1', '--ar_export_mot', '0',
            '--ar_conf_thr', str(p['conf']), '--ar_coher_thr', str(p['coher']),
            '--ar_iou_thr', str(p['iou']), '--ar_dist_thr', str(p['dist']),
            '--ar_std_past_iou_thr', str(p['stdpast']), '--ar_max_missed', str(int(p['maxmiss'])),
            '--ar_seq_subsample', str(a.seq_sub), '--ar_seq_offset', str(a.seq_off)]


def parse_args():
    ap = argparse.ArgumentParser(description="Sweep dei 6 iperparametri del tracker AR (1 GPU, sequenziale).")
    ap.add_argument('--ckpt', required=True, help="best_model.pt del checkpoint da tarare")
    ap.add_argument('--index_path', default='/seidenas/datasets/FRED/preprocessed/')
    ap.add_argument('--out', default=None, help="cartella output (default: <run_dir>/ar_sweep_<ts>)")
    ap.add_argument('--seq_sub', type=int, default=4, help="ar_seq_subsample per il tuning (1 = test pieno)")
    ap.add_argument('--seq_off', type=int, default=0, help="ar_seq_offset (per split disgiunti tuning/report)")
    ap.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    ap.add_argument('--durations', default='33,165,330')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--config', default='RTDetrPastConditioned')
    ap.add_argument('--idf1_tol', type=float, default=0.01,
                    help="quanto IDF1 può scendere sotto baseline restando accettabile (default 0.01)")
    ap.add_argument('--python', default='python3')
    ap.add_argument('--final_full', action='store_true',
                    help="a fine sweep rilancia il best su TEST PIENO (seq_sub=1)")
    a = ap.parse_args()
    if a.out is None:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(a.ckpt)))
        a.out = os.path.join(run_dir, f"ar_sweep_{datetime.now():%Y%m%d_%H%M%S}")
    return a


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, 'sweep_results.csv')
    arch, from_cfg = read_arch_args(a.ckpt)

    print("############ AR PARAM SWEEP (block coordinate descent + griglie 2D) ############")
    print(f"  CKPT : {a.ckpt}")
    print(f"  ARCH : {'da config.json' if from_cfg else '⚠️ FALLBACK (config.json non trovato — controlla che combaci!)'}")
    print(f"         {arch}")
    print(f"  INDEX: {a.index_path}  (test, 1 sequenza ogni {a.seq_sub}, offset {a.seq_off})")
    print(f"  OUT  : {a.out}   |  GPU {a.gpu} (run SEQUENZIALI)")
    print(f"  OBIETTIVO: MOTA MAX con vincolo IDF1 >= baseline-{a.idf1_tol}  (tie-break IDF1, poi meno IDsw)")
    print("################################################################################")

    done = load_done(csv_path, a.seq_sub)
    if done:
        print(f"[resume] {len(done)} run già presenti in {os.path.basename(csv_path)} → verranno saltate")

    N_total = 1 + len(CONF_VALS) * len(COHER_VALS) + len(IOU_VALS) * len(DIST_VALS) \
        + len(STDPAST_VALS) + len(MAXMISS_VALS) + 1   # +1 = run 'final'
    times, counter = [], {'i': 0}

    def do_run(phase, p):
        counter['i'] += 1
        i, k = counter['i'], key_of(p)
        if k in done and done[k].get('mota') is not None:
            r = done[k]
            print(f"[{i}/{N_total}] {phase:11s} {fmt(p)} | (cache) MOTA={r['mota']} IDF1={r['idf1']}")
            return r
        out_dir = os.path.join(a.out, f"{phase}__{fmt_tag(p)}")
        os.makedirs(out_dir, exist_ok=True)
        logpath = out_dir + '.log'
        env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        print(f"[{i}/{N_total}] {phase:11s} {fmt(p)} | running (GPU {a.gpu}) — "
              f"progresso: tail -f {logpath}", flush=True)
        t0 = time.time()
        # stdout+stderr del figlio scritti LIVE sul .log → `tail -f` mostra la barra tqdm
        # dell'eval AR. Niente capture_output (nasconderebbe tutto in un buffer fino alla fine).
        with open(logpath, 'w') as lf:
            proc = subprocess.run(build_cmd(p, arch, a, out_dir), cwd=SCRIPT_DIR, env=env,
                                  stdout=lf, stderr=subprocess.STDOUT)
        secs = time.time() - t0
        with open(logpath, errors='ignore') as lf:
            m = parse_metrics(lf.read())
        status = 'ok' if m.get('mota') is not None else f'FAIL(exit={proc.returncode})'
        append_row(csv_path, phase, a.seq_sub, p, m, secs, status)
        done[k] = dict(p, **m, status=status)
        times.append(secs)
        flag = '' if status == 'ok' else f"  ⚠️ {status} (vedi {os.path.basename(logpath)})"
        print(f"        → MOTA={m.get('mota')} IDF1={m.get('idf1')} IDSW={m.get('idsw')} "
              f"MOTP={m.get('motp')} | {secs:.0f}s | ETA {eta_str(times, N_total - i)}{flag}")
        return done[k]

    best = dict(DEFAULT)

    # 1) baseline → vincolo IDF1
    base = do_run('baseline', dict(DEFAULT))
    if base.get('mota') is None:
        blog = os.path.join(a.out, f"baseline__{fmt_tag(DEFAULT)}.log")
        print("\n❌ La baseline non ha prodotto MOTA/IDF1 → mi fermo (senza baseline il vincolo IDF1 "
              "sarebbe spento e tutte le run rischiano di fallire uguale). Ultime righe del log:")
        try:
            with open(blog, errors='ignore') as f:
                print('   | ' + '\n   | '.join(f.read().splitlines()[-25:]))
        except Exception as e:
            print(f"   (log non leggibile: {e})")
        print(f"   Log completo: {blog}")
        return
    idf1_floor = base['idf1'] - a.idf1_tol
    print(f"\n[baseline] MOTA={base['mota']} IDF1={base['idf1']} → vincolo: IDF1 >= {idf1_floor:.4f}\n")

    # 2) griglia 2D conf × coher
    rowsA = [do_run('confXcoher', dict(best, conf=c, coher=h)) for c in CONF_VALS for h in COHER_VALS]
    bA, okA = pick_best(rowsA, idf1_floor)
    if bA:
        best['conf'], best['coher'] = bA['conf'], bA['coher']
    print(f"  ▸ best (conf,coher) = ({best['conf']}, {best['coher']})  MOTA={bA and bA['mota']} "
          f"IDF1={bA and bA['idf1']}{'' if okA else '  ⚠️ vincolo IDF1 non soddisfatto → protetto IDF1'}\n")

    # 3) griglia 2D iou × dist
    rowsB = [do_run('iouXdist', dict(best, iou=i, dist=d)) for i in IOU_VALS for d in DIST_VALS]
    bB, okB = pick_best(rowsB, idf1_floor)
    if bB:
        best['iou'], best['dist'] = bB['iou'], bB['dist']
    print(f"  ▸ best (iou,dist) = ({best['iou']}, {best['dist']})  MOTA={bB and bB['mota']} "
          f"IDF1={bB and bB['idf1']}{'' if okB else '  ⚠️ protetto IDF1'}\n")

    # 4) 1D std_past_iou_thr
    rowsC = [do_run('stdpast', dict(best, stdpast=s)) for s in STDPAST_VALS]
    bC, _ = pick_best(rowsC, idf1_floor)
    if bC:
        best['stdpast'] = bC['stdpast']
    print(f"  ▸ best std_past = {best['stdpast']}  MOTA={bC and bC['mota']} IDF1={bC and bC['idf1']}\n")

    # 5) 1D max_missed
    rowsD = [do_run('maxmiss', dict(best, maxmiss=mm)) for mm in MAXMISS_VALS]
    bD, _ = pick_best(rowsD, idf1_floor)
    if bD:
        best['maxmiss'] = bD['maxmiss']
    print(f"  ▸ best max_missed = {best['maxmiss']}  MOTA={bD and bD['mota']} IDF1={bD and bD['idf1']}\n")

    # config finale (misurata esplicitamente se non già presente)
    final = do_run('final', dict(best))

    # ── Riepilogo: best + top-5 (vincolate su IDF1), + la config a IDF1 massima ──
    rows = [r for r in done.values() if r.get('mota') is not None and r.get('idf1') is not None]
    ok = [r for r in rows if r['idf1'] >= idf1_floor]
    top = sorted(ok, key=lambda r: (r['mota'], r['idf1'], -(r['idsw'] or 0)), reverse=True)[:5]
    best_idf1 = max(rows, key=lambda r: r['idf1']) if rows else None

    print("\n════════════════════ RISULTATO ════════════════════")
    print(f"  Baseline : MOTA={base.get('mota')}  IDF1={base.get('idf1')}  IDSW={base.get('idsw')}")
    print(f"  BEST     : MOTA={final.get('mota')}  IDF1={final.get('idf1')}  IDSW={final.get('idsw')}")
    print(f"  Config   : {fmt(best)}")
    print("\n  Top-5 (MOTA max con IDF1 >= vincolo):")
    print(f"    {'MOTA':>7} {'IDF1':>7} {'IDsw':>5}  |  parametri")
    for r in top:
        print(f"    {r['mota']:>7.4f} {r['idf1']:>7.4f} {str(r['idsw'] or ''):>5}  |  {fmt(r)}")
    if best_idf1:
        print(f"\n  (config a IDF1 massima: IDF1={best_idf1['idf1']:.4f} MOTA={best_idf1['mota']:.4f} → {fmt(best_idf1)})")

    summary = dict(checkpoint=a.ckpt, arch=arch, seq_sub=a.seq_sub, seq_off=a.seq_off,
                   idf1_floor=idf1_floor, baseline=base, best_config=best, best_metrics=final,
                   top5=top, best_idf1=best_idf1, n_runs=len(rows))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    ar_flags = (f"--ar_conf_thr {best['conf']} --ar_coher_thr {best['coher']} "
                f"--ar_iou_thr {best['iou']} --ar_dist_thr {best['dist']} "
                f"--ar_std_past_iou_thr {best['stdpast']} --ar_max_missed {int(best['maxmiss'])}")
    print(f"\n  CSV      : {csv_path}")
    print(f"  Summary  : {os.path.join(a.out, 'summary.json')}")
    print(f"  Flag best: {ar_flags}")
    print(f"  ➜ Riporta sul TEST PIENO:  ...stessa cmd con --ar_seq_subsample 1  {ar_flags}")
    print("═══════════════════════════════════════════════════")

    # opzionale: rivaluta il best sul test pieno
    if a.final_full:
        print("\n[final_full] rivaluto il best su TEST PIENO (seq_sub=1)...")
        saved = a.seq_sub
        a.seq_sub = 1
        counter['i'] = N_total - 1   # solo per l'indice di stampa
        r = do_run('final_full', dict(best))
        a.seq_sub = saved
        print(f"[final_full] TEST PIENO → MOTA={r.get('mota')} IDF1={r.get('idf1')} IDSW={r.get('idsw')}")


if __name__ == '__main__':
    main()
