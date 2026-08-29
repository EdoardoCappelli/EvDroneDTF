#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  SWEEP batch_size × num_workers  —  micro-benchmark di throughput
#  Usa la pipeline REALE (dataset + forward + backward), NON input finti.
#  Per ogni combo misura N batch e stampa: samples/s, data-wait vs compute, mem GPU.
#  NON allena per intero e NON fa eval: esce dopo il benchmark (SystemExit 0).
# ─────────────────────────────────────────────────────────────────────────────
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

# NIENTE `set -e`: un OOM su una combo NON deve fermare lo sweep.
set -uo pipefail

# ── Griglia da testare (modificala come vuoi) ───────────────────────────────
BS_LIST=(8 16 24 32)
NW_LIST=(8 12 16 24)

BENCH_BATCHES=80        # batch misurati per combo (dopo il warmup)
BENCH_WARMUP=15         # batch iniziali scartati (spin-up worker + JIT numba)

# ── Config fissa (uguale al training shared-weights che vuoi ottimizzare) ────
CONFIG="RTDetrPastConditioned"
DURATION=33,165,330
TRAIN_SUBSAMPLE=10
PHASE=2
QUERY_MODE='both'
NUM_STD_QUERIES=50
USE_SHARED_WEIGHTS=1
SHARED_CAT=1
FREEZE_DETECTOR=0
USE_CUSTOM_NORMALIZATION=1
SEED=42
INDEX_PATH="/seidenas/datasets/FRED/preprocessed/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output minimale (solo config.json; nessun checkpoint perché usciamo prima)
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_DIR="${SCRIPT_DIR}/runs/_bench_sweep_${STAMP}"
mkdir -p "${BENCH_DIR}"
SUMMARY="${BENCH_DIR}/summary.txt"

printf "%-6s %-6s %-8s %-9s %-10s %-11s %-13s %-12s\n" \
    "bs" "nw" "sec/b" "samp/s" "data_ms" "comp_ms" "gpu_alloc_MB" "status" | tee "${SUMMARY}"
echo "-------------------------------------------------------------------------------------" | tee -a "${SUMMARY}"

for BS in "${BS_LIST[@]}"; do
  for NW in "${NW_LIST[@]}"; do
    RUN_LOG="${BENCH_DIR}/bench_bs${BS}_nw${NW}.log"
    echo ">>> Benchmark bs=${BS} nw=${NW} ..."

    python3 main.py \
        --mode train \
        --config "${CONFIG}" \
        --index_path "${INDEX_PATH}" \
        --durations "${DURATION}" \
        --subsample "${TRAIN_SUBSAMPLE}" \
        --epochs 1 \
        --train_batch_size "${BS}" \
        --num_workers "${NW}" \
        --output_dir "${BENCH_DIR}/bs${BS}_nw${NW}" \
        --phase "${PHASE}" \
        --query_mode "${QUERY_MODE}" \
        --num_standard_queries "${NUM_STD_QUERIES}" \
        --use_shared_weights "${USE_SHARED_WEIGHTS}" \
        --shared_cat "${SHARED_CAT}" \
        --freeze_detector "${FREEZE_DETECTOR}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" \
        --seed "${SEED}" \
        --use_wandb 0 \
        --bench_batches "${BENCH_BATCHES}" \
        --bench_warmup "${BENCH_WARMUP}" \
        > "${RUN_LOG}" 2>&1
    RC=$?

    LINE=$(grep -a -m1 '\[BENCH\]' "${RUN_LOG}" | tr -d '\r' || true)
    if [ -n "${LINE}" ]; then
        SECB=$(echo "${LINE}"  | grep -o 'sec/batch=[0-9.]*'  | cut -d= -f2)
        SPS=$(echo "${LINE}"   | grep -o 'samples_s=[0-9.]*'  | cut -d= -f2)
        DMS=$(echo "${LINE}"   | grep -o 'data_ms=[0-9.]*'    | cut -d= -f2)
        CMS=$(echo "${LINE}"   | grep -o 'compute_ms=[0-9.]*' | cut -d= -f2)
        GAL=$(echo "${LINE}"   | grep -o 'gpu_alloc_MB=[0-9.]*' | cut -d= -f2)
        printf "%-6s %-6s %-8s %-9s %-10s %-11s %-13s %-12s\n" \
            "${BS}" "${NW}" "${SECB}" "${SPS}" "${DMS}" "${CMS}" "${GAL}" "OK" | tee -a "${SUMMARY}"
    elif grep -qi 'out of memory' "${RUN_LOG}"; then
        printf "%-6s %-6s %-8s %-9s %-10s %-11s %-13s %-12s\n" \
            "${BS}" "${NW}" "-" "-" "-" "-" "-" "OOM" | tee -a "${SUMMARY}"
    else
        printf "%-6s %-6s %-8s %-9s %-10s %-11s %-13s %-12s\n" \
            "${BS}" "${NW}" "-" "-" "-" "-" "-" "ERR(rc=${RC})" | tee -a "${SUMMARY}"
    fi
  done
done

echo ""
echo "============================================================"
echo "  Sweep finito. Riepilogo:  ${SUMMARY}"
echo "  Log per-combo:            ${BENCH_DIR}/bench_bs*_nw*.log"
echo "============================================================"
cat "${SUMMARY}"