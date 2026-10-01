#!/usr/bin/env bash
# Fast, repeatable evaluation on the fixed FRED canonical mini-test.
#
# Usage:
#   ./run_quick_test.sh /path/to/checkpoint.pt [output_dir]
#
# Optional environment overrides:
#   FRED_MINI_INDEX_PATH, PYTHON_BIN, CUDA_VISIBLE_DEVICES

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 CHECKPOINT [OUTPUT_DIR]" >&2
    exit 2
fi

CHECKPOINT=$(realpath "$1")
OUTPUT_DIR=${2:-"quick_test_$(basename "${CHECKPOINT%.pt}")"}
MINI_INDEX=${FRED_MINI_INDEX_PATH:-/media/becattini/SSD4TB/datasets/FRED/preprocessed_ar_mini}
# The dataset loader appends the index filename directly to index_path.
MINI_INDEX="${MINI_INDEX%/}/"
PYTHON_BIN=${PYTHON_BIN:-python3}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "${MINI_INDEX}test_windows_33ms.json" ]]; then
    echo "FRED canonical mini-test index not found: ${MINI_INDEX}test_windows_33ms.json" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

COMMON_ARGS=(
    --config RTDetrPastConditioned
    --mode eval
    --evaluator_type past_conditioned_detr
    --checkpoint_path "$CHECKPOINT"
    --index_path "$MINI_INDEX"
    --output_dir "$OUTPUT_DIR"
    --durations 33,165,330
    --subsample 1
    --num_workers 0
    --test_batch_size 1
    --phase 2
    --num_standard_queries 50
    --num_past_annotations 12
    --num_future_annotations 24
    --num_future_steps 24
    --forecast_head_type transformer
    --use_shared_weights 0
    --shared_cat 0
    --use_cv_anchor 0
    --vel_avg_k 3
    --processor_threshold_eval 0.3
    --use_only_annotated 0
    --use_custom_normalization 1
    --use_nms 1
    --vis_every_n_batches 1000000
)

run_pass() {
    local name=$1
    shift
    echo
    echo "=== Quick test: $name ==="
    "$PYTHON_BIN" -u main.py "${COMMON_ARGS[@]}" "$@" \
        2>&1 | tee "$OUTPUT_DIR/$name.log"
}

cd "$SCRIPT_DIR"

# Cold-start detector: mAP/mAP50 without past trajectories.
run_pass detection \
    --query_mode standard_only \
    --eval_forecasting 0 \
    --autoregressive 0

# Oracle-past forecasting: ADE/FDE and future IoU.
run_pass forecasting \
    --query_mode both \
    --eval_forecasting 1 \
    --autoregressive 0

# Closed-loop tracking: predicted tracks become the next frame's past input.
run_pass autoregressive \
    --query_mode both \
    --eval_forecasting 0 \
    --autoregressive 1 \
    --ar_oracle_past 0 \
    --ar_export_mot 0 \
    --ar_std_box_priority 1

echo
echo "Quick test complete. Logs: $OUTPUT_DIR"
