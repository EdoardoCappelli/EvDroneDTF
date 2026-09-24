#!/bin/bash
# Visualizza la past-box augmentation (DRIFT + JUMP) su un subsample del TRAIN set → PNG.
# Le box "sfasate" dal jump sono evidenziate in ROSSO. Non allena, non carica il modello.
#
#   ./run_viz_past_aug.sh                                  # metavision, jump default
#   REPR=time_surface DURATION=330 NORM=0 ./run_viz_past_aug.sh
#   JUMPP=0.3 JUMPSTD=0.4 N=12 ./run_viz_past_aug.sh       # jump più aggressivo
#   JUMPP=0 ./run_viz_past_aug.sh                          # solo drift (niente jump)
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

INDEX_PATH=${INDEX_PATH:-/seidenas/datasets/FRED/preprocessed/}
DURATIONS=${DURATION:-33,165,330}
NUM_PAST=${NUM_PAST:-12}
RENDER_MODE=${REPR:-metavision}   # DEVE combaciare col training da visualizzare
NORM=${NORM:-0}                   # 0/1 come il training
SUBSAMPLE=${SUBSAMPLE:-50}
N_SAMPLES=${N:-8}
AUG_P=${AUGP:-1.0}                # 1.0 = perturba ogni track (per vedere sempre l'effetto)
AUG_STD=${PASTAUGSTD:-0.05}       # drift
AUG_MAX=${AUGMAX:-0.2}
JUMP_P=${JUMPP:-0.2}              # prob per-box di outlier "sfasato"
JUMP_STD=${JUMPSTD:-0.3}          # magnitudine del salto
JUMP_MAX=${JUMPMAX:-0.5}
OUT=${OUT:-./past_aug_jump_viz}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[viz] render=${RENDER_MODE} norm=${NORM} drift_std=${AUG_STD} jump_p=${JUMP_P} jump_std=${JUMP_STD}"
python3 -u visualize_past_aug.py \
    --index_path "${INDEX_PATH}" --durations "${DURATIONS}" --num_past "${NUM_PAST}" \
    --render_mode "${RENDER_MODE}" --use_custom_normalization "${NORM}" \
    --subsample "${SUBSAMPLE}" --n_samples "${N_SAMPLES}" \
    --aug_p "${AUG_P}" --aug_std "${AUG_STD}" --aug_max "${AUG_MAX}" \
    --jump_p "${JUMP_P}" --jump_std "${JUMP_STD}" --jump_max "${JUMP_MAX}" \
    --output_dir "${OUT}"

echo "[viz] PNG in: ${OUT}   (scaricali con scp)"
