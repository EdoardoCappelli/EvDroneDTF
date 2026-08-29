#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

# ═════════════════════════════════════════════════════════════════════════════
#  DETECTOR BASE P=12 — con ENTRAMBE le augmentation anti-deriva: past_dropout + fake_past.
#  (Config identica a run_train_detector_p12_mixed_pastdrop.sh — che pure le aveva già
#   entrambe on — ma con nome/RUN_NAME espliciti + contatore FP-by-source sull'eval AR.)
#
#  È il rimedio LATO-TRAINING all'interferenza (recall) e ai fantasmi (precision), cioè
#  l'alternativa "morbida" alla maschera a blocchi / 2-pass. Le due leve:
#    - USE_PAST_DROPOUT=1 → droni con passato droppati a random (finiscono in new_labels):
#      il modello impara a NON dipendere dal passato → le standard ritrovano il drone anche
#      quando il passato lagga (attacca l'INTERFERENZA / recall).
#    - USE_FAKE_PAST=1    → passati FINTI che puntano a non-droni → classificati BACKGROUND:
#      il modello dà score BASSO quando il passato punta al vuoto → il track fantasma prende
#      score basso → viene mancato → potato (attacca i FANTASMI / precision).
#  (Entrambe girano SOLO nei batch 'both'; con mixed p_both=0.4 → ~40% dei batch.)
#
#  🔴 use_shared_weights=0 (RAMI SEPARATI): riusabile come PRETRAINED_DETECTOR_PATH dalle run
#     di forecasting.  🔵 P=12 (400ms).
#
#  ⚠️ SUB-CAMPIONAMENTO: a TRAIN_SUBSAMPLE=10 il pastdrop precedente SOTTO-ALLENAVA la
#     detection (oracle 0.97 → 0.79). Per un detector VERO — e per giudicare davvero se le
#     augmentation bastano da sole — metti TRAIN_SUBSAMPLE=1 (più lento ma è il numero buono).
#
#  📏 L'eval AR stampa il blocco FP-BY-SOURCE (--diag_fp_source 1, solo misura): dice quanti
#     FP vengono dal ramo 'past' vs 'standard' → conferma diretta che il fake_past uccide i
#     fantasmi del passato (gli FP 'past' devono crollare vs il detector senza augmentation).
# ═════════════════════════════════════════════════════════════════════════════
CONFIG="RTDetrPastConditioned"
EPOCHS=20
TRAIN_BATCH_SIZE=16
NUM_WORKERS=8
DURATION=33,165,330
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
TRAIN_SUBSAMPLE=10           # 10% = DIREZIONE (undertraina la detection). Per il detector finale: 1
USE_WANDB=1
USE_NMS=0
USE_ANNOTATED=0
PHASE=2
# ── MIXED QUERY MODE (identico a base_scratch / pastdrop) ──
USE_MIXED_QUERY_MODE=1
P_BOTH=0.4                    # p(standard_only) = 1 - p_both - p_past = 0.40
P_PAST=0.2                    # p(past_only) = 0.20
QUERY_MODE='both'
NUM_STD_QUERIES=50
NUM_PAST_ANNOTATIONS=12       # <<< P=12
USE_CUSTOM_NORMALIZATION=1
SEED=42

# --- RAMI SEPARATI ---
USE_SHARED_WEIGHTS=0
SHARED_CAT=0

# --- DETECTOR ONLY (niente testa di forecasting) ---
NUM_FUTURE_ANNOTATIONS=0
NUM_FUTURE_STEPS=0
FORECAST_HEAD_TYPE=transformer

# --- FROM SCRATCH (da HF/COCO) ---
FREEZE_DETECTOR=0
PRETRAINED_DETECTOR_PATH=""
TRAINABLE_WHEN_FROZEN=past_encoder   # ignorato con FREEZE_DETECTOR=0

# ── AUGMENTATION ANTI-DERIVA: ENTRAMBE ON ──
USE_PAST_DROPOUT=1           # <<< dropout del passato → riduce la dipendenza dal passato (recall/interferenza)
PAST_DROPOUT_P=0.3
USE_FAKE_PAST=1             # <<< passati finti → background (uccide i track fantasma) (precision)
FAKE_PAST_P=0.3
FAKE_MAX_K=3
FAKE_COLLIDE_THR=0.15

PERSISTENT_WORKERS=0

WANDB_ENTITY="edoardocappelli1999"
WANDB_PROJECT="detector_base"
WANDB_DIR="/equilibrium/ecappelli"
RUN_NAME="detector_scratch_p12_mixed_pastdrop_fakepast"
VIS_FREQ=1000
VIS_EVERY_N_BATCHES=1000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="/seidenas/datasets/FRED/preprocessed/"
RESUME_CHECKPOINT=""          # FRESH

# --- EVALUATION ---
EVAL_BATCH_SIZE=16
EVAL_SUBSAMPLE=1
EVALUATOR_TYPE=past_conditioned_detr
EVAL_USE_NMS=1
EVAL_USE_ANNOTATED=0
EVAL_PROCESSOR_TH=0.3

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"   # FRESH: nuova RUN_DIR ogni lancio
RUNS_DIR="/equilibrium/ecappelli/runs"
RUN_DIR="${RUNS_DIR}/${RUN_NAME}_${TIMESTAMP}"
BEST_MODEL_PATH="${RUN_DIR}/checkpoints/best_model.pt"
TRAIN_LOG_FILE="${RUN_DIR}/train_${RUN_NAME}.log"
TEST_LOG_FILE="${RUN_DIR}/test_${RUN_NAME}.log"

AVAIL_G=$(df -BG --output=avail /equilibrium 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${AVAIL_G:-0}" -lt 25 ]; then
    echo "❌ Solo ${AVAIL_G}G liberi su /equilibrium (rami separati → checkpoint pesante)."; exit 1
fi

mkdir -p "${RUN_DIR}"

ARCH_ARGS=(
    --use_shared_weights "${USE_SHARED_WEIGHTS}"
    --shared_cat         "${SHARED_CAT}"
    --num_past_annotations   "${NUM_PAST_ANNOTATIONS}"
    --num_future_annotations "${NUM_FUTURE_ANNOTATIONS}"
    --num_future_steps       "${NUM_FUTURE_STEPS}"
    --forecast_head_type     "${FORECAST_HEAD_TYPE}"
)

# ── TRAINING ──
{
    echo "============================================================"
    echo "  ${RUN_NAME} — Training (detector P=12, RAMI SEPARATI, da HF)"
    echo "  Run dir : ${RUN_DIR}  |  Free: ${AVAIL_G}G su /equilibrium"
    echo "  Anti-deriva: past_dropout=${USE_PAST_DROPOUT} | fake_past=${USE_FAKE_PAST} (solo batch 'both')"
    echo "  Atteso nel log: NESSUNA riga '[pretrained] Carico detector' (da HF) | NESSUN '[forecasting]' (T=0)"
    echo "============================================================"
    set +e
    RESUME_ARGS=()
    if [ -n "${RESUME_CHECKPOINT}" ]; then
        if [ -f "${RESUME_CHECKPOINT}" ]; then
            echo "   Resume da: ${RESUME_CHECKPOINT}"
            RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
        else
            echo "⚠️  RESUME_CHECKPOINT non trovato: ${RESUME_CHECKPOINT} — parto da zero."
        fi
    fi
    python3 -u main.py \
        --mode train --index_path "${INDEX_PATH}" --config "${CONFIG}" \
        --epochs "${EPOCHS}" --subsample "${TRAIN_SUBSAMPLE}" --num_workers "${NUM_WORKERS}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" --durations "${DURATION}" \
        --learning_rate "${LEARNING_RATE}" --weight_decay "${WEIGHT_DECAY}" --optimizer "${OPTIMIZER}" \
        --output_dir "${RUN_DIR}" --wandb_path "${WANDB_DIR}" \
        --phase "${PHASE}" --query_mode "${QUERY_MODE}" --num_standard_queries "${NUM_STD_QUERIES}" \
        --use_wandb "${USE_WANDB}" --wandb_entity "${WANDB_ENTITY}" --wandb_project "${WANDB_PROJECT}" \
        --run_name "${RUN_NAME}" --use_nms "${USE_NMS}" --use_only_annotated "${USE_ANNOTATED}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --seed "${SEED}" \
        --freeze_detector "${FREEZE_DETECTOR}" --pretrained_detector_path "${PRETRAINED_DETECTOR_PATH}" \
        --use_past_dropout "${USE_PAST_DROPOUT}" --past_dropout_p "${PAST_DROPOUT_P}" \
        --use_fake_past "${USE_FAKE_PAST}" --fake_past_p "${FAKE_PAST_P}" \
        --fake_max_k "${FAKE_MAX_K}" --fake_collide_thr "${FAKE_COLLIDE_THR}" \
        --use_mixed_query_mode "${USE_MIXED_QUERY_MODE}" --p_both "${P_BOTH}" --p_past "${P_PAST}" \
        "${ARCH_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --trainable_when_frozen "${TRAINABLE_WHEN_FROZEN}" \
        --persistent_workers "${PERSISTENT_WORKERS}" --vis_freq "${VIS_FREQ}"
    echo "  Training finished — exit ${?}"
    set -e
} 2>&1 | tee -a "${TRAIN_LOG_FILE}"

# ── EVALUATION: mAP (EF=0) + autoregressive (con FP-by-source). NIENTE EF=1 (T=0 → errore). ──
{
    echo ""
    echo "============================================================"
    echo "  ${RUN_NAME} — Evaluation (detection mAP + autoregressive + FP-by-source)"
    echo "============================================================"
    if [ ! -f "${BEST_MODEL_PATH}" ]; then
        echo "❌ best_model.pt non trovato: ${BEST_MODEL_PATH}"; exit 1
    fi
    set +e
    for QM in both standard_only; do
        echo "── DETECTION mAP (oracle) — query_mode=${QM} ──"
        python3 -u main.py \
            --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
            --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
            --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample "${EVAL_SUBSAMPLE}" \
            --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode "${QM}" \
            --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
            --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
            --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
            --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 0 --autoregressive 0 \
            --diag_best_iou 1
    done

    echo "── AUTOREGRESSIVE closed-loop (+ export MOT + FP-by-source) ──"
    python3 -u main.py \
        --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
        --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
        --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample "${EVAL_SUBSAMPLE}" \
        --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode "${QUERY_MODE}" \
        --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
        --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
        --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 0 --autoregressive 1 \
        --ar_export_mot 1 --diag_fp_source 1
    set -e
    echo "  Evaluation finished"
} 2>&1 | tee -a "${TEST_LOG_FILE}"
