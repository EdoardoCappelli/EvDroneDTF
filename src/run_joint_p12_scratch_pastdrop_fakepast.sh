#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

# ═════════════════════════════════════════════════════════════════════════════
#  JOINT DETECTOR + FORECASTER — from scratch (COCO), P=12  ·  past_dropout + fake_past
#
#  Gemello di run_joint_p12_scratch.sh MA con USE_FAKE_PAST=1 (oltre a past_dropout).
#  Perché: il joint con solo past_dropout (fake_past=0) in AR faceva MOTA −45 (track
#  fantasma auto-sostenuti). Il fake_past è il killer diretto: insegna che un passato
#  che punta a NON-droni → background → il track fantasma prende score basso → potato.
#  (Il forecasting AMPLIFICA i fantasmi perché è allenato a estrapolare traiettorie,
#  quindi qui il fake_past serve ancora di più.)
#
#  Alleno TUTTO insieme in 'both': ramo passato + ramo standard (loss UFFICIALE
#  RT-DETR) + forecasting.
#  🔴 use_shared_weights=0 (rami separati).  🔵 P=12 (400ms, competitor-comparable).
#  ⚖️  std_loss_weight=0.5 per non schiacciare passato+forecasting.
# ═════════════════════════════════════════════════════════════════════════════
CONFIG="RTDetrPastConditioned"
EPOCHS=20
TRAIN_BATCH_SIZE=16
NUM_WORKERS=8
DURATION=33,165,330
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
TRAIN_SUBSAMPLE=10            # 10% = test di DIREZIONE. Per il numero del paper: 1
USE_WANDB=1
USE_NMS=0
USE_ANNOTATED=0
PHASE=2
# ── QUERY MODE: 'both' puro (past + standard ogni batch) → forecasting allenato SEMPRE ──
USE_MIXED_QUERY_MODE=0
P_BOTH=0.6
P_PAST=0.2
QUERY_MODE='both'
NUM_STD_QUERIES=50
NUM_PAST_ANNOTATIONS=12       # <<< P=12
USE_CUSTOM_NORMALIZATION=1
SEED=42

# --- RAMI SEPARATI (compatibilità forecasting) ---
USE_SHARED_WEIGHTS=0
SHARED_CAT=0

# ── FORECASTING (task ausiliario) ──
NUM_FUTURE_ANNOTATIONS=24
NUM_FUTURE_STEPS=24           # T=24 → riporta 0.4s (12 step) e 0.8s (24 step) dalla stessa run
FORECAST_HEAD_TYPE=transformer
FORECAST_LOSS_WEIGHT=1.0      # DA TARARE {0.5,1.0,2.0}
USE_CV_ANCHOR=0               # 0 = niente ancora CV → la testa impara il moto
VEL_AVG_K=3
STD_LOSS_WEIGHT=0.5           # <<< bilancia il ramo standard vs passato+forecasting in joint

# ── JOINT from scratch: da HF/COCO, allena tutto ──
FREEZE_DETECTOR=0
PRETRAINED_DETECTOR_PATH=""   # VUOTO = da HF (COCO)
TRAINABLE_WHEN_FROZEN=forecasting_head   # ignorato con FREEZE_DETECTOR=0

# ── augmentation ANTI-FANTASMA: past_dropout + fake_past ON ──
USE_PAST_DROPOUT=1
PAST_DROPOUT_P=0.3
USE_FAKE_PAST=1              # <<< la differenza vs run_joint_p12_scratch (era 0): killer dei track fantasma
FAKE_PAST_P=0.3
FAKE_MAX_K=3
FAKE_COLLIDE_THR=0.15

PERSISTENT_WORKERS=0

WANDB_ENTITY=${WANDB_ENTITY:-edoardocappelli099-org}
WANDB_PROJECT="joint_det_forecast"
WANDB_DIR="/equilibrium/ecappelli"
RUN_NAME="joint_scratch_p12_pastdrop_fakepast"
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

# Flag di ARCHITETTURA: identici fra training ed eval, o strict-load fallisce.
# num_past_annotations tocca la shape del PastEncoder → DEVE stare qui.
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
    echo "  ${RUN_NAME} — Training (JOINT det+forecast, P=12, past_dropout+fake_past)"
    echo "  Run dir : ${RUN_DIR}  |  Free: ${AVAIL_G}G su /equilibrium"
    echo "  Attesi nel log: NESSUN '[pretrained] Carico detector' (da HF) | '[forecasting] head = transformer'"
    echo "  Spia fix: wandb train/std_loss_*_enc deve SCENDERE (enc_score_head impara)"
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
        --forecast_loss_weight "${FORECAST_LOSS_WEIGHT}" --use_cv_anchor "${USE_CV_ANCHOR}" --vel_avg_k "${VEL_AVG_K}" \
        --std_loss_weight "${STD_LOSS_WEIGHT}" \
        "${ARCH_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --trainable_when_frozen "${TRAINABLE_WHEN_FROZEN}" \
        --persistent_workers "${PERSISTENT_WORKERS}" --vis_freq "${VIS_FREQ}"
    echo "  Training finished — exit ${?}"
    set -e
} 2>&1 | tee -a "${TRAIN_LOG_FILE}"

# ── EVALUATION: mAP (both + standard_only + diag) → ADE/FDE → autoregressive ──
{
    echo ""
    echo "============================================================"
    echo "  ${RUN_NAME} — Evaluation (mAP + ADE/FDE + autoregressive)"
    echo "============================================================"
    if [ ! -f "${BEST_MODEL_PATH}" ]; then
        echo "❌ best_model.pt non trovato: ${BEST_MODEL_PATH}"; exit 1
    fi
    set +e

    # 1) mAP detection: both e standard_only (standard_only = cold-start AR). diag = localizzazione.
    for QM in both standard_only; do
        echo "── DETECTION mAP (oracle) — query_mode=${QM} ──"
        python3 -u main.py \
            --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
            --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
            --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample "${EVAL_SUBSAMPLE}" \
            --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode "${QM}" \
            --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
            --use_cv_anchor "${USE_CV_ANCHOR}" --vel_avg_k "${VEL_AVG_K}" \
            --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
            --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
            --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 0 --autoregressive 0 \
            --diag_best_iou 1
    done

    # 2) Forecasting ADE/FDE (EF=1, sulle query-passato)
    echo "── FORECASTING ADE/FDE (--eval_forecasting 1) ──"
    python3 -u main.py \
        --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
        --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
        --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample "${EVAL_SUBSAMPLE}" \
        --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode both \
        --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
        --use_cv_anchor "${USE_CV_ANCHOR}" --vel_avg_k "${VEL_AVG_K}" \
        --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
        --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 1 --autoregressive 0

    # 3) Autoregressive closed-loop (+ export MOT + motmetrics)
    echo "── AUTOREGRESSIVE closed-loop (--autoregressive 1 + export MOT) ──"
    python3 -u main.py \
        --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
        --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
        --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample "${EVAL_SUBSAMPLE}" \
        --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode both \
        --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
        --use_cv_anchor "${USE_CV_ANCHOR}" --vel_avg_k "${VEL_AVG_K}" \
        --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
        --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 0 --autoregressive 1 \
        --ar_export_mot 1
    set -e
    echo "  Evaluation finished"
} 2>&1 | tee -a "${TEST_LOG_FILE}"
