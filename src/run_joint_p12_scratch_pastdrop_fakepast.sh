#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

CONFIG="RTDetrPastConditioned"
EPOCHS=${EPOCHS:-20}          # env: EPOCHS=40 ./run_... per il test "train longer"
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
USE_MIXED_QUERY_MODE=${MIXED:-0}   # env: MIXED=1 → mixed query mode (both/past_only/standard_only)
P_BOTH=0.4                         # usato SOLO se MIXED=1 (p_std = 1-0.4-0.2 = 0.4); inerte se MIXED=0
P_PAST=0.2
QUERY_MODE='both'
NUM_STD_QUERIES=50
NUM_PAST_ANNOTATIONS=12       # <<< P=12
USE_CUSTOM_NORMALIZATION=1
SEED=42

# --- RAMI SEPARATI (compatibilità forecasting) ---
USE_SHARED_WEIGHTS=${SHARED_WEIGHTS:-0}   # env: SHARED_WEIGHTS=1 → un solo branch condiviso tra le durate
SHARED_CAT=${SHARED_CAT:-0}               # env: SHARED_CAT=1 → batcha le durate in UN forward (richiede SHARED_WEIGHTS=1)
USE_PAST_CLASS_HEAD=${PAST_HEAD:-0}   # env: PAST_HEAD=1 → class_embed DEDICATA al ramo passato (esperimento anti loss_cls↑)
BLOCK_DIAG=${BLOCKDIAG:-0}            # env: BLOCKDIAG=1 → self-attn a BLOCCHI nel decoder (past↔past, std↔std, NIENTE past↔std): ramo standard standalone in 'both'. Flag ARCHITETTURA: combacia train/eval
RENDER_MODE=${REPR:-metavision}      # rappresentazione input: metavision (default) | time_surface | tencode | metavision_acc | delta. env: REPR=time_surface. DEVE combaciare train/eval
SELECT_NO_CLS=${NOCLS:-0}             # env: NOCLS=1 → best/early-stop su val_loc (val_loss SENZA cls); la cls resta nel training
USE_PAST_AUG=${PASTAUG:-0}           # env: PASTAUG=1 → augmentation past-oracle (random walk sulle past-box reali) anti exposure-bias AR. Solo-training
PAST_AUG_STD=${PASTAUGSTD:-0.05}     # std del passo del random walk (frazione); CALIBRARE sull'errore AR misurato (AR oracle vs self)

# ── FORECASTING (task ausiliario) ──
NUM_FUTURE_ANNOTATIONS=24
NUM_FUTURE_STEPS=24           # T=24 → riporta 0.4s (12 step) e 0.8s (24 step) dalla stessa run
FORECAST_HEAD_TYPE=transformer
FORECAST_LOSS_WEIGHT=1.0      # DA TARARE {0.5,1.0,2.0}
USE_CV_ANCHOR=${CV:-0}        # env: CV=1 → ancora a velocità costante (present + t·vel + delta)
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
# tag run_name per le varianti (MIXED / CV) → non collidono col winner
_VARTAG=""
[ "${USE_MIXED_QUERY_MODE}" = "1" ] && _VARTAG="${_VARTAG}_mixed"
[ "${USE_CV_ANCHOR}" = "1" ] && _VARTAG="${_VARTAG}_cv"
[ "${USE_SHARED_WEIGHTS}" = "1" ] && _VARTAG="${_VARTAG}_shared"
[ "${SHARED_CAT}" = "1" ] && _VARTAG="${_VARTAG}_shared_cat"
[ "${USE_PAST_CLASS_HEAD}" = "1" ] && _VARTAG="${_VARTAG}_pasthead"
[ "${BLOCK_DIAG}" = "1" ] && _VARTAG="${_VARTAG}_blockdiag"
[ "${RENDER_MODE}" != "metavision" ] && _VARTAG="${_VARTAG}_${RENDER_MODE}"
[ "${SELECT_NO_CLS}" = "1" ] && _VARTAG="${_VARTAG}_nocls"
[ "${USE_PAST_AUG}" = "1" ] && _VARTAG="${_VARTAG}_pastaug"
[ -n "${SPLIT_TAG:-}" ] && _VARTAG="${_VARTAG}_${SPLIT_TAG}"   # es. SPLIT_TAG=challenging → run separata dal canonical
RUN_NAME="joint_scratch_p12_pastdrop_fakepast${_VARTAG}"
VIS_FREQ=1000
VIS_EVERY_N_BATCHES=1000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# --- SPLIT: SPLIT_TAG sceglie ANCHE i dati, non solo il nome ---------------------
# Se NON passi INDEX_PATH esplicito, SPLIT_TAG=challenging punta al preprocessed challenging.
# Un INDEX_PATH esplicito vince sempre. Così RUN_NAME (nome) e INDEX_PATH (dati) non divergono.
if [ -z "${INDEX_PATH:-}" ] && [ "${SPLIT_TAG:-}" = "challenging" ]; then
    INDEX_PATH="${CHALLENGING_INDEX_PATH:-/equilibrium/ecappelli/preprocessed_challenging/}"
fi
INDEX_PATH="${INDEX_PATH:-/seidenas/datasets/FRED/preprocessed/}"   # default: canonical
echo "[split] SPLIT_TAG='${SPLIT_TAG:-<none>}'  ->  INDEX_PATH='${INDEX_PATH}'"
RESUME_CHECKPOINT=""          # FRESH

# --- EVALUATION ---
EVAL_BATCH_SIZE=16
EVAL_SUBSAMPLE=1
EVALUATOR_TYPE=past_conditioned_detr
EVAL_USE_NMS=1
EVAL_USE_ANNOTATED=0
EVAL_PROCESSOR_TH=0.3
AR_STD_BOX_PRIORITY=${STDBOX:-0}   # env: STDBOX=1 → merge-fix (box standard vince, il passato riempie i buchi + dà identità → AR ≥ standard). 0 = M4 (passato prioritario)
AR_CONF_THR=${AR_CONF:-0.65}       # soglia score detection CORRENTE del tracker AR (default config 0.35; qui tarato 0.65). env: AR_CONF=0.5 ./run_...

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
    --use_past_class_head    "${USE_PAST_CLASS_HEAD}"
    --block_diag_decoder_attn "${BLOCK_DIAG}"
    --render_mode            "${RENDER_MODE}"
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
        --select_exclude_cls "${SELECT_NO_CLS}" \
        --use_past_aug "${USE_PAST_AUG}" --past_aug_std "${PAST_AUG_STD}" \
        "${ARCH_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --trainable_when_frozen "${TRAINABLE_WHEN_FROZEN}" \
        --persistent_workers "${PERSISTENT_WORKERS}" --vis_freq "${VIS_FREQ}"
    echo "  Training finished — exit ${?}"
    set -e
} 2>&1 | tee -a "${TRAIN_LOG_FILE}"

# ── EVALUATION: mAP (both + standard_only + diag) → ADE/FDE → tracking AR ──
{
    echo ""
    echo "============================================================"
    echo "  ${RUN_NAME} — Evaluation (mAP + ADE/FDE + tracking AR)"
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

    # 3) TRACKING autoregressivo closed-loop (passato = predizioni del modello; + export MOT + motmetrics)
    echo "── TRACKING AR closed-loop (--ar_oracle_past 0, merge-fix std_box_priority=${AR_STD_BOX_PRIORITY}) ──"
    python3 -u main.py \
        --config "${CONFIG}" --mode eval --evaluator_type "${EVALUATOR_TYPE}" \
        --checkpoint_path "${BEST_MODEL_PATH}" --test_batch_size "${EVAL_BATCH_SIZE}" \
        --durations "${DURATION}" --output_dir "${RUN_DIR}" --subsample 1 \
        --index_path "${INDEX_PATH}/" --phase "${PHASE}" --query_mode both \
        --num_standard_queries "${NUM_STD_QUERIES}" "${ARCH_ARGS[@]}" \
        --use_cv_anchor "${USE_CV_ANCHOR}" --vel_avg_k "${VEL_AVG_K}" \
        --processor_threshold_eval "${EVAL_PROCESSOR_TH}" --use_only_annotated "${EVAL_USE_ANNOTATED}" \
        --use_custom_normalization "${USE_CUSTOM_NORMALIZATION}" --use_nms "${EVAL_USE_NMS}" \
        --vis_every_n_batches "${VIS_EVERY_N_BATCHES}" --eval_forecasting 0 --autoregressive 1 \
        --ar_oracle_past 0 --ar_export_mot 1 --ar_std_box_priority "${AR_STD_BOX_PRIORITY}" \
        --ar_conf_thr "${AR_CONF_THR}"

    set -e
    echo "  Evaluation finished"
} 2>&1 | tee -a "${TEST_LOG_FILE}"
