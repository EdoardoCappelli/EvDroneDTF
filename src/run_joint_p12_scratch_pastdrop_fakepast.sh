#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

CONFIG="RTDetrPastConditioned"
EPOCHS=${EPOCHS:-20}
TRAIN_BATCH_SIZE=16
NUM_WORKERS=4
DURATION=${DURATION:-33,165,330}
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
TRAIN_SUBSAMPLE=${TRAIN_SUBSAMPLE:-10}             # 10 = test veloce, 1 = numeri del paper
USE_WANDB=1
USE_NMS=0
USE_ANNOTATED=0
PHASE=2
# --- Query mode ---
USE_MIXED_QUERY_MODE=${MIXED:-0}   # MIXED=1: mixed query mode (both/past_only/standard_only)
P_BOTH=0.4                         # solo con MIXED=1 (p_std = 0.4)
P_PAST=0.2
QUERY_MODE='both'
NUM_STD_QUERIES=50
NUM_PAST_ANNOTATIONS=12
USE_CUSTOM_NORMALIZATION=1
SEED=42

# --- Varianti (env) ---
USE_SHARED_WEIGHTS=${SHARED_WEIGHTS:-0}   # SHARED_WEIGHTS=1: un solo branch per tutte le durate
SHARED_CAT=${SHARED_CAT:-0}               # SHARED_CAT=1: durate in un solo forward (richiede SHARED_WEIGHTS=1)
USE_PAST_CLASS_HEAD=${PAST_HEAD:-0}   # PAST_HEAD=1: class_embed separata per il ramo passato
BLOCK_DIAG=${BLOCKDIAG:-0}            # BLOCKDIAG=1: niente attenzione past <-> std nel decoder
RENDER_MODE=${REPR:-metavision}      # metavision | time_surface | tencode | metavision_acc | delta
SELECT_NO_CLS=${NOCLS:-0}             # NOCLS=1: best/early-stop su val_loss senza cls
USE_PAST_AUG=${PASTAUG:-0}           # PASTAUG=1: random walk sulle past-box
PAST_AUG_STD=${PASTAUGSTD:-0.05}     # std del passo (frazione)
USE_PAST_NOISE=${PASTNOISE:-0}       # PASTNOISE=1: rumore AR(1) sul passato
USE_VANISH_PAST=${VANISH:-0}         # VANISH=1: droni che spariscono come negativi
USE_SELF_PAST=${SELFPAST:-0}         # SELFPAST=1: curriculum con passato closed-loop
SELF_PAST_SUBSET=${SELFPAST_SUBSET:-0.1}         # frazione delle sequenze di training per rollout
SELF_PAST_VAL_SUBSET=${SELFPAST_VAL_SUBSET:-0.2} # frazione delle sequenze di val per misurare q
SELF_PAST_REFRESH=${SELFPAST_REFRESH:-1}         # rollout ogni N epoche
SELF_PAST_WARMUP=${SELFPAST_WARMUP:-4}           # epoche di solo passato GT
SELF_PAST_QMIN=${SELFPAST_QMIN:-0.6}             # ramp: soglia di qualità del passato per far salire p
SELF_PAST_SCHEDULE=${SELFPAST_SCHEDULE:-ramp}    # ramp | product_iou | product_map
SELF_PAST_FLOOR=${SELFPAST_FLOOR:-0.0}           # product_*: minimo del fattore di qualità
SELF_PAST_BATCH=${SELFPAST_BATCH:-16}            # sequenze avanzate insieme nel rollout (1 = una alla volta)
SELF_PAST_WORKERS=${SELFPAST_WORKERS:-${NUM_WORKERS}}   # worker del rollout

# --- Forecasting ---
NUM_FUTURE_ANNOTATIONS=24
NUM_FUTURE_STEPS=24           # 0.4 s (12 step) e 0.8 s (24 step) dalla stessa run
FORECAST_HEAD_TYPE=transformer
FORECAST_LOSS_WEIGHT=1.0
USE_CV_ANCHOR=${CV:-0}        # CV=1: ancora a velocità costante
VEL_AVG_K=3
STD_LOSS_WEIGHT=0.5           # peso del ramo standard

# --- Joint da scratch (pesi HF/COCO) ---
FREEZE_DETECTOR=0
PRETRAINED_DETECTOR_PATH=""   # vuoto = pesi HF (COCO)
TRAINABLE_WHEN_FROZEN=forecasting_head   # ignorato con FREEZE_DETECTOR=0

# --- past_dropout + fake_past ---
USE_PAST_DROPOUT=1
PAST_DROPOUT_P=0.3
USE_FAKE_PAST=1
FAKE_PAST_P=0.3
FAKE_MAX_K=3
FAKE_COLLIDE_THR=0.15

PERSISTENT_WORKERS=0

WANDB_ENTITY=${WANDB_ENTITY:-edoardocappelli099-org}
WANDB_PROJECT="joint_det_forecast"
WANDB_DIR="/equilibrium/ecappelli"
# suffissi del run_name per le varianti
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
[ "${USE_PAST_NOISE}" = "1" ] && _VARTAG="${_VARTAG}_pastnoise"
[ "${USE_VANISH_PAST}" = "1" ] && _VARTAG="${_VARTAG}_vanish"
[ "${USE_SELF_PAST}" = "1" ] && _VARTAG="${_VARTAG}_selfpast"
[ "${USE_SELF_PAST}" = "1" ] && [ "${SELF_PAST_SCHEDULE}" != "ramp" ] && _VARTAG="${_VARTAG}_${SELF_PAST_SCHEDULE}"
[ -n "${SPLIT_TAG:-}" ] && _VARTAG="${_VARTAG}_${SPLIT_TAG}"   # es. SPLIT_TAG=challenging
RUN_NAME="joint_scratch_p12_pastdrop_fakepast${_VARTAG}"
VIS_FREQ=1000
VIS_EVERY_N_BATCHES=1000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SPLIT_TAG=challenging senza INDEX_PATH usa il preprocessed challenging
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
AR_STD_BOX_PRIORITY=${STDBOX:-0}   # STDBOX=1: vince la box standard, 0: vince il passato
AR_TRACKER=${AR_TRACKER:-legacy}   # legacy | tracktrack
AR_CONF_THR=${AR_CONF:-0.65}       # soglia score del tracker AR (default config 0.35)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"   # FRESH: nuova RUN_DIR ogni lancio
RUNS_DIR="/equilibrium/ecappelli/runs"
RUN_DIR="${RUNS_DIR}/${RUN_NAME}_${TIMESTAMP}"
BEST_MODEL_PATH="${RUN_DIR}/checkpoints/best_model.pt"
BEST_MODEL_PATH="${RUN_DIR}/checkpoints/last_checkpoint.pt"
TRAIN_LOG_FILE="${RUN_DIR}/train_${RUN_NAME}.log"
TEST_LOG_FILE="${RUN_DIR}/test_${RUN_NAME}.log"

AVAIL_G=$(df -BG --output=avail /equilibrium 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${AVAIL_G:-0}" -lt 25 ]; then
    echo "❌ Solo ${AVAIL_G}G liberi su /equilibrium (rami separati → checkpoint pesante)."; exit 1
fi

mkdir -p "${RUN_DIR}"

# flag di architettura: uguali fra training ed eval (strict load)
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

# --- Training ---
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
        --use_past_noise "${USE_PAST_NOISE}" --use_vanish_past "${USE_VANISH_PAST}" \
        --use_self_past "${USE_SELF_PAST}" --self_past_subset "${SELF_PAST_SUBSET}" \
        --self_past_val_subset "${SELF_PAST_VAL_SUBSET}" --self_past_refresh "${SELF_PAST_REFRESH}" \
        --self_past_warmup "${SELF_PAST_WARMUP}" --self_past_q_min "${SELF_PAST_QMIN}" --self_past_workers "${SELF_PAST_WORKERS}" \
        --self_past_batch "${SELF_PAST_BATCH}" \
        --self_past_schedule "${SELF_PAST_SCHEDULE}" --self_past_floor "${SELF_PAST_FLOOR}" \
        --ar_conf_thr "${AR_CONF_THR}" --ar_std_box_priority "${AR_STD_BOX_PRIORITY}" --ar_tracker "${AR_TRACKER}" \
        "${ARCH_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        --trainable_when_frozen "${TRAINABLE_WHEN_FROZEN}" \
        --persistent_workers "${PERSISTENT_WORKERS}" --vis_freq "${VIS_FREQ}"
    echo "  Training finished — exit ${?}"
    set -e
} 2>&1 | tee -a "${TRAIN_LOG_FILE}"

# --- Evaluation: mAP, ADE/FDE, tracking AR ---
{
    echo ""
    echo "============================================================"
    echo "  ${RUN_NAME} — Evaluation (mAP + ADE/FDE + tracking AR)"
    echo "============================================================"
    if [ ! -f "${BEST_MODEL_PATH}" ]; then
        echo "❌ best_model.pt non trovato: ${BEST_MODEL_PATH}"; exit 1
    fi
    set +e

    # 1) mAP detection, both e standard_only
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

    # 2) forecasting ADE/FDE
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

    # 3) tracking AR closed-loop (+ export MOT)
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
        --ar_conf_thr "${AR_CONF_THR}" --ar_tracker "${AR_TRACKER}"

    set -e
    echo "  Evaluation finished"
} 2>&1 | tee -a "${TEST_LOG_FILE}"
