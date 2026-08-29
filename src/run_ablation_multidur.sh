#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

# ═════════════════════════════════════════════════════════════════════════════
#  ABLATION MULTI-DURATA — "quanto aiuta l'input a multi-accumulazione?"
#
#  Allena lo STESSO detector (ricetta detector_scratch_p12_mixed: rami separati, P=12,
#  mixed, da HF/COCO) variando SOLO le durate di accumulazione, e ne misura la mAP
#  oracle (both + standard_only). Il trend mAP-vs-#durate risponde alla domanda.
#
#  Configurazioni (3):   33  |  33,165  |  33,165,330
#  Obiettivo: verificare se la map_50 SALE aggiungendo accumuli (1 → 2 → 3 durate).
#
#  ▸ LANCI:
#     - senza argomenti  →  cicla tutte e 3 in sequenza (1 GPU)
#     - con un argomento →  allena SOLO quella config, per 3 lanci PARALLELI su 3 GPU:
#         CUDA_VISIBLE_DEVICES=0 ./run_ablation_multidur.sh "33" &
#         CUDA_VISIBLE_DEVICES=1 ./run_ablation_multidur.sh "33,165" &
#         CUDA_VISIBLE_DEVICES=2 ./run_ablation_multidur.sh "33,165,330" &
#
#  ⚠️ CONFOUND (onestà sperimentale): rami SEPARATI = un RT-DETR per durata → più durate
#     = più parametri. Il trend include quindi anche l'effetto-capacità. Va bene come
#     ablation "del NOSTRO design multi-durata" (il modello finale è separate 33/165/330).
#     Per isolare il SOLO effetto-input servirebbe un controllo a pesi condivisi
#     (use_shared_weights=1): dimmelo e lo aggiungo come secondo script.
#
#  ▸ Config extra: puoi passare a mano "3,33,165,330" come argomento, ma richiede l'accumulo
#     a 3 ms nel preprocessed FRED (altrimenti quella sola run fallisce al caricamento).
# ═════════════════════════════════════════════════════════════════════════════

# ── Le 4 configurazioni di durata (o una sola se passata come argomento) ──
if [ $# -ge 1 ]; then
    DUR_CONFIGS=("$1")
else
    DUR_CONFIGS=("33" "33,165" "33,165,330")
fi

# ── Iperparametri FISSI: identici per TUTTE le config → unica variabile = le durate ──
CONFIG="RTDetrPastConditioned"
EPOCHS=20
TRAIN_BATCH_SIZE=16
NUM_WORKERS=8
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
OPTIMIZER="adamw"
TRAIN_SUBSAMPLE=${SUBSAMPLE:-10}   # env: SUBSAMPLE=1 → dati pieni (numero finale). Default 10 (direzione)
# tag run_name: sub-10 senza suffisso (compat coi run già fatti); altri subsample → suffisso s{N}_
if [ "${TRAIN_SUBSAMPLE}" = "10" ]; then SUBTAG=""; else SUBTAG="s${TRAIN_SUBSAMPLE}_"; fi
USE_WANDB=1
USE_NMS=0
USE_ANNOTATED=0
PHASE=2
USE_MIXED_QUERY_MODE=${USE_MIXED:-1}   # env: USE_MIXED=0 → 'both' puro (ablation mixed vs no-mixed)
# tag per distinguere i run_name delle due varianti
if [ "${USE_MIXED_QUERY_MODE}" = "1" ]; then MIXTAG=""; else MIXTAG="nomix_"; fi
P_BOTH=0.4
P_PAST=0.2
QUERY_MODE='both'
NUM_STD_QUERIES=50
NUM_PAST_ANNOTATIONS=12       # P=12, FISSO (varia solo la durata)
USE_CUSTOM_NORMALIZATION=1
SEED=42                       # stesso seed per tutte → confronto equo

# rami separati (come il modello finale), detector-only (T=0), da scratch
USE_SHARED_WEIGHTS=${SHARED:-0}   # env: SHARED=1 → pesi CONDIVISI (controllo capacità: #params costante al variare delle durate)
if [ "${USE_SHARED_WEIGHTS}" = "1" ]; then SHTAG="shared_"; else SHTAG=""; fi
SHARED_CAT=0
NUM_FUTURE_ANNOTATIONS=0
NUM_FUTURE_STEPS=0            # 0 = forecasting_head non creata (ablation di DETECTION)
FORECAST_HEAD_TYPE=transformer
FREEZE_DETECTOR=0
PRETRAINED_DETECTOR_PATH=""   # VUOTO = da HF/COCO
TRAINABLE_WHEN_FROZEN=past_encoder

# augmentation OFF (base pulita)
USE_PAST_DROPOUT=0
PAST_DROPOUT_P=0.3
USE_FAKE_PAST=0
FAKE_PAST_P=0.3
FAKE_MAX_K=3
FAKE_COLLIDE_THR=0.15

PERSISTENT_WORKERS=0

WANDB_ENTITY=${WANDB_ENTITY:-edoardocappelli099-org}
WANDB_PROJECT="ablation_multidur"
WANDB_DIR="/equilibrium/ecappelli"
VIS_FREQ=1000
VIS_EVERY_N_BATCHES=1000
INDEX_PATH="/seidenas/datasets/FRED/preprocessed/"
RUNS_DIR="/equilibrium/ecappelli/runs"

# --- EVALUATION: mAP PIENA (subsample=1) per numeri affidabili anche se il train è sub-campionato ---
EVAL_BATCH_SIZE=16
EVAL_SUBSAMPLE=1
EVALUATOR_TYPE=past_conditioned_detr
EVAL_USE_NMS=1
EVAL_USE_ANNOTATED=0
EVAL_PROCESSOR_TH=0.3

# Flag di architettura FISSI (identici train ed eval, o strict-load fallisce).
# Le DURATE si passano a parte (--durations), perché sono l'unica cosa che varia.
ARCH_ARGS=(
    --use_shared_weights "${USE_SHARED_WEIGHTS}"
    --shared_cat         "${SHARED_CAT}"
    --num_past_annotations   "${NUM_PAST_ANNOTATIONS}"
    --num_future_annotations "${NUM_FUTURE_ANNOTATIONS}"
    --num_future_steps       "${NUM_FUTURE_STEPS}"
    --forecast_head_type     "${FORECAST_HEAD_TYPE}"
)

echo "############################################################"
echo "#  ABLATION MULTI-DURATA — configs: ${DUR_CONFIGS[*]}  |  mixed=${USE_MIXED_QUERY_MODE}  shared=${USE_SHARED_WEIGHTS} (run: ablation_multidur_${SHTAG}${MIXTAG}*)"
echo "#  subsample_train=${TRAIN_SUBSAMPLE}  epochs=${EPOCHS}  P=${NUM_PAST_ANNOTATIONS}  (rami separati)"
echo "############################################################"

for DURATION in "${DUR_CONFIGS[@]}"; do
    DURATION="${DURATION// /}"                 # tolgo spazi ("3, 33" → "3,33")
    DUR_TAG="${DURATION//,/_}"                  # "33,165,330" → "33_165_330"
    RUN_NAME="ablation_multidur_${SUBTAG}${SHTAG}${MIXTAG}${DUR_TAG}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${RUNS_DIR}/${RUN_NAME}_${TIMESTAMP}"
    BEST_MODEL_PATH="${RUN_DIR}/checkpoints/best_model.pt"
    TRAIN_LOG_FILE="${RUN_DIR}/train_${RUN_NAME}.log"
    TEST_LOG_FILE="${RUN_DIR}/test_${RUN_NAME}.log"

    AVAIL_G=$(df -BG --output=avail /equilibrium 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ "${AVAIL_G:-0}" -lt 25 ]; then
        echo "❌ [${DURATION}] Solo ${AVAIL_G}G liberi su /equilibrium — SKIP questa config."; continue
    fi
    mkdir -p "${RUN_DIR}"

    # ── TRAINING ──
    {
        echo "============================================================"
        echo "  ${RUN_NAME} — Training (durate=${DURATION}, rami separati, da HF)"
        echo "  Run dir : ${RUN_DIR}  |  Free: ${AVAIL_G}G"
        echo "  Spie: NESSUN '[pretrained] Carico detector' (da HF) | NESSUN '[forecasting]' (T=0)"
        echo "============================================================"
        set +e
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
            --trainable_when_frozen "${TRAINABLE_WHEN_FROZEN}" \
            --persistent_workers "${PERSISTENT_WORKERS}" --vis_freq "${VIS_FREQ}"
        echo "  Training finished — exit ${?}"
        set -e
    } 2>&1 | tee -a "${TRAIN_LOG_FILE}"

    # ── EVALUATION: solo mAP detection (both + standard_only + diag). NO forecasting, NO AR. ──
    {
        echo ""
        echo "============================================================"
        echo "  ${RUN_NAME} — Evaluation (detection mAP oracle, durate=${DURATION})"
        echo "============================================================"
        if [ ! -f "${BEST_MODEL_PATH}" ]; then
            echo "❌ best_model.pt non trovato: ${BEST_MODEL_PATH} — SKIP eval di questa config."
        else
            set +e
            for QM in both standard_only; do
                echo "── DETECTION mAP (oracle) — durate=${DURATION} — query_mode=${QM} ──"
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
            set -e
        fi
        echo "  Evaluation finished — durate=${DURATION}"
    } 2>&1 | tee -a "${TEST_LOG_FILE}"
done

echo "############################################################"
echo "#  ABLATION MULTI-DURATA finita. Estrai i numeri con:"
echo "#    python3 extract_results.py --run-filter ablation_multidur"
echo "############################################################"
