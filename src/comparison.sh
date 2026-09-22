#!/bin/bash
source /andromeda/personal/ecappelli/miniconda3/bin/activate
conda activate FRED++

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # override da env per lanci paralleli
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

set -euo pipefail

# ================= EDIT THESE =================
SRC=/andromeda/personal/ecappelli/EvDroneDTF/src                 # where main.py lives
CKPT=/equilibrium/ecappelli/runs/joint_scratch_p12_pastdrop_fakepast_20260827_161738/checkpoints/last_checkpoint.pt
INDEX=/seidenas/datasets/FRED/preprocessed                        # canonical (or challenging dir)
OUT=/equilibrium/ecappelli/runs/joint_scratch_p12_pastdrop_fakepast_20260827_161738/ar_cmp                      # FRESH dir for this comparison
DURATION=33,165,330
REPR=metavision            # MUST match training (metavision / tencode / ...)
NORM=1                     # MUST match training: 1 if trained with custom norm, else 0
CONF=0.65                  # same ar_conf_thr you used for the AR numbers you're worried about
# --- arch flags: MUST match training ---
NUM_PAST=12 ; NUM_FUT_ANN=24 ; NUM_FUT=24 ; FHEAD=transformer
PAST_HEAD=0 ; BLOCKDIAG=0 ; SHARED=0 ; SHARED_CAT=0 ; CVANCHOR=0
# =============================================

cd "$SRC"
mkdir -p "$OUT"
rm -rf "$OUT/mot_tracks" "$OUT/mot_tracks_oracle"                 # no stale files

COMMON="--config RTDetrPastConditioned --mode eval --evaluator_type past_conditioned_detr \
 --checkpoint_path $CKPT --test_batch_size 16 --durations $DURATION --output_dir $OUT \
 --subsample 1 --index_path $INDEX/ --phase 2 --query_mode both --num_standard_queries 50 \
 --num_past_annotations $NUM_PAST --num_future_annotations $NUM_FUT_ANN --num_future_steps $NUM_FUT \
 --forecast_head_type $FHEAD --use_past_class_head $PAST_HEAD --block_diag_decoder_attn $BLOCKDIAG \
 --render_mode $REPR --use_shared_weights $SHARED --shared_cat $SHARED_CAT \
 --use_cv_anchor $CVANCHOR --vel_avg_k 3 --processor_threshold_eval 0.3 --use_only_annotated 0 \
 --use_custom_normalization $NORM --use_nms 1 --vis_every_n_batches 999999 \
 --eval_forecasting 0 --autoregressive 1 --ar_export_mot 1 --ar_std_box_priority 0 --ar_conf_thr $CONF"

echo "########## AR — ORACLE PAST (past = GT) ##########"
python3 -u main.py $COMMON --ar_oracle_past 1 2>&1 | tee "$OUT/ar_oracle.log"

echo "########## AR — SELF PAST (past = own detections) ##########"
python3 -u main.py $COMMON --ar_oracle_past 0 2>&1 | tee "$OUT/ar_self.log"

echo ""; echo "===================== CONFRONTO ====================="
for L in ar_oracle ar_self; do
  echo "----- $L -----"
  grep -iE "mAP|MOTA|IDF1|OVERALL|switch|mota|idf1" "$OUT/$L.log" | tail -14
done