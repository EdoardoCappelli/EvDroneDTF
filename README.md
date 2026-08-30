# EVDroneDFT  

Past conditioned multi-duration RT-DETRv2 for drone detection, tracking and forecasting on FRED dataset.

## Requirements
1. **conda env** `FRED++`:
   ```bash
   source /andromeda/personal/ecappelli/miniconda3/bin/activate
   conda activate FRED++
   ```
2. **Plugin HDF5**:
   ```bash
   export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
   export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH
   ```

## 1) Multi-accumulation ablation - `run_ablation_multidur.sh`
Train and evaluate the same detector varying the number of accumulations given as input to the model. It allows to measure how meaningfull is the multi-accumulated input. Seed=42 is used.


| env | default | desc |
|---|---|---|
| `SUBSAMPLE` | `10` | frazione train data |
| `USE_MIXED` | `1` | `1` = mixed query mode |
| `SHARED` | `0` | `1` = single encoder with shared weights |
| `WANDB_ENTITY` | `edoardocappelli099-org` | entity wandb |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use |

Examples:
```bash
SUBSAMPLE=10 ./run_ablation_multidur.sh "33,165,330"    
SUBSAMPLE=10 SHARED=1 ./run_ablation_multidur.sh "33,165,330" 
```

See the `train_*.log`, `test_*.log` for the details. In order to extract the evaluations results the extract_results.py file can be used.
```bash
python3 extract_results.py --run-filter ablation_multidur --since 20260829
```

## 2) Detection / Tracking video - `create_detection_video.py`
```bash
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

python3 create_detection_video.py \
  --checkpoint /path/to/run/checkpoints/best_model.pt \
  --durations 33 165 330 --num_past_steps 12 \
  --start_frame 0 --clip_seconds 20 \
  --iou_thr 0.2 --coher_thr 0.10 \
  --autoregressive 1 \
  --output detection.mp4
```

## 3) Forecasting video - `create_forecast_video.py`
```bash
export HDF5_PLUGIN_PATH=/seidenas/datasets/FRED/plugins
export LD_LIBRARY_PATH=/seidenas/datasets/FRED/plugins:$LD_LIBRARY_PATH

python3 create_forecast_video.py \
  --checkpoint /path/to/checkpoints/best_model.pt \
  --durations 33 165 330 --num_past_steps 12 --num_future_steps 24 \
  --forecast_head_type transformer --use_present_refine 1 --use_cv_anchor 0 --vel_avg_k 3 \
  --start_frame 0 --clip_seconds 20 \
  --autoregressive 1 \
  --output forecast.mp4
```

## Project structure

```
src/
  main.py                      
  run_*.sh                     # experiment scripts
  create_detection_video.py    # video detection/tracking
  create_forecast_video.py     # video forecasting
  extract_results.py           # extracts mAP/ADE/FDE/MOTA form test.log files -> summary_results.json
  configs/                     # config + factory
  models/Multiscale_ERT_Detr.py   # model
  train/trainers/              # trainer
  evaluation/evaluators/       # evaluator
  data/                        # dataset FRED + split train/val (seq_split)
  utils/                       # preprocessing dataset, box utils
```




