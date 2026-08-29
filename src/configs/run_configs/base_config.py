from functools import partial
import os
import torch.nn as nn
import yaml
import json
from types import SimpleNamespace
import socket
import argparse


# ============================================================
# Base Configuration
# ============================================================
class BaseConfig(SimpleNamespace):
    """Base class providing shared behavior and utilities for all configs."""

    def __init__(self):
        self.dataset_args = {
            'dataset_name': 'FRED',
            'img_height': 720,
            'img_width': 1280,
            'subsample': 10,
            'render_mode': 'metavision',
            'index_path': '/home/gmagrini/datasets/FRED_split/preprocessed/',
            'durations': [3, 33, 165, 330],
            'prefetch_factor': 2,
            'use_only_annotated': 0,
            'use_custom_normalization': 0,
        }
        
        self.image_processor_args = {
            'processor_threshold_train': 0.00,
            'processor_threshold_eval': 0.01,
            'use_nms': 1,
            'nms_threshold': 0.45,
        }

        self.training_args = {
            'epochs': 20,
            'train_batch_size': 4,
            'num_workers': 24,
            'patience': 5,
            'resume_from_checkpoint': '',
            'checkpoint_dir': '../checkpoints/',   
            'checkpoint_root': '',
            'vis_freq': 200,
            'trainer_type': 'past_conditioned_detr',
            'task_modality': 'detection',   
            'mode': 'train',  # 'train', 'eval', or 'both'
            'val_epoch_freq': 2,
            'save_checkpoints_freq': 4,
            'freeze_decoder_epochs': 0,
            'queries': 50,
            'vfl_loss_coefficient': 1.0,
            'apply_bias_fix': 0,
            'use_event_box_paste': 0,
            'event_box_paste_p': 0.5,
            'event_box_paste_min_events': 30,
            'use_shared_weights': False,
            'shared_cat': False,
            # --- Benchmark throughput (sweep batch_size / num_workers) ---
            'bench_batches': 0, # >0 = misura N batch di training e poi esce (no epoche complete, no eval)
            'bench_warmup': 10, # batch iniziali esclusi dalla misura (spin-up worker + JIT numba)
            # --- Resume ---
            'resume_strict': 1, # 1 = in resume vuole match esatto chiavi/shape; 0 = carica parziale
            # --- DataLoader RAM ---
            'persistent_workers': 1,
            # Soglia GB sotto la quale _save_checkpoint logga un warning di spazio basso
            'min_free_gb_checkpoint': 2.0,
        }

        self.optimizer_args = {
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
        }

        self.testing_args = {
            'test_dataset_name': 'FRED',
            'test_batch_size': 4,
            'num_workers': 24,
            'confidence_threshold': 0.25,
            'checkpoint_path': None,
            'save_predictions': 0,
            'evaluator_type': 'past_conditioned_detr',
        }

        self.val_args = {'val_batch_size': 4, 'val_num_workers': 2}

        self.logging_args = {
            'use_wandb': 1,
            'hostname': socket.gethostname(),
            'wandb_entity': 'gmagrini',
            'wandb_project': 'FRED++',
            'wandb_path': '/equilibrium/ecappelli',  
            'log_every': 1,
            'save_path': '../runs/',
            'run_name': 'default_run',
            'vis_every_n_batches': 200,
            'vis_threshold': 0.25,
            'output_dir': '../outputs/',
            'save_every': 2
        }

        self.merge_args()


    # ============================================================
    # Shared Utilities
    # ============================================================
    def merge_args(self):
        for attr, value in list(self.__dict__.items()):
            if attr.endswith("_args") and isinstance(value, dict):
                self.__dict__.update(value)

    @staticmethod
    def load_model_config(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        def resolve(obj):
            if isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, str):
                if obj == "LayerNorm":
                    return partial(nn.LayerNorm, eps=1e-6)
                if obj == "ReLU":
                    return nn.ReLU
                return obj
            return obj

        return SimpleNamespace(**resolve(cfg))

    def to_json(self):
        return json.dumps(self.__dict__, default=str, indent=2)

    def save(self, path='config.json'):
        with open(path, 'w') as f:
            f.write(self.to_json())

    def print_args(self):
        print("===== CONFIGURATION =====")
        for k, v in self.__dict__.items():
            if not isinstance(v, dict):
                print(f"{k}: {v}")
        print("=========================")


    def serialize_args(self, file_path):
        """Save the current configuration to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(self.to_json())