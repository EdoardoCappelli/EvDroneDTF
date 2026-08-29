import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
import time
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from abc import ABC, abstractmethod
from data.dataset_factory import get_dataset
from transformers import DetrConfig
from utils.data_utils import collate_fn_with_processor
from PIL import Image as PILImage

import math
import torch.nn as nn

class BaseTrainer(ABC):
    """Trainer class for MultiDurationDetr model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Will be initialized in setup()
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')

        self.current_epoch = 0
        self.start_epoch, self.global_step, self.best_val_loss = 0, 0, float('inf')
        
        self._setup_dirs()
        self._setup_logging()
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        self._setup_scheduler()
        

    def _setup_dirs(self):
        """Create output directory structure."""
        # Root della run: runs/{run_name}/  → log e visualizations restano qui
        run_dir = os.path.join(self.config.output_dir)
        self.output_dir = run_dir
        
        ckpt_root = (getattr(self.config, 'checkpoint_root', '') or '').strip()
        if ckpt_root:
            run_name = os.path.basename(os.path.normpath(run_dir))
            self.checkpoint_dir = os.path.join(ckpt_root, run_name)
        else:
            self.checkpoint_dir = os.path.join(run_dir, "checkpoints")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "visualizations", "train"), exist_ok=True)
        print(f"[dirs] output_dir     : {self.output_dir}")
        print(f"[dirs] checkpoint_dir : {self.checkpoint_dir}")

        # Config salvata nei checkpoints
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w") as f:
            f.write(self.config.to_json())

    def _setup_datasets(self):
        """Initialize datasets and dataloaders."""
        print("  Setting up datasets and dataloaders...")

        # Determine modality - 'mode' could be 'train'/'eval' or 'detection'/'tracking'
        # Use 'detection' as default task modality
        task_modality = getattr(self.config, 'task_modality', 'detection')
        if task_modality not in ['detection', 'tracking']:
            task_modality = 'detection'  # Default fallback

        print(f"    Loading training dataset: {self.config.dataset_name} | Split: train")
        print(f"    Loading validation dataset: {self.config.dataset_name} | Split: val")
        print(f"    Using index file: {self.config.index_path}")
        print(f"    Task modality: {task_modality}")

        # Use collate_fn that processes both images AND annotations with image_processor
        # This ensures boxes are properly transformed when images are resized
        self.collate_fn = lambda batch: collate_fn_with_processor(  
            batch,
            image_processor=self.image_processor
        )

        self.train_dataset = get_dataset(
            self.config.dataset_name,
            split='train',
            modality=task_modality,
            config=self.config,
        )
        self.val_dataset = get_dataset(
            self.config.dataset_name,
            split='val',   
            modality=task_modality,
            config=self.config
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=self.config.prefetch_factor,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        print(f" Training samples: {len(self.train_dataset)} | Validation samples: {len(self.val_dataset)}")


    def _setup_model(self):
        """Subclasses must implement this to define their models and optimizer."""
        raise NotImplementedError


    @abstractmethod
    def _setup_optimizer(self):
        """Subclasses must implement this to define their models and optimizer."""
        raise NotImplementedError
        

    def _setup_logging(self):
        """Initializes Weights & Biases for training."""
        if self.config.use_wandb:
            
            import wandb

            # wandb scrive la sua cartella locale in <wandb_path>/wandb/ e può crescere
            # parecchio: assicuriamoci che la dir esista e NON stia su un filesystem pieno
            # (con wandb_path='../' finiva accanto al progetto → /andromeda, saturo).
            wandb_dir = self.config.wandb_path
            os.makedirs(wandb_dir, exist_ok=True)
            print(f"[dirs] wandb_dir      : {os.path.join(wandb_dir, 'wandb')}")

            self.run = wandb.init(
                name=self.config.train_full_run_name,
                config={k: v for k, v in self.config.__dict__.items() if not k.startswith('__')},
                dir=wandb_dir,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
            )

            print(f"[Trainer] Weights & Biases run initialized: {self.run.name}")
        else:
            self.run = None
            print("[Trainer] Weights & Biases is disabled.")
                
        
    def _prepare_targets(self, targets: List[Dict]) -> List[Dict]:
        """Convert targets to DETR format and move to device.

        Handles both:
        - Old format: {'labels': ..., 'boxes': ...}
        - DetrImageProcessor format: {'class_labels': ..., 'boxes': ...}
        """
        prepared = []
        for t in targets:
            # Handle both 'labels' and 'class_labels' keys
            if 'class_labels' in t:
                labels = t['class_labels']
            elif 'labels' in t:
                labels = t['labels']
            else:
                labels = torch.zeros((0,), dtype=torch.int64)

            boxes = t.get('boxes', torch.zeros((0, 4)))

            prepared.append({
                'class_labels': labels.to(self.device),
                'boxes': boxes.to(self.device),
            })
        return prepared
    
        
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_vfl = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_giou = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):

            # Move frames to device
            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            detr_labels = self._prepare_targets(batch['labels'])
            
            # Forward pass
            if self.config.model_name == 'rtdetrv2':
                duration=33
                frames = frames_dict[duration]

                labels = []
                for lbl in detr_labels:
                    labels.append({
                        "class_labels": lbl["class_labels"],   # LongTensor [N]
                        "boxes":        lbl["boxes"],           # FloatTensor [N, 4] cxcywh norm
                    })
                outputs = self.model(pixel_values=frames, labels=labels)
            else:
                outputs = self.model(event_frames=frames_dict, labels=detr_labels)
            
            loss = outputs.loss
            
            # Visualization
            self.global_step = epoch * len(self.train_loader) + batch_idx
            if self.global_step % self.config.vis_freq == 0: 
                self._save_visualization(frames_dict, batch['labels'], loss.item(), epoch)
        
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate losses
            epoch_loss += loss.item()

            if outputs.loss_dict:
                # epoch_loss_ce += outputs.loss_dict.get('loss_ce', torch.tensor(0)).item() # no con rdetrv2
                epoch_loss_vfl += outputs.loss_dict.get('loss_vfl', torch.tensor(0)).item()
                epoch_loss_bbox += outputs.loss_dict.get('loss_bbox', torch.tensor(0)).item()
                epoch_loss_giou += outputs.loss_dict.get('loss_giou', torch.tensor(0)).item()
            
            if self.config.use_wandb and self.run is not None:
                import wandb
                wandb.log({ 
                    "epoch": epoch + 1,
                    "train/loss_total": loss.item(), 
                    # "train/loss_ce": epoch_loss_ce, 
                    "train/loss_vfl": outputs.loss_dict.get('loss_vfl', torch.tensor(0)).item(), 
                    "train/loss_bbox": outputs.loss_dict.get('loss_bbox', torch.tensor(0)).item(), 
                    "train/loss_giou": outputs.loss_dict.get('loss_giou', torch.tensor(0)).item(), 
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'], 
                    "step": (epoch * len(self.train_loader)) + batch_idx, 
                })


            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            
        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            # 'loss_ce': epoch_loss_ce / num_batches,
            'loss_vfl': epoch_loss_vfl / num_batches,
            'loss_bbox': epoch_loss_bbox / num_batches,
            'loss_giou': epoch_loss_giou / num_batches,
        }
    
        
    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Run validation."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()

        val_loss      = 0.0
        val_loss_ce   = 0.0
        val_loss_vfl  = 0.0
        val_loss_bbox = 0.0
        val_loss_giou = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Same format as _train_epoch: batch is a dict with 'frames' and 'labels'
            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            detr_labels = self._prepare_targets(batch['labels'])
            
            if self.config.model_name == 'rtdetrv2':
                duration=33
                frames = frames_dict[duration]

                labels = []
                for lbl in detr_labels:
                    labels.append({
                        "class_labels": lbl["class_labels"],   # LongTensor [N]
                        "boxes":        lbl["boxes"],           # FloatTensor [N, 4] cxcywh norm
                    })
                outputs = self.model(pixel_values=frames, labels=labels)
            else:
                outputs = self.model(event_frames=frames_dict, labels=detr_labels)

            val_loss += outputs.loss.item()

            if outputs.loss_dict:
                # val_loss_ce   += outputs.loss_dict.get('loss_ce',   torch.tensor(0)).item()
                val_loss_vfl  += outputs.loss_dict.get('loss_vfl',  torch.tensor(0)).item()
                val_loss_bbox += outputs.loss_dict.get('loss_bbox', torch.tensor(0)).item()
                val_loss_giou += outputs.loss_dict.get('loss_giou', torch.tensor(0)).item()

        n = len(self.val_loader)
        avg_val_loss = val_loss / n

        if self.config.use_wandb:
            import wandb
            wandb.log({
                "val/loss_total": avg_val_loss,
                # "val/loss_ce":   val_loss_ce   / n,
                "val/loss_vfl":  val_loss_vfl  / n,
                "val/loss_bbox": val_loss_bbox / n,
                "val/loss_giou": val_loss_giou / n,
                "epoch":         epoch,
            })
        
        return avg_val_loss
    
    
    def _build_ground_truths_batch(self, targets: List[Dict]) -> List[Dict]:
        """
        Convert targets from collate_fn format to evaluation format.
 
        Input boxes:  cxcywh normalized, relative to processor output size
        Output boxes: xyxy absolute pixel coords in orig_size space
        """
        ground_truths_batch = []
 
        for target in targets:
            boxes  = target['boxes'].to(self.device)        # (N,4) cxcywh norm
            labels = target['class_labels'].to(self.device) # (N,)
 
            # Use the original image size (before processor resize)
            # orig_h = target['orig_size'][0].item()
            # orig_w = target['orig_size'][1].item()
            orig_h = 720
            orig_w = 1280
            
            if boxes.numel() > 0:
                cx, cy, w, h = boxes.unbind(-1)
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                boxes_xyxy = torch.zeros((0, 4), device=self.device)
 
            ground_truths_batch.append({
                'boxes':     boxes_xyxy,
                'labels':    labels,
                'orig_size': (orig_h, orig_w),  # kept for visualization
            })
 
        return ground_truths_batch

        
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (torch.save normale)."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'global_step': getattr(self, 'global_step', 0),
            'patience_counter': getattr(self, 'patience_counter', 0),
        }
        if getattr(self, 'scheduler', None) is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Early-warning spazio disco (non blocca). NB: la QUOTA per-utente del cluster NON
        # è visibile via shutil.disk_usage/statvfs — questo copre solo il disco fisico pieno.
        try:
            free_gb = shutil.disk_usage(self.checkpoint_dir).free / (1024 ** 3)
            min_gb = float(getattr(self.config, 'min_free_gb_checkpoint', 2.0))
            if free_gb < min_gb:
                tqdm.write(f"⚠️  [checkpoint] spazio libero basso: {free_gb:.2f} GB "
                           f"(< {min_gb} GB) su {self.checkpoint_dir} — rischio save fallito.")
        except Exception:
            pass

        # last_checkpoint.pt: salvato SEMPRE (ogni epoca), sovrascritto → resume dall'ultima epoca.
        last_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pt")
        torch.save(checkpoint, last_path)

        # best_model.pt: solo quando la validazione migliora.
        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, path)
            tqdm.write(f"💾 Best model saved: {path}")

        
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # ── Controllo chiavi/shape: intercetta un checkpoint di ARCHITETTURA diversa
        #    (es. testa forecasting cambiata dopo il salvataggio) prima di caricare. ──
        ckpt_sd = checkpoint['model_state_dict']
        model_sd = self.model.state_dict()
        ckpt_keys, model_keys = set(ckpt_sd.keys()), set(model_sd.keys())
        missing    = sorted(model_keys - ckpt_keys)   # nel modello, assenti nel checkpoint
        unexpected = sorted(ckpt_keys - model_keys)   # nel checkpoint, assenti nel modello
        mismatched = sorted(
            k for k in (model_keys & ckpt_keys)
            if tuple(model_sd[k].shape) != tuple(ckpt_sd[k].shape)
        )
        n_match = len(model_keys & ckpt_keys) - len(mismatched)
        print(f"[resume] key check: {n_match}/{len(model_keys)} tensori combaciano | "
              f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mismatched)}")

        def _preview(name, keys, fmt=None):
            if not keys:
                return
            print(f"[resume]   {name} ({len(keys)}):")
            for k in keys[:12]:
                print(f"[resume]     - {k}" + (f"  {fmt(k)}" if fmt else ""))
            if len(keys) > 12:
                print(f"[resume]     ... (+{len(keys) - 12} altre)")

        _preview("MISSING  (nel modello, non nel ckpt)", missing)
        _preview("UNEXPECTED (nel ckpt, non nel modello)", unexpected)
        _preview("SHAPE MISMATCH", mismatched,
                 fmt=lambda k: f"model={tuple(model_sd[k].shape)} ckpt={tuple(ckpt_sd[k].shape)}")

        strict_resume = bool(getattr(self.config, 'resume_strict', 1))
        if missing or unexpected or mismatched:
            msg = ("[resume] ⚠️  il checkpoint NON combacia con il modello attuale "
                   f"(missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={len(mismatched)}). "
                   "Probabile architettura diversa (es. ForecastingHead cambiata dopo il salvataggio).")
            if strict_resume:
                raise RuntimeError(
                    msg + " Interrompo per non riprendere un training incoerente. "
                    "Se vuoi forzare un caricamento parziale usa --resume_strict 0."
                )
            print(msg + " Continuo con caricamento PARZIALE: i tensori non combacianti restano ai valori attuali.")
            filtered = {k: v for k, v in ckpt_sd.items()
                        if k in model_keys and tuple(model_sd[k].shape) == tuple(ckpt_sd[k].shape)}
            self.model.load_state_dict(filtered, strict=False)
        else:
            self.model.load_state_dict(ckpt_sd, strict=True)   # match esatto → load pulito
            print("[resume] ✓ tutte le chiavi combaciano — load esatto")

        # Optimizer: se l'architettura è diversa i param-group non combaciano → non far crashare il resume
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"[resume] ⚠️  optimizer state non caricato ({e}); riparto con optimizer fresco.")

        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)

        # Ripristino dello scheduler (LR). Se il checkpoint non lo contiene (versione
        # vecchia), fast-forward al punto di ripresa così l'LR è coerente.
        if getattr(self, 'scheduler', None) is not None:
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif self.train_loader is not None:
                n_steps = self.current_epoch * len(self.train_loader)
                for _ in range(n_steps):
                    self.scheduler.step()
                print(f"[resume] scheduler fast-forward di {n_steps} step (checkpoint senza stato scheduler)")

        print(f"Loaded checkpoint from epoch {self.current_epoch} (global_step={self.global_step})")

    def apply_bias_fix(self, model, prior_prob=0.01):
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        fixed_count = 0
        for name, module in model.named_modules():
            if 'class_embed' in name and isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, bias_value)
                print(f" Bias fix applicato su: {name} → {bias_value:.4f}")
                fixed_count += 1
        
        if fixed_count == 0:
            print("  WARNING: nessun layer 'class_embed' Linear trovato")
        else:
            print(f"  → {fixed_count} layer modificati. sigmoid({bias_value:.3f}) ≈ {prior_prob}")
    
    def train(self):
        """Main training loop."""
        if self.model is None:
            self.setup()

        if self.config.apply_bias_fix:
            print("Bias fixed")
            self.apply_bias_fix(self.model)

        
        if self.config.freeze_decoder_epochs > 0:
            for name, param in self.model.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad = False
            print(f"[Train] Backbone frozen per le prime {self.config.freeze_decoder_epochs} epoche")

        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config.epochs):

            if epoch == self.config.freeze_decoder_epochs and self.config.freeze_decoder_epochs > 0:
                for param in self.model.parameters():
                    param.requires_grad = True
                print(f"[Train] Epoch {epoch+1}: backbone sbloccato, lr backbone = {self.optimizer.param_groups[0]['lr']:.2e}")

            self.current_epoch = epoch

            # Train   
            metrics = self._train_epoch(epoch)
            self.train_losses.append(metrics['loss'])

            # VFL instead of CE
            print(f"\nEpoch {epoch + 1} | Loss: {metrics['loss']:.4f} | "
                f"VFL: {metrics['loss_vfl']:.4f} | "
                f"BBox: {metrics['loss_bbox']:.4f} | "
                f"GIoU: {metrics['loss_giou']:.4f}")
                
            # Save curves and checkpoint
            self._save_checkpoint(epoch)
            
            # Validate
            if self.val_loader and epoch % self.config.val_epoch_freq == 0:
                val_loss = self._validate(epoch)
                self.val_losses.append(val_loss)

                # Early Stopping
                if val_loss < self.best_val_loss:
                    print(f"New Best Val Loss: {val_loss:.4f} (was {self.best_val_loss:.4f})")
                    self.best_val_loss = val_loss
                    self.patience_counter = 0  
                    self.patience = self.config.patience 
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                    
                
                if self.patience_counter >= self.patience:
                    print("\n" + "!" * 60)
                    print(f"EARLY STOPPING triggered at epoch {epoch + 1}")
                    print("!" * 60)
                    break
           
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Output: {self.output_dir}")
        
        return self.train_losses, self.val_losses



    @torch.no_grad()
    def _save_visualization(
        self,
        frames_dict: Dict[int, torch.Tensor],
        targets: List[Dict],
        loss: float,
        epoch: int,
    ):
        """Save visualization during training showing all duration frames in a row."""

        PRED_COLORS = [
            '#FF3333', '#FF8C00', '#00BFFF', '#FF00FF', '#FFD700',
            '#FF69B4', '#00CED1', '#FF6347', '#BA55D3', '#FFA07A',
        ]
        GT_COLOR = '#00FF00'

        self.model.eval()

        if self.config.model_name == 'rtdetrv2':
            duration = 33
            frames = frames_dict[duration]
            outputs = self.model(pixel_values=frames)
        else:
            outputs = self.model(event_frames=frames_dict)

        # Use _build_ground_truths_batch to convert targets to xyxy in orig_size coords
        # (targets here are the raw batch labels, which still carry orig_size)
        ground_truths = self._build_ground_truths_batch(targets)

        sorted_durations = sorted(frames_dict.keys(), reverse=False)
        num_durations    = len(sorted_durations)
        main_duration    = sorted_durations[0]

        # orig_size from last sample in batch (consistent with [-1] indexing below)
        orig_h, orig_w = ground_truths[-1]['orig_size']

        batch_size = next(iter(frames_dict.values())).shape[0]

        # Post-process using orig_size → predicted boxes in orig_size pixel coords
        target_sizes = [(orig_h, orig_w)] * batch_size
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.processor_threshold_train,
        )

        # Aspect-ratio-correct figure
        aspect = orig_w / orig_h
        fig, axes = plt.subplots(1, num_durations, figsize=(5 * num_durations * aspect, 5))
        if num_durations == 1:
            axes = [axes]

        fig.suptitle(f"Step {self.global_step} | Epoch {epoch + 1} | Loss: {loss:.4f}", fontsize=12)

        for idx, duration in enumerate(sorted_durations):
            ax           = axes[idx]
            image_tensor = frames_dict[duration][-1]  # Last sample in batch
            image        = image_tensor.cpu().numpy()

            # Resize tensor to orig_size via PIL for correct aspect ratio
            if image.shape[0] == 1:
                image   = image.squeeze(0)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image   = np.array(pil_img) / 255.0
                ax.imshow(image, cmap='gray')
            elif image.shape[0] == 3:
                image   = np.transpose(image, (1, 2, 0))
                image   = np.clip(image, 0, 1)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image   = np.array(pil_img) / 255.0
                ax.imshow(image)
            else:
                image = image[0]
                ax.imshow(image, cmap='gray')

            ax.set_title(f"{duration}ms")

            if duration == main_duration:
                legend_entries = []
                pred_idx = 0  # outside both loops

                # GT boxes: already xyxy in orig_size coords from _build_ground_truths_batch
                gt = ground_truths[-1]
                if 'boxes' in gt:
                    gt_boxes  = gt['boxes']
                    gt_labels = gt['labels']
                    if isinstance(gt_boxes,  torch.Tensor): gt_boxes  = gt_boxes.cpu().numpy()
                    if isinstance(gt_labels, torch.Tensor): gt_labels = gt_labels.cpu().numpy()

                    for box, label in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=GT_COLOR, facecolor='none'
                        )
                        ax.add_patch(rect)
                        legend_entries.append((GT_COLOR, "GT"))

                # Predicted boxes: already in orig_size coords from post_process
                if len(results) > 0:
                    for box, score, label in zip(results[-1]['boxes'], results[-1]['scores'], results[-1]['labels']):
                        color = PRED_COLORS[pred_idx % len(PRED_COLORS)]
                        x1, y1, x2, y2 = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        # add text: score and predicted class
                        ax.text(
                            x1, y1 - 5,
                            f"P{pred_idx + 1} {score:.2f} ({label.item()})",
                            fontsize=6, fontfamily='monospace', color=color,
                            va='bottom', ha='left',
                            bbox=dict(
                                boxstyle='round,pad=0.15',
                                facecolor='black', alpha=0.45, linewidth=0,
                            ),
                        )
                        legend_entries.append((color, f"P{pred_idx + 1} {score:.2f}"))
                        pred_idx += 1

                        if pred_idx >= 10:
                            break  # limit to top 10 predictions for visualization

                if legend_entries:
                    row_h   = 0.065
                    start_y = 0.98
                    for entry_idx, (color, text) in enumerate(legend_entries):
                        ax.text(
                            0.98, start_y - entry_idx * row_h,
                            f"● {text}",
                            transform=ax.transAxes, fontsize=7,
                            fontfamily='monospace', color=color,
                            va='top', ha='right',
                            bbox=dict(
                                boxstyle='round,pad=0.15',
                                facecolor='black', alpha=0.45, linewidth=0,
                            ),
                        )

            ax.axis('off')

        plt.tight_layout()

        filename  = f"step_{self.global_step:06d}_epoch_{epoch + 1:03d}.png"
        save_path = os.path.join(self.output_dir, "visualizations", "train", filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")

        self.model.train()