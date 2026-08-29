import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from torchmetrics.detection import MeanAveragePrecision
from transformers import DetrImageProcessor, DetrForObjectDetection

from data.dataset_factory import get_dataset
from utils.data_utils import collate_fn_with_processor

from PIL import Image as PILImage   

from utils.data_utils import _apply_nms   

class BaseEvaluator(ABC):
    """Base evaluator class for object detection models."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.image_processor = None
        self.test_dataset = None
        self.test_loader = None
        self.CLASSES = ['drone']
        print(f"self.CLASSES: {self.CLASSES}")

        self.predictions = []
        self.ground_truths = []
        self.metrics = {}

        self._setup_dirs()
        self._setup_model()
        self._setup_dataset()


    def _setup_dirs(self):
        """Create output directory structure."""
        run_dir = os.path.join(self.config.output_dir)
        self.output_dir = run_dir

        os.makedirs(os.path.join(run_dir, "visualizations", "test"), exist_ok=True)

        # Save config
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            f.write(self.config.to_json())
        print(f"[Evaluator] Output directory created at: {self.output_dir}")


    def _setup_dataset(self):
        """Initialize test dataset and dataloader."""
        print("💿 Setting up test dataset and dataloader...")

        task_modality = getattr(self.config, 'task_modality', 'detection')
        if task_modality not in ['detection', 'tracking']:
            task_modality = 'detection'

        print(f" Loading test dataset: {self.config.dataset_name} | Split: test")
        print(f" Using index file: {self.config.index_path}")
        print(f" Task modality: {task_modality}")

        self.collate_fn = lambda batch: collate_fn_with_processor(
            batch,
            image_processor=self.image_processor
        )

        self.test_dataset = get_dataset(
            self.config.dataset_name,
            split='test',
            modality=task_modality,
            config=self.config
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        print(f" Test samples: {len(self.test_dataset)} | Test batches: {len(self.test_loader)}")


    @abstractmethod
    def _setup_model(self):
        """Subclasses must implement this to load their specific model."""
        raise NotImplementedError

    def _prepare_targets(self, targets: List[Dict]) -> List[Dict]:
        """Convert targets to DETR format and move to device."""
        return [
            {
                'class_labels': t['labels'].to(self.device),
                'boxes': t['boxes'].to(self.device),
            }
            for t in targets
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def _build_ground_truths_batch(self, targets: List[Dict]) -> List[Dict]:
        """
        Convert targets from collate_fn format to evaluation format.
 
        Input boxes:  cxcywh normalized, relative to processor output size
        Output boxes: xyxy absolute pixel coords in orig_size space
        """
        ground_truths_batch = []
 
        for target in targets:
            boxes = target['boxes'].to(self.device)        # (N,4) cxcywh norm
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
 
    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
 
    def _save_visualization(
        self,
        frames_dict: Dict[int, torch.Tensor],
        results: List[Dict],
        ground_truths: List[Dict],
        batch_idx: int,
        sample_idx: int = 0,
    ):
        """
        Save visualization showing all duration frames in a row.
 
        - Image is resized back to orig_size for display
        - GT boxes and predictions are both in orig_size pixel coords
        """
        sorted_durations = sorted(frames_dict.keys(), reverse=False)
        num_durations = len(sorted_durations)
        main_duration = sorted_durations[-1]
 
        # Retrieve orig_size for this sample
        orig_h, orig_w = ground_truths[sample_idx].get('orig_size', (640, 640))
 
        # Aspect-ratio-correct figure size
        aspect = orig_w / orig_h
        fig, axes = plt.subplots(
            1, num_durations,
            figsize=(5 * num_durations * aspect, 5)
        )
        if num_durations == 1:
            axes = [axes]
 
        fig.suptitle(f"Batch {batch_idx} | Sample {sample_idx}", fontsize=12)
 
        PRED_COLORS = [
            '#FF3333', '#FF8C00', '#00BFFF', '#FF00FF', '#FFD700',
            '#FF69B4', '#00CED1', '#FF6347', '#BA55D3', '#FFA07A',
        ]
        GT_COLOR = '#00FF00'
 
        for idx, duration in enumerate(sorted_durations):
            ax = axes[idx]
            image_tensor = frames_dict[duration][sample_idx]
            image = image_tensor.cpu().numpy()
 
            if image.shape[0] == 1:
                image = image.squeeze(0)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8), mode='L')
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image = np.array(pil_img) / 255.0
                ax.imshow(image, cmap='gray')
 
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image = np.clip(image, 0, 1)
                pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
                pil_img = pil_img.resize((orig_w, orig_h), PILImage.BILINEAR)
                image = np.array(pil_img) / 255.0
                ax.imshow(image)
 
            else:
                image = image[0]
                ax.imshow(image, cmap='gray')
 
            ax.set_title(f"{duration}ms")
 
            # Draw boxes only on main duration
            if duration == main_duration:
                legend_entries = []
                pred_idx = 0  
 
                # Ground truth boxes
                if len(ground_truths) > sample_idx and 'boxes' in ground_truths[sample_idx]:
                    gt_boxes = ground_truths[sample_idx]['boxes']
                    gt_labels = ground_truths[sample_idx]['labels']
 
                    if isinstance(gt_boxes,  torch.Tensor): gt_boxes = gt_boxes.cpu().numpy()
                    if isinstance(gt_labels, torch.Tensor): gt_labels = gt_labels.cpu().numpy()
 
                    for box, label in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=GT_COLOR, facecolor='none'
                        )
                        ax.add_patch(rect)
                        label_text = self.CLASSES[label] if label < len(self.CLASSES) else f"cls_{label}"
                        legend_entries.append((GT_COLOR, f"GT  {label_text}"))
 
                # Predicted boxes
                if len(results) > sample_idx:
                    pred_boxes = results[sample_idx]['boxes']
                    pred_scores = results[sample_idx]['scores']
                    pred_labels = results[sample_idx]['labels']
 
                    if isinstance(pred_boxes,  torch.Tensor): pred_boxes = pred_boxes.cpu().numpy()
                    if isinstance(pred_scores, torch.Tensor): pred_scores = pred_scores.cpu().numpy()
                    if isinstance(pred_labels, torch.Tensor): pred_labels = pred_labels.cpu().numpy()
 
                    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                        color = PRED_COLORS[pred_idx % len(PRED_COLORS)]
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=1, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        label_text = self.CLASSES[label] if label < len(self.CLASSES) else f"cls_{label}"
                        legend_entries.append((color, f"P{pred_idx + 1}  {label_text}  {score:.2f}"))
                        pred_idx += 1

                        if pred_idx >= 10:
                            break  # limit to top 10 predictions for visualization
 
                # Legend
                if legend_entries:
                    row_h = 0.065
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
                                facecolor='black', alpha=0.45, linewidth=0
                            ),
                        )
 
            ax.axis('off')
 
        plt.tight_layout()
        filename = f"batch_{batch_idx:06d}_sample_{sample_idx:02d}.png"
        out_dir = os.path.join(self.output_dir, "visualizations", f"test")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"batch_{batch_idx:06d}_sample_{sample_idx:02d}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization: {filename}")
 
    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        from torchvision.ops import box_iou
        """
        Run inference on test set and compute mAP + TP/FP/FN/Precision/Recall.
        """
        self.model.eval()
        self.metric = MeanAveragePrecision(iou_type='bbox')

        total_tp = 0
        total_fp = 0
        total_fn = 0

        iou_thr = getattr(self.config, 'eval_iou_threshold', 0.5)
        score_thr = getattr(self.config, 'processor_threshold_eval', 0.1)

        print("\nStarting evaluation...")
        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Inference")):

            frames_dict = {d: f.to(self.device) for d, f in batch['frames'].items()}
            targets = batch['labels']

            # Forward pass
            if self.config.model_name == 'rtdetrv2':
                frames = frames_dict[33]
                outputs = self.model(pixel_values=frames)
            else:
                outputs = self.model(event_frames=frames_dict)

            # Ground truths in xyxy pixel space
            ground_truths_batch = self._build_ground_truths_batch(targets)

            target_sizes = torch.tensor(
                [gt['orig_size'] for gt in ground_truths_batch]
            ).to(self.device)

            results = self.image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=score_thr,
            )

            if self.config.use_nms:
                results = _apply_nms(results, iou_threshold=self.config.nms_threshold)

            for pred, gt in zip(results, ground_truths_batch):
                pred_boxes = pred['boxes']   # [N, 4] xyxy assoluto
                gt_boxes = gt['boxes']     # [M, 4] xyxy assoluto

                n_pred = len(pred_boxes)
                n_gt = len(gt_boxes)

                if n_pred == 0 and n_gt == 0:
                    continue                         
                elif n_pred == 0:
                    total_fn += n_gt # tutto mancato
                    continue
                elif n_gt == 0:
                    total_fp += n_pred # tutto falso positivo
                    continue

                # [N_pred, N_gt]
                iou_matrix = box_iou(pred_boxes.cpu(), gt_boxes.cpu())

                matched_gt = set()
                matched_pred = set()

                sorted_iou = (iou_matrix >= iou_thr).nonzero(as_tuple=False)
                iou_vals = [(iou_matrix[i, j].item(), i.item(), j.item())
                            for i, j in sorted_iou]
                iou_vals.sort(key=lambda x: -x[0])

                for iou_val, i_pred, i_gt in iou_vals:
                    if i_pred not in matched_pred and i_gt not in matched_gt:
                        matched_pred.add(i_pred)
                        matched_gt.add(i_gt)

                tp = len(matched_pred)
                fp = n_pred - tp
                fn = n_gt   - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn

            if batch_idx % self.config.vis_every_n_batches == 0:
                self._save_visualization(
                    frames_dict=frames_dict,
                    results=results,
                    ground_truths=ground_truths_batch,
                    batch_idx=batch_idx,
                    sample_idx=0,
                )

            gt_for_metric = [
                {'boxes': gt['boxes'], 'labels': gt['labels']}
                for gt in ground_truths_batch
            ]
            self.metric.update(results, gt_for_metric)

        print("\nComputing final mAP score...")
        final_results = self.metric.compute()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

        print("\n" + "=" * 50)
        print(f"  IoU threshold per TP/FP/FN : {iou_thr}")
        print(f"  Score threshold            : {score_thr}")
        print("-" * 50)
        print(f"  True  Positives (TP) : {total_tp}")
        print(f"  False Positives (FP) : {total_fp}  ← rilevazioni sbagliate")
        print(f"  False Negatives (FN) : {total_fn}  ← droni mancati")
        print("-" * 50)
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print("=" * 50)

        import pprint
        pprint.pprint(final_results)

        final_results['tp'] = total_tp
        final_results['fp'] = total_fp
        final_results['fn'] = total_fn
        final_results['precision'] = precision
        final_results['recall'] = recall
        final_results['f1'] = f1

        return final_results
