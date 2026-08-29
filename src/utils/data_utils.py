import torch
import numpy as np


def collate_fn_with_processor(batch, image_processor):
    """
    Collate function that uses DetrImageProcessor for both images AND annotations.

    This is the proper way to use DETR - the processor handles:
    1. Image resizing
    2. Box coordinate transformation to match resized images
    3. Proper label formatting

    The dataset should return:
    - images: PIL Images or numpy arrays
    - targets: Dict with 'image_id' and 'annotations' in COCO format
               where annotations have 'bbox' in [x, y, w, h] pixel format
    """
    # Separate images and annotations
    # Expecting batch items to be (images_dict, target) where target has COCO-style annotations

    durations = list(batch[0][0].keys())

    # Process each duration separately
    batched_frames = {}
    processed_labels = None

    for d in durations:
        images = []
        annotations = []

        for frames_dict, target in batch:
            # Get image for this duration
            img = frames_dict[d]
            if isinstance(img, torch.Tensor):
                # Convert tensor to numpy (H, W, C) format for processor
                if img.dim() == 3:
                    img = img.permute(1, 2, 0).numpy()
            # Keep as float32 in [0, 1] range - processor will handle rescaling
            images.append(img)

            # Get COCO-style annotations
            if 'annotations' in target:
                # Already in COCO format
                annotations.append(target)
            else:
                # Convert from normalized [cx, cy, w, h] to COCO [x, y, w, h]
                # Need image dimensions for denormalization
                if isinstance(img, np.ndarray):
                    h, w = img.shape[:2]
                else:
                    h, w = img.shape[-2:]

                coco_anns = []
                boxes = target.get('boxes', torch.zeros((0, 4)))
                labels = target.get('labels', torch.zeros((0,), dtype=torch.int64))

                for box, label in zip(boxes, labels):
                    cx, cy, bw, bh = box.tolist()
                    # Convert normalized cxcywh to pixel xywh
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    box_w = bw * w
                    box_h = bh * h

                    coco_anns.append({
                        'bbox': [x, y, box_w, box_h],
                        'category_id': int(label),
                        'area': box_w * box_h,
                        'iscrowd': 0,
                        'id': len(coco_anns),
                    })

                annotations.append({
                    'image_id': target.get('image_id', 0),
                    'annotations': coco_anns
                })

        # Process images AND annotations together
        # Images are in [0, 1] float range, so no rescaling needed (do_rescale=False)
        # But normalization (ImageNet mean/std) will still be applied
        encoding = image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
            do_rescale=False,  # Images already in [0, 1] range,
        )

        batched_frames[d] = encoding['pixel_values']

        # Labels are the same for all durations (same annotations)
        if processed_labels is None:
            processed_labels = encoding['labels']

    batch = {'frames': batched_frames, 'labels': processed_labels}

    return batch


def collate_fn_no_resize(batch):
    """
    Simple collate function that does NOT resize images.

    Use this when you want to keep original image resolution.
    Targets should have 'boxes' in normalized [cx, cy, w, h] format.
    """
    batched_frames = {}
    targets = []

    durations = list(batch[0][0].keys())
    temp_lists = {d: [] for d in durations}

    for frames, tgt in batch:
        targets.append(tgt)
        for d in durations:
            temp_lists[d].append(frames[d])

    for d in durations:
        batched_frames[d] = torch.stack(temp_lists[d], dim=0)

    return batched_frames, targets

from typing import List, Dict
from torchvision.ops import nms

def _apply_nms(results: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        filtered = []
        for pred in results:
            boxes  = pred['boxes']    # xyxy pixel
            scores = pred['scores']
            labels = pred['labels']

            if boxes.numel() == 0:
                filtered.append(pred)
                continue

            keep = nms(boxes.float(), scores.float(), iou_threshold)
            filtered.append({
                'boxes':  boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep],
            })
        return filtered