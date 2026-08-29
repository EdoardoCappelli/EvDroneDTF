"""
Multi-duration event dataset for FRED with COCO-style annotations.

Supports:
- Multi-temporal event frame rendering at different durations
- COCO-style annotation format compatible with DetrImageProcessor
- Both PIL Image and Tensor output formats
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from numba import njit
from PIL import Image
import threading

os.environ['HDF5_PLUGIN_PATH'] = os.environ.get('HDF5_PLUGIN_PATH', '') + ':/usr/lib/x86_64-linux-gnu/hdf5/plugins'
import hdf5plugin  # noqa: F401
import h5py


# ============================================================================
# NUMBA RENDERING FUNCTIONS
# ============================================================================

@njit(cache=True)
def render_delta_numba(x, y, p, height, width):
    """Render delta frame (positive - negative events)."""
    frame = np.zeros((height, width), dtype=np.float32)
    n = len(x)
    for i in range(n):
        xi, yi, pi = x[i], y[i], p[i]
        if 0 <= xi < width and 0 <= yi < height:
            frame[yi, xi] += 1.0 if pi == 1 else -1.0
    return frame


@njit(cache=True)
def render_metavision_blue_white_numba(x, y, p, height, width):
    """Render Metavision-style: Blue (ON) and White (OFF) on black background."""
    frame = np.zeros((height, width, 3), dtype=np.float32)
    n = len(x)
    for i in range(n):
        xi, yi, pi = x[i], y[i], p[i]
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                frame[yi, xi, 0] = 0.0
                frame[yi, xi, 1] = 0.5
                frame[yi, xi, 2] = 1.0
            else:
                frame[yi, xi, 0] = 1.0
                frame[yi, xi, 1] = 1.0
                frame[yi, xi, 2] = 1.0
    return frame


@njit(cache=True)
def render_metavision_accumulate_numba(x, y, p, height, width):
    """Render Metavision-style with accumulation (brighter = more events)."""
    on_counts = np.zeros((height, width), dtype=np.float32)
    off_counts = np.zeros((height, width), dtype=np.float32)

    n = len(x)
    for i in range(n):
        xi, yi, pi = x[i], y[i], p[i]
        if 0 <= xi < width and 0 <= yi < height:
            if pi == 1:
                on_counts[yi, xi] += 1.0
            else:
                off_counts[yi, xi] += 1.0

    on_max = on_counts.max()
    off_max = off_counts.max()

    if on_max > 0:
        on_counts = on_counts / on_max
    if off_max > 0:
        off_counts = off_counts / off_max

    frame = np.zeros((height, width, 3), dtype=np.float32)
    for yi in range(height):
        for xi in range(width):
            on_val = on_counts[yi, xi]
            off_val = off_counts[yi, xi]
            if on_val > 0 or off_val > 0:
                total = on_val + off_val
                on_ratio = on_val / total if total > 0 else 0
                off_ratio = off_val / total if total > 0 else 0
                intensity = min(1.0, total)
                frame[yi, xi, 2] = intensity * (on_ratio * 1.0 + off_ratio * 1.0)
                frame[yi, xi, 1] = intensity * (on_ratio * 0.5 + off_ratio * 1.0)
                frame[yi, xi, 0] = intensity * (on_ratio * 0.0 + off_ratio * 1.0)

    return frame


@njit(cache=True)
def binary_search_start_idx(t, t_start):
    """Binary search to find first index where t[i] >= t_start."""
    n = len(t)
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if t[mid] < t_start:
            left = mid + 1
        else:
            right = mid
    return left


# ============================================================================
# HDF5 FILE POOL
# ============================================================================

class HDF5FilePool:
    """Thread-safe pool of open HDF5 files with cached indexes."""

    def __init__(self, max_open_files=32):
        self._files = {}
        self._indexes = {}
        self._lock = threading.Lock()
        self._file_locks = {}
        self.max_open_files = max_open_files

    def _get_file_lock(self, path):
        with self._lock:
            if path not in self._file_locks:
                self._file_locks[path] = threading.Lock()
            return self._file_locks[path]

    def get_file_and_index(self, hdf5_path):
        """Get open file handle and cached index."""
        file_lock = self._get_file_lock(hdf5_path)

        with file_lock:
            if hdf5_path not in self._files:
                if len(self._files) >= self.max_open_files:
                    oldest_path = next(iter(self._files))
                    self._files[oldest_path].close()
                    del self._files[oldest_path]
                    del self._indexes[oldest_path]

                f = h5py.File(hdf5_path, 'r', swmr=True)
                self._files[hdf5_path] = f

                cdindexes = f["CD/indexes"][:]
                offset = int(f['CD/indexes'].attrs.get('offset', 0))
                self._indexes[hdf5_path] = {
                    'indexes': cdindexes,
                    'ts': cdindexes['ts'],
                    'idx': cdindexes['id'] if 'id' in cdindexes.dtype.names else cdindexes['index'],
                    'offset': offset
                }

            return self._files[hdf5_path], self._indexes[hdf5_path]

    def load_events(self, hdf5_path, t_start, t_end):
        """Load events for a time range using cached index."""
        f, idx_data = self.get_file_and_index(hdf5_path)

        ts_array = idx_data['ts']
        offset = idx_data['offset']

        search_times = np.array([t_start, t_end]) + offset
        idx_pos_start, idx_pos_end = np.searchsorted(ts_array, search_times, side='left')
        idx_pos_start = max(0, idx_pos_start - 1)
        idx_pos_end = min(len(ts_array) - 1, idx_pos_end)

        idx_col = idx_data['idx']
        start_event_idx = int(idx_col[idx_pos_start])
        end_event_idx = int(idx_col[idx_pos_end])

        if start_event_idx >= end_event_idx:
            return None

        events = f["CD/events"][start_event_idx:end_event_idx]

        if len(events) == 0:
            return None

        t = events['t']
        start_idx = np.searchsorted(t, t_start, side='left')
        end_idx = np.searchsorted(t, t_end, side='right')

        return events[start_idx:end_idx]

    def close_all(self):
        """Close all open files."""
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files.clear()
            self._indexes.clear()


# Global file pool
_file_pool = HDF5FilePool()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_annotations_coco(annotations):
    """
    Convert raw annotations to COCO format [x, y, width, height].

    Args:
        annotations: List of [x1, y1, x2, y2, ...] annotations

    Returns:
        List of COCO-formatted annotation dicts
    """
    coco_annotations = []
    for ann in annotations:
        x1, y1, x2, y2 = ann[:4]
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        coco_annotations.append({
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "category_id": 0,
            "area": float(w * h),
            "iscrowd": 0,
            "id": len(coco_annotations),
        })

    return coco_annotations


def parse_annotations_normalized(annotations, width, height):
    """
    Convert raw annotations to normalized [cx, cy, w, h] format.

    Args:
        annotations: List of [x1, y1, x2, y2, ...] annotations
        width: Frame width for normalization
        height: Frame height for normalization

    Returns:
        boxes: List of [cx, cy, w, h] normalized boxes
        labels: List of class labels
    """
    boxes = []
    labels = []

    for ann in annotations:
        x1, y1, x2, y2 = ann[:4]
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.001, min(1.0, w))
        h = max(0.001, min(1.0, h))

        if w > 0.001 and h > 0.001:
            boxes.append([cx, cy, w, h])
            labels.append(0)

    return boxes, labels


# ============================================================================
# FRAME RENDERER
# ============================================================================

class EventFrameRenderer:
    """Renders event data into frames with different modes."""

    def __init__(self, width, height, render_mode='metavision'):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_channels = 1 if render_mode == 'delta' else 3

    def render(self, x, y, p):
        """Render events to frame based on render_mode."""
        if len(x) == 0:
            if self.render_mode == 'delta':
                return np.zeros((self.height, self.width), dtype=np.float32)
            else:
                return np.zeros((self.height, self.width, 3), dtype=np.float32)

        if self.render_mode == 'delta':
            frame = render_delta_numba(x, y, p, self.height, self.width)
            abs_max = max(abs(frame.min()), abs(frame.max()), 1.0)
            frame = frame / abs_max
            return frame
        elif self.render_mode == 'metavision':
            return render_metavision_blue_white_numba(x, y, p, self.height, self.width)
        elif self.render_mode == 'metavision_acc':
            return render_metavision_accumulate_numba(x, y, p, self.height, self.width)
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def to_pil(self, frame):
        """Convert rendered frame to PIL Image."""
        if self.render_mode == 'delta':
            frame_uint8 = ((frame + 1) * 127.5).astype(np.uint8)
            return Image.fromarray(frame_uint8, mode='L').convert('RGB')
        else:
            frame_uint8 = (frame * 255).astype(np.uint8)
            return Image.fromarray(frame_uint8, mode='RGB')

    def to_tensor(self, frame):
        """Convert rendered frame to PyTorch tensor (C, H, W)."""
        if self.render_mode == 'delta':
            return torch.from_numpy(frame).unsqueeze(0)
        else:
            return torch.from_numpy(frame).permute(2, 0, 1)


# ============================================================================
# DATASETS
# ============================================================================

class FREDDataset(Dataset):
    """
    FRED event dataset with COCO-style annotations.

    Returns (image, target) where:
    - image: PIL Image of the event frame
    - target: Dict with 'image_id' and 'annotations' in COCO format

    Args:
        index_file: Path to preprocessed JSON index
        width: Frame width (default 1280)
        height: Frame height (default 720)
        duration_ms: Single duration for frame accumulation (default 33)
        render_mode: 'delta', 'metavision', 'metavision_acc'
        subsample: Subsample factor (use every Nth sample)
    """

    def __init__(
        self,
        index_file: str,
        width: int = 1280,
        height: int = 720,
        duration_ms: int = 33,
        render_mode: str = 'metavision',
        subsample: int = 1,
    ):
        self.width = width
        self.height = height
        self.duration_ms = duration_ms
        self.duration_us = duration_ms * 1000

        self.renderer = EventFrameRenderer(width, height, render_mode)

        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

        print(f"FREDDataset: {len(self.windows)} samples, {duration_ms}ms duration")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start = t_end - self.duration_us

        events = _file_pool.load_events(window['hdf5_path'], t_start, t_end)

        if events is not None and len(events) > 0:
            x = events['x'].astype(np.int32)
            y = events['y'].astype(np.int32)
            p = events['p'].astype(np.int32)
            frame = self.renderer.render(x, y, p)
        else:
            frame = self.renderer.render(
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )

        image = self.renderer.to_pil(frame)

        coco_annotations = []
        if window['has_annotations']:
            coco_annotations = parse_annotations_coco(window['annotations'])

        target = {"image_id": idx, "annotations": coco_annotations}

        return image, target


class FREDMultiDurationDataset(Dataset):
    """
    Multi-duration FRED dataset with COCO-style annotations.

    Returns (images_dict, target) where:
    - images_dict: Dict mapping duration_ms -> PIL Image
    - target: Dict with 'image_id' and 'annotations' in COCO format

    Args:
        index_file: Path to preprocessed JSON index
        width: Frame width (default 1280)
        height: Frame height (default 720)
        durations_ms: List of accumulation durations (e.g., [33, 22, 11, 3])
        render_mode: 'delta', 'metavision', 'metavision_acc'
        subsample: Subsample factor
        base_duration_ms: Alternative duration specification (base * i)
        num_backward_frames: Number of frames for base_duration_ms mode
    """

    def __init__(
        self,
        index_file: str,
        width: int = 1280,
        height: int = 720,
        durations_ms: list = None,
        render_mode: str = 'metavision',
        subsample: int = 1,
        base_duration_ms: int = None,
        num_backward_frames: int = None,
        num_workers_hint: int = 0,
    ):
        self.width = width
        self.height = height

        if base_duration_ms is not None and num_backward_frames is not None:
            durations_ms = [base_duration_ms * (i + 1) for i in range(num_backward_frames)]
        elif durations_ms is None:
            durations_ms = [33, 22, 11, 3]

        self.durations_ms = sorted(durations_ms, reverse=True)
        self.renderer = EventFrameRenderer(width, height, render_mode)

        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

        print(f"FREDMultiDurationDataset: {len(self.windows)} samples")
        print(f"Durations: {self.durations_ms} ms")

        self._duration_us = [int(d * 1000) for d in self.durations_ms]
        self._max_duration_us = max(self._duration_us)

        if num_workers_hint > 0:
            _file_pool.max_open_files = max(32, num_workers_hint * 4)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start_needed = t_end - self._max_duration_us

        events = _file_pool.load_events(window['hdf5_path'], t_start_needed, t_end)

        images = {}

        if events is not None and len(events) > 0:
            t_all = events['t'].astype(np.int64)
            x_all = events['x'].astype(np.int32)
            y_all = events['y'].astype(np.int32)
            p_all = events['p'].astype(np.int32)

            for duration_us in self._duration_us:
                t_start = t_end - duration_us
                start_idx = binary_search_start_idx(t_all, t_start)
                frame = self.renderer.render(
                    x_all[start_idx:], y_all[start_idx:], p_all[start_idx:]
                )
                images[duration_us // 1000] = self.renderer.to_pil(frame)
        else:
            empty_frame = self.renderer.render(
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )
            for d in self.durations_ms:
                images[d] = self.renderer.to_pil(empty_frame)

        coco_annotations = []
        if window['has_annotations']:
            coco_annotations = parse_annotations_coco(window['annotations'])

        target = {"image_id": idx, "annotations": coco_annotations}

        return images, target


class FREDMultiDurationTensorDataset(Dataset):
    """
    Multi-duration FRED dataset returning tensors instead of PIL images.

    Returns (frames_dict, target) where:
    - frames_dict: Dict mapping duration_ms -> Tensor (C, H, W)
    - target: Dict with 'boxes' and 'labels' in normalized format

    This is compatible with the original training pipeline.
    """

    def __init__(
        self,
        index_file: str,
        width: int = 1280,
        height: int = 720,
        durations_ms: list = None,
        render_mode: str = 'metavision',
        subsample: int = 1,
        base_duration_ms: int = None,
        num_backward_frames: int = None,
        num_workers_hint: int = 0,
        use_only_annotated: bool = False,
        augmentation=None,
    ):
        self.width = width
        self.height = height

        if base_duration_ms is not None and num_backward_frames is not None:
            durations_ms = [base_duration_ms * (i + 1) for i in range(num_backward_frames)]
        elif durations_ms is None:
            durations_ms = [33, 22, 11, 3]

        self.durations_ms = sorted(durations_ms, reverse=True)
        self.renderer = EventFrameRenderer(width, height, render_mode)

        self.use_only_annotated = use_only_annotated
        
        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

         
        if self.use_only_annotated:
            print(f"Filtering to only annotated windows")
            original_len = len(self.windows)
            self.windows = [w for w in self.windows if w.get('has_annotations', False)]
            print(f"Filtered annotated only: {original_len} -> {len(self.windows)} windows")
        else:
            print(f"Using all windows (annotated and non-annotated): {len(self.windows)} windows")
        

        print(f"FREDMultiDurationTensorDataset: {len(self.windows)} samples")
        print(f"Durations: {self.durations_ms} ms")

        self._duration_us = np.array([d * 1000 for d in self.durations_ms], dtype=np.int64)
        self._max_duration_us = int(self._duration_us.max())

        if num_workers_hint > 0:
            _file_pool.max_open_files = max(32, num_workers_hint * 4)

        self.augmentation = augmentation

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start_needed = t_end - self._max_duration_us

        events = _file_pool.load_events(window['hdf5_path'], t_start_needed, t_end)

        if self.augmentation is not None and events is not None and len(events) > 0:
            events, annotations = self.augmentation(events, annotations, t_end, _file_pool)


        frames = {}

        if events is not None and len(events) > 0:
            t_all = events['t'].astype(np.int64)
            x_all = events['x'].astype(np.int32)
            y_all = events['y'].astype(np.int32)
            p_all = events['p'].astype(np.int32)

            for duration_ms, duration_us in zip(self.durations_ms, self._duration_us):
                t_start = t_end - duration_us
                # t_start = max(t_end - duration_us, window['window_start_time']) # Edo 
                start_idx = binary_search_start_idx(t_all, t_start)
                frame = self.renderer.render(
                    x_all[start_idx:], y_all[start_idx:], p_all[start_idx:]
                )
                frames[duration_ms] = self.renderer.to_tensor(frame)
        else:
            for d in self.durations_ms:
                if self.renderer.render_mode == 'delta':
                    frames[d] = torch.zeros(1, self.height, self.width, dtype=torch.float32)
                else:
                    frames[d] = torch.zeros(3, self.height, self.width, dtype=torch.float32)

        boxes, labels = [], []
        if window['has_annotations']:
            boxes, labels = parse_annotations_normalized(
                window['annotations'], self.width, self.height
            )

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }

        return frames, target


def collate_fn(batch):
    """Collate function for multi-duration tensor datasets."""
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


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Alias for backward compatibility with existing code
MultiDurationEventDataset = FREDMultiDurationTensorDataset

