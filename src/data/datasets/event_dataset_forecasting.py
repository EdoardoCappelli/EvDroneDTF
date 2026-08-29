"""
Multi-duration event dataset for forecasting with COCO-style annotations.

Supports:
- Multi-temporal event frame rendering at different durations
- Past and future annotation windows for temporal modeling
- COCO-style annotation format compatible with DetrImageProcessor
"""

import random
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from numba import njit
from PIL import Image

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
    """Render Metavision-style with accumulation."""
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


# ============================================================================
# HDF5 INDEX CACHE
# ============================================================================

class HDF5IndexCache:
    """Cache for HDF5 time indexes and file handles to avoid reloading/reopening for each sample."""

    def __init__(self):
        self._cache = {}
        self._file_handles = {}  # Persistent file handles for faster I/O

    def get_file(self, hdf5_path):
        """Get or create a persistent file handle (SWMR mode for safe multi-read)."""
        if hdf5_path not in self._file_handles:
            self._file_handles[hdf5_path] = h5py.File(hdf5_path, 'r', swmr=True)
        return self._file_handles[hdf5_path]

    def get_index(self, hdf5_path):
        if hdf5_path not in self._cache:
            f = self.get_file(hdf5_path)
            cdindexes = f["CD/indexes"][:]
            offset = int(f['CD/indexes'].attrs.get('offset', 0))
            self._cache[hdf5_path] = {
                'indexes': cdindexes,
                'offset': offset
            }
        return self._cache[hdf5_path]

    def load_events_by_time(self, hdf5_path, t_start, t_end):
        """Load events between t_start and t_end (in microseconds) using the time index."""
        idx_data = self.get_index(hdf5_path)
        cdindexes = idx_data['indexes']
        offset = idx_data['offset']

        idx_pos_start, idx_pos_end = np.searchsorted(
            cdindexes['ts'],
            np.array([t_start, t_end]) + offset,
            side='left'
        ) - 1

        idx_pos_start = max(0, idx_pos_start)
        idx_pos_end = min(len(cdindexes) - 1, idx_pos_end + 1)

        # Use cached file handle instead of opening/closing each time
        f = self.get_file(hdf5_path)
        start_event_idx = int(cdindexes[idx_pos_start][0])
        end_event_idx = int(cdindexes[idx_pos_end][0])
        events = f["CD/events"][start_event_idx:end_event_idx]

        if len(events) == 0:
            return None

        actual_start, actual_end = np.searchsorted(
            events['t'],
            [t_start, t_end],
            side='left'
        )
        events = events[actual_start:actual_end]

        return events

    def close_all(self):
        """Close all open file handles."""
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __del__(self):
        self.close_all()


# Global cache instance
_hdf5_cache = HDF5IndexCache()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_annotations_coco(annotations, width, height):
    """
    Convert raw annotations to COCO format.

    Args:
        annotations: List of [x1, y1, x2, y2, ...] annotations
        width: Frame width for normalization
        height: Frame height for normalization

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

        bbox = [float(x1), float(y1), float(w), float(h)]
        area = float(w * h)

        coco_annotations.append({
            "bbox": bbox,
            "category_id": 0,  # Single class (person/drone)
            "area": area,
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
        ids: List of annotation IDs
    """
    boxes = []
    labels = []
    ids = []

    for ann in annotations:
        x1, y1, x2, y2, id = ann[:5]
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
            ids.append(id)


    return boxes, labels, ids


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
            # Normalize to [0, 255] for grayscale
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

class FREDForecastingDataset(Dataset):
    """
    FRED event dataset for forecasting with COCO-style annotations.

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
        num_future_annotations: Number of future annotation windows to include
    """

    def __init__(
        self,
        index_file: str,
        width: int = 1280,
        height: int = 720,
        duration_ms: int = 33,
        render_mode: str = 'metavision',
        subsample: int = 1,
        num_future_annotations: int = 1,
    ):
        self.width = width
        self.height = height
        self.duration_ms = duration_ms
        self.duration_us = duration_ms * 1000
        self.num_future_annotations = num_future_annotations

        self.renderer = EventFrameRenderer(width, height, render_mode)

        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']

        # Build window lookup for future annotations
        if num_future_annotations > 0:
            self._build_window_lookup()

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

        print(f"Dataset: {len(self.windows)} samples, {duration_ms}ms, future_ann={num_future_annotations}")

    def _build_window_lookup(self):
        """Build lookup structure to find future windows efficiently."""
        self._windows_by_file = {}
        for i, w in enumerate(self.windows):
            path = w['hdf5_path']
            if path not in self._windows_by_file:
                self._windows_by_file[path] = []
            self._windows_by_file[path].append({
                'idx': i,
                'end_time': w['window_end_time'],
                'annotations': w.get('annotations', []) if w.get('has_annotations', False) else [],
                'has_annotations': w.get('has_annotations', False)
            })

        for path in self._windows_by_file:
            self._windows_by_file[path].sort(key=lambda x: x['end_time'])

    def _get_future_windows(self, hdf5_path, current_end_time, num_future):
        """Get future annotation windows."""
        if hdf5_path not in self._windows_by_file:
            return []

        file_windows = self._windows_by_file[hdf5_path]
        end_times = [w['end_time'] for w in file_windows]
        current_idx = np.searchsorted(end_times, current_end_time, side='right') - 1

        future_windows = []
        for i in range(1, num_future + 1):
            future_idx = current_idx + i
            if future_idx < len(file_windows):
                future_windows.append(file_windows[future_idx])
            else:
                break

        return future_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start = t_end - self.duration_us

        # Load and render events
        events = _hdf5_cache.load_events_by_time(window['hdf5_path'], t_start, t_end)

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

        # Build COCO-style target
        coco_annotations = []
        if window['has_annotations']:
            coco_annotations = parse_annotations_coco(
                window['annotations'], self.width, self.height
            )

        target = {
            "image_id": idx,
            "annotations": coco_annotations,
        }

        # Add future annotations if requested
        if self.num_future_annotations > 0:
            future_windows = self._get_future_windows(
                window['hdf5_path'], t_end, self.num_future_annotations
            )

            future_annotations = []
            for fw in future_windows:
                if fw['has_annotations']:
                    fa = parse_annotations_coco(fw['annotations'], self.width, self.height)
                else:
                    fa = []
                future_annotations.append(fa)

            # Pad if needed
            while len(future_annotations) < self.num_future_annotations:
                future_annotations.append([])

            target["future_annotations"] = future_annotations

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
        durations_ms: List of accumulation durations (e.g., [33, 66, 99, 132])
        render_mode: 'delta', 'metavision', 'metavision_acc'
        subsample: Subsample factor
        num_future_annotations: Number of future annotation windows to include
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
        num_future_annotations: int = 0,
    ):
        self.width = width
        self.height = height

        # Handle duration specification
        if base_duration_ms is not None and num_backward_frames is not None:
            durations_ms = [base_duration_ms * (i + 1) for i in range(num_backward_frames)]
        elif durations_ms is None:
            durations_ms = [3, 33, 165, 330]

        self.durations_ms = sorted(durations_ms, reverse=True)
        self.num_future_annotations = num_future_annotations

        self.renderer = EventFrameRenderer(width, height, render_mode)

        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']

        if num_future_annotations > 0:
            self._build_window_lookup()

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

        print(f"Dataset: {len(self.windows)} samples")
        print(f"Durations: {self.durations_ms} ms, future_ann={num_future_annotations}")

        self._duration_us = [int(d * 1000) for d in self.durations_ms]
        self._max_duration_us = max(self._duration_us)

    def _build_window_lookup(self):
        """Build lookup structure for future windows."""
        self._windows_by_file = {}
        for i, w in enumerate(self.windows):
            path = w['hdf5_path']
            if path not in self._windows_by_file:
                self._windows_by_file[path] = []
            self._windows_by_file[path].append({
                'idx': i,
                'end_time': w['window_end_time'],
                'annotations': w.get('annotations', []) if w.get('has_annotations', False) else [],
                'has_annotations': w.get('has_annotations', False)
            })

        for path in self._windows_by_file:
            self._windows_by_file[path].sort(key=lambda x: x['end_time'])

    def _get_future_windows(self, hdf5_path, current_end_time, num_future):
        """Get future annotation windows."""
        if hdf5_path not in self._windows_by_file:
            return []

        file_windows = self._windows_by_file[hdf5_path]
        end_times = [w['end_time'] for w in file_windows]
        current_idx = np.searchsorted(end_times, current_end_time, side='right') - 1

        future_windows = []
        for i in range(1, num_future + 1):
            future_idx = current_idx + i
            if future_idx < len(file_windows):
                future_windows.append(file_windows[future_idx])
            else:
                break

        return future_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start_needed = t_end - self._max_duration_us

        events = _hdf5_cache.load_events_by_time(
            window['hdf5_path'], t_start_needed, t_end
        )

        images = {}

        if events is not None and len(events) > 0:
            t_all = events['t']
            x_all = events['x'].astype(np.int32)
            y_all = events['y'].astype(np.int32)
            p_all = events['p'].astype(np.int32)

            for duration_us in self._duration_us:
                t_start = t_end - duration_us
                mask = t_all >= t_start
                frame = self.renderer.render(x_all[mask], y_all[mask], p_all[mask])
                images[duration_us // 1000] = self.renderer.to_pil(frame)
        else:
            empty_frame = self.renderer.render(
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )
            for d in self.durations_ms:
                images[d] = self.renderer.to_pil(empty_frame)

        # Build COCO-style target
        coco_annotations = []
        if window['has_annotations']:
            coco_annotations = parse_annotations_coco(
                window['annotations'], self.width, self.height
            )

        target = {
            "image_id": idx,
            "annotations": coco_annotations,
        }

        # Add future annotations
        if self.num_future_annotations > 0:
            future_windows = self._get_future_windows(
                window['hdf5_path'], t_end, self.num_future_annotations
            )

            future_annotations = []
            for fw in future_windows:
                if fw['has_annotations']:
                    fa = parse_annotations_coco(fw['annotations'], self.width, self.height)
                else:
                    fa = []
                future_annotations.append(fa)

            while len(future_annotations) < self.num_future_annotations:
                future_annotations.append([])

            target["future_annotations"] = future_annotations

        return images, target


class FREDMultiDurationTensorDatasetTracking(Dataset):
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
        num_past_annotations: int = 0,
        num_future_annotations: int = 12,
    ):
        self.width = width
        self.height = height
        self.num_past_annotations = num_past_annotations
        self.num_future_annotations = num_future_annotations

        # Handle duration specification
        if base_duration_ms is not None and num_backward_frames is not None:
            durations_ms = [base_duration_ms * (i + 1) for i in range(num_backward_frames)]
        elif durations_ms is None:
            durations_ms = [3, 33, 165, 330]

        self.durations_ms = sorted(durations_ms, reverse=True)
        self.renderer = EventFrameRenderer(width, height, render_mode)

        print(f"Loading index from {index_file}...")
        with open(index_file, 'r') as f:
            data = json.load(f)
            self.windows = data['windows']


        if num_past_annotations > 0 or num_future_annotations > 0:
            self._build_window_lookup()

        if subsample > 1:
            original_len = len(self.windows)
            self.windows = self.windows[::subsample]
            print(f"Subsampled: {original_len} -> {len(self.windows)} (1/{subsample})")

        print(f"Dataset: {len(self.windows)} samples")
        print(f"Durations: {self.durations_ms} ms")
        print(f"Past annotations: {num_past_annotations}, Future: {num_future_annotations}")

        self._duration_us = [int(d * 1000) for d in self.durations_ms]
        self._max_duration_us = max(self._duration_us)

    def _build_window_lookup(self):
        """Build lookup structure for past/future windows."""
        self._windows_by_file = {}
        for i, w in enumerate(self.windows):
            path = w['hdf5_path']
            if path not in self._windows_by_file:
                self._windows_by_file[path] = []
            self._windows_by_file[path].append({
                'idx': i,
                'end_time': w['window_end_time'],
                'annotations': w.get('annotations', []) if w.get('has_annotations', False) else [],
                'has_annotations': w.get('has_annotations', False)
            })

        for path in self._windows_by_file:
            self._windows_by_file[path].sort(key=lambda x: x['end_time'])

    def _get_past_windows(self, hdf5_path, current_end_time, num_past):
        """Get past annotation windows (most recent first)."""
        if hdf5_path not in self._windows_by_file:
            return []

        file_windows = self._windows_by_file[hdf5_path]
        end_times = [w['end_time'] for w in file_windows]
        current_idx = np.searchsorted(end_times, current_end_time, side='right') - 1

        past_windows = []
        for i in range(1, num_past + 1):
            past_idx = current_idx - i
            if past_idx >= 0:
                past_windows.append(file_windows[past_idx])
            else:
                break

        return past_windows

    def _get_future_windows(self, hdf5_path, current_end_time, num_future):
        """Get future annotation windows (nearest first)."""
        if hdf5_path not in self._windows_by_file:
            return []

        file_windows = self._windows_by_file[hdf5_path]
        end_times = [w['end_time'] for w in file_windows]
        current_idx = np.searchsorted(end_times, current_end_time, side='right') - 1

        future_windows = []
        for i in range(1, num_future + 1):
            future_idx = current_idx + i
            if future_idx < len(file_windows):
                future_windows.append(file_windows[future_idx])
            else:
                break

        return future_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        t_end = window['window_end_time']
        t_start_needed = t_end - self._max_duration_us

        events = _hdf5_cache.load_events_by_time(
            window['hdf5_path'], t_start_needed, t_end
        )

        frames = {}

        if events is not None and len(events) > 0:
            t_all = events['t']
            x_all = events['x'].astype(np.int32)
            y_all = events['y'].astype(np.int32)
            p_all = events['p'].astype(np.int32)

            for duration_us in self._duration_us:
                t_start = t_end - duration_us
                mask = t_all >= t_start
                frame = self.renderer.render(x_all[mask], y_all[mask], p_all[mask])
                frames[duration_us // 1000] = self.renderer.to_tensor(frame)
        else:
            for d in self.durations_ms:
                if self.renderer.render_mode == 'delta':
                    frames[d] = torch.zeros(1, self.height, self.width, dtype=torch.float32)
                else:
                    frames[d] = torch.zeros(3, self.height, self.width, dtype=torch.float32)

        # Build target with normalized boxes
        boxes, labels, ids = [], [], []
        if window['has_annotations']:
            boxes, labels, ids = parse_annotations_normalized(
                window['annotations'], self.width, self.height
            )

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'ids': torch.tensor(ids, dtype=torch.int64) if ids else torch.zeros((0,), dtype=torch.int64),
        }

        # Add past annotations
        if self.num_past_annotations > 0:
            past_windows = self._get_past_windows(
                window['hdf5_path'], t_end, self.num_past_annotations
            )

            past_boxes_list, past_labels_list, past_ids_list, past_times = [], [], [], []
            for pw in past_windows:
                if pw['has_annotations']:
                    pb, pl, pid = parse_annotations_normalized(pw['annotations'], self.width, self.height)
                else:
                    pb, pl, pid = [], [], []

                past_boxes_list.append(
                    torch.tensor(pb, dtype=torch.float32) if pb else torch.zeros((0, 4), dtype=torch.float32)
                )
                past_labels_list.append(
                    torch.tensor(pl, dtype=torch.int64) if pl else torch.zeros((0,), dtype=torch.int64)
                )
                past_ids_list.append(
                    torch.tensor(pid, dtype=torch.int64) if pid else torch.zeros((0,), dtype=torch.int64)
                )

                past_times.append(pw['end_time'] - t_end)

            # Pad if needed
            while len(past_boxes_list) < self.num_past_annotations:
                past_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
                past_labels_list.append(torch.zeros((0,), dtype=torch.int64))
                past_ids_list.append(torch.zeros((0,), dtype=torch.int64))
                past_times.append(0)

            target['past_boxes'] = past_boxes_list
            target['past_labels'] = past_labels_list
            target['past_ids'] = past_ids_list
            target['past_time_deltas'] = torch.tensor(past_times, dtype=torch.int64)

        # Add future annotations
        if self.num_future_annotations > 0:
            future_windows = self._get_future_windows(
                window['hdf5_path'], t_end, self.num_future_annotations
            )

            future_boxes_list, future_labels_list, future_ids_list, future_times = [], [], [], []
            for fw in future_windows:
                if fw['has_annotations']:
                    fb, fl, fid = parse_annotations_normalized(fw['annotations'], self.width, self.height)
                else:
                    fb, fl, fid = [], [], []

                future_boxes_list.append(
                    torch.tensor(fb, dtype=torch.float32) if fb else torch.zeros((0, 4), dtype=torch.float32)
                )
                future_labels_list.append(
                    torch.tensor(fl, dtype=torch.int64) if fl else torch.zeros((0,), dtype=torch.int64)
                )
                future_ids_list.append(
                    torch.tensor(fid, dtype=torch.int64) if fid else torch.zeros((0,), dtype=torch.int64)
                )
                future_times.append(fw['end_time'] - t_end)

            # Pad if needed
            while len(future_boxes_list) < self.num_future_annotations:
                future_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
                future_labels_list.append(torch.zeros((0,), dtype=torch.int64))
                future_ids_list.append(torch.zeros((0,), dtype=torch.int64))
                future_times.append(0)
                

            target['future_boxes'] = future_boxes_list
            target['future_labels'] = future_labels_list
            target['future_time_deltas'] = torch.tensor(future_times, dtype=torch.int64)
            target['future_ids'] = future_ids_list

        return frames, target


"""
Visualizer for FRED dataset with past, present, and future boxes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse


def denormalize_box(box, width, height):
    """Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return x1, y1, x2, y2


def draw_boxes(ax, boxes, color, label, alpha=1.0, linestyle='-'):
    """Draw boxes on axis."""
    if len(boxes) == 0:
        return

    width, height = 1280, 720  # Default FRED resolution

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = denormalize_box(box, width, height)
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=alpha,
            linestyle=linestyle,
            label=label if i == 0 else None
        )
        ax.add_patch(rect)


def visualize_sample(frames, target, duration_key=None, save_path=None):
    """
    Visualize a single sample with past, present, and future boxes.

    Args:
        frames: Dict of duration_ms -> tensor (C, H, W)
        target: Dict with boxes, past_boxes, future_boxes
        duration_key: Which duration to display (default: first available)
        save_path: Path to save figure (optional)
    """
    # Get frame to display
    if duration_key is None:
        duration_key = list(frames.keys())[0]

    frame = frames[duration_key]

    # Convert tensor to displayable format
    if frame.shape[0] == 1:  # Grayscale
        img = frame.squeeze(0).numpy()
        cmap = 'gray'
    else:  # RGB
        img = frame.permute(1, 2, 0).numpy()
        cmap = None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)

    # Draw present boxes (green, solid)
    if 'boxes' in target and len(target['boxes']) > 0:
        draw_boxes(ax, target['boxes'], 'lime', 'Present', alpha=1.0, linestyle='-')

    # Draw past boxes (blue, dashed) - older = more transparent
    if 'past_boxes' in target:
        num_past = len(target['past_boxes'])
        for i, past_boxes in enumerate(target['past_boxes']):
            if len(past_boxes) > 0:
                alpha = 0.3 + 0.5 * (num_past - i) / num_past
                label = f'Past t-{i+1}' if i == 0 else None
                draw_boxes(ax, past_boxes, 'cyan', label, alpha=alpha, linestyle='--')

    # Draw future boxes (red/orange, dotted) - farther = more transparent
    if 'future_boxes' in target:
        num_future = len(target['future_boxes'])
        for i, future_boxes in enumerate(target['future_boxes']):
            if len(future_boxes) > 0:
                alpha = 0.3 + 0.5 * (num_future - i) / num_future
                # Gradient from yellow (near) to red (far)
                r = 1.0
                g = max(0, 1.0 - i / num_future)
                color = (r, g, 0)
                label = f'Future t+{i+1}' if i == 0 else None
                draw_boxes(ax, future_boxes, color, label, alpha=alpha, linestyle=':')

    ax.set_title(f'FRED Sample - Duration: {duration_key}ms')
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_temporal_sequence(frames, target, save_path=None):
    """
    Visualize temporal sequence: past -> present -> future.

    Creates a multi-panel figure showing the trajectory.
    """
    width, height = 1280, 720

    # Get frame
    duration_key = list(frames.keys())[0]
    frame = frames[duration_key]

    if frame.shape[0] == 1:
        img = frame.squeeze(0).numpy()
        cmap = 'gray'
    else:
        img = frame.permute(1, 2, 0).numpy()
        cmap = None

    # Count panels
    num_past = len(target.get('past_boxes', []))
    num_future = len(target.get('future_boxes', []))
    total_panels = num_past + 1 + num_future

    # Create figure
    fig, axes = plt.subplots(1, total_panels, figsize=(4 * total_panels, 4))
    if total_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Past panels
    for i in range(num_past - 1, -1, -1):  # Reverse order (oldest first)
        ax = axes[panel_idx]
        if cmap:
            ax.imshow(img, cmap=cmap, alpha=0.5)
        else:
            ax.imshow(img * 0.5)  # Dim background

        past_boxes = target['past_boxes'][i]
        if len(past_boxes) > 0:
            draw_boxes(ax, past_boxes, 'cyan', None, alpha=1.0, linestyle='-')

        ax.set_title(f't-{i+1}')
        ax.axis('off')
        panel_idx += 1

    # Present panel
    ax = axes[panel_idx]
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)

    if 'boxes' in target and len(target['boxes']) > 0:
        draw_boxes(ax, target['boxes'], 'lime', None, alpha=1.0, linestyle='-')

    ax.set_title('Present (t=0)')
    ax.axis('off')
    panel_idx += 1

    # Future panels
    for i in range(num_future):
        ax = axes[panel_idx]
        if cmap:
            ax.imshow(img, cmap=cmap, alpha=0.5)
        else:
            ax.imshow(img * 0.5)

        future_boxes = target['future_boxes'][i]
        if len(future_boxes) > 0:
            r = 1.0
            g = max(0, 1.0 - i / max(num_future, 1))
            draw_boxes(ax, future_boxes, (r, g, 0), None, alpha=1.0, linestyle='-')

        ax.set_title(f't+{i+1}')
        ax.axis('off')
        panel_idx += 1

    plt.suptitle('Temporal Sequence: Past → Present → Future')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_trajectory(frames, target, save_path=None):
    """
    Visualize object trajectories across time on a single frame.

    Draws lines connecting box centers through time.
    """
    width, height = 1280, 720

    # Get frame
    duration_key = list(frames.keys())[0]
    frame = frames[duration_key]

    if frame.shape[0] == 1:
        img = frame.squeeze(0).numpy()
        cmap = 'gray'
    else:
        img = frame.permute(1, 2, 0).numpy()
        cmap = None

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)

    # Collect all box centers through time
    def get_centers(boxes):
        if len(boxes) == 0:
            return []
        centers = []
        for box in boxes:
            cx, cy = box[0] * width, box[1] * height
            centers.append((cx, cy))
        return centers

    # Past trajectory (blue)
    if 'past_boxes' in target:
        for i, past_boxes in enumerate(reversed(target['past_boxes'])):
            centers = get_centers(past_boxes)
            for cx, cy in centers:
                alpha = 0.3 + 0.5 * (i + 1) / len(target['past_boxes'])
                ax.plot(cx, cy, 'o', color='cyan', markersize=8, alpha=alpha)

    # Present (green)
    if 'boxes' in target:
        centers = get_centers(target['boxes'])
        for cx, cy in centers:
            ax.plot(cx, cy, 'o', color='lime', markersize=12)

    # Future trajectory (red/orange)
    if 'future_boxes' in target:
        for i, future_boxes in enumerate(target['future_boxes']):
            centers = get_centers(future_boxes)
            for cx, cy in centers:
                alpha = 0.3 + 0.5 * (len(target['future_boxes']) - i) / len(target['future_boxes'])
                r = 1.0
                g = max(0, 1.0 - i / max(len(target['future_boxes']), 1))
                ax.plot(cx, cy, 'o', color=(r, g, 0), markersize=8, alpha=alpha)

    # Draw present boxes
    if 'boxes' in target and len(target['boxes']) > 0:
        draw_boxes(ax, target['boxes'], 'lime', 'Present', alpha=0.5, linestyle='-')

    ax.set_title('Object Trajectories (Past=Cyan, Present=Green, Future=Yellow→Red)')
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_dataset_sample(dataset, idx, mode='single', save_path=None):
    """
    Visualize a sample from the dataset.

    Args:
        dataset: FREDMultiDurationTensorDatasetTracking instance
        idx: Sample index
        mode: 'single', 'sequence', or 'trajectory'
        save_path: Path to save figure
    """
    frames, target = dataset[idx]

    if mode == 'single':
        visualize_sample(frames, target, save_path=save_path)
    elif mode == 'sequence':
        visualize_temporal_sequence(frames, target, save_path=save_path)
    elif mode == 'trajectory':
        visualize_trajectory(frames, target, save_path=save_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Quick test with dummy data
if __name__ == "__main__":
    print("Creating dummy data for visualization test...")

    # Load your real dataset
    dataset = FREDMultiDurationTensorDatasetTracking(
        index_file='/home/gmagrini/datasets/FRED_split/preprocessed/test_windows_33ms.json',
        num_past_annotations=5,
        num_future_annotations=12,
    )

    # Visualize a real sample
    frames, target = dataset[random.randint(0, len(dataset) - 1)]
    visualize_sample(frames, target, save_path='real_sample.png')
