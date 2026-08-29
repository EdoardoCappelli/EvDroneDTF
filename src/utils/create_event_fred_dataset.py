import os
import numpy as np
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import json
import concurrent.futures
from functools import partial

# Set HDF5 plugin path before importing h5py
os.environ['HDF5_PLUGIN_PATH'] = os.environ.get('HDF5_PLUGIN_PATH', '') + ':/usr/lib/x86_64-linux-gnu/hdf5/plugins'
import hdf5plugin  # noqa: F401
import h5py

def read_coordinates(file_path):
    """
    Reads drone coordinates. Optimized slightly for cleaner IO.
    """
    coordinates = {}
    if not os.path.exists(file_path):
        return coordinates
    
    with open(file_path, "r") as f:
        for line in f:
            try:
                # Splitting by fixed delimiter is faster
                timestamp_str, rest = line.strip().split(": ", 1)
                parts = rest.split(", ")
                # Parse necessary parts
                x1, y1, x2, y2, idx = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(float(parts[4]))
                class_name = parts[5]
                
                if timestamp_str not in coordinates:
                    coordinates[timestamp_str] = []
                coordinates[timestamp_str].append([x1, y1, x2, y2, idx, class_name])
            except (ValueError, IndexError):
                continue
    return coordinates

# Keep lambda identical to ensure key matching with coord file
to_timestamp = lambda idx: str(float("{:.6f}".format((idx+1)*0.033333)))

def process_single_video_job(job_args):
    """
    Worker function for parallel processing.
    """
    video_folder, accumulation_time_ms = job_args
    
    hdf5_path = os.path.join(video_folder, 'Event', 'events.hdf5')
    coordinate_file = os.path.join(video_folder, 'coordinates.txt')
    video_name = os.path.basename(video_folder.rstrip('/'))
    
    if not os.path.exists(hdf5_path):
        return None

    accumulation_time_us = int(accumulation_time_ms * 1000)

    # 1. READ COORDINATES
    coordinates = read_coordinates(coordinate_file)

    # 2. READ TIMESTAMPS
    # Open HDF5 only for the duration of the read
    with h5py.File(hdf5_path, "r") as f:
        # Load only the timestamps into memory (uint32 is efficient)
        timestamps = f["CD"]["events"]['t'][:]
        
    if len(timestamps) == 0:
        return []

    t_min = timestamps[0]
    t_max = timestamps[-1]

    # 3. VECTORIZED WINDOW CALCULATION
    # Create all window boundaries at once
    window_starts = np.arange(t_min, t_max, accumulation_time_us)
    window_ends = window_starts + accumulation_time_us
    
    # 4. BINARY SEARCH (The huge speedup)
    # Instead of scanning the array N times, we use searchsorted to find split points instantly
    # side='left' means: find first index i such that timestamps[i] >= value
    # This perfectly matches [start, end) logic
    start_indices = np.searchsorted(timestamps, window_starts, side='left')
    end_indices = np.searchsorted(timestamps, window_ends, side='left')
    
    windows = []
    
    # We zip through the calculated boundaries and indices
    # This loop is now pure Python object creation, no heavy array math
    for i, (w_start, w_end, idx_start, idx_end) in enumerate(zip(window_starts, window_ends, start_indices, end_indices)):
        
        num_events = idx_end - idx_start
        
        if num_events > 0:
            timestamp_key = to_timestamp(i)
            has_annotations = timestamp_key in coordinates
            
            # Using int() to ensure JSON serializability
            window_info = {
                'video_id': video_name,
                'hdf5_path': hdf5_path,
                'window_idx': i,
                'start_event_idx': int(idx_start),
                'end_event_idx': int(idx_end),
                'num_events': int(num_events),
                'window_start_time': int(w_start),
                'window_end_time': int(w_end),
                'timestamp_key': timestamp_key,
                'has_annotations': has_annotations,
                'annotations': coordinates[timestamp_key] if has_annotations else []
            }
            windows.append(window_info)
            
    return windows

def preprocess_dataset(dataset_path, split, accumulation_time_ms=33.333, output_dir=None, max_workers=None):
    """
    Main driver using ProcessPoolExecutor for parallelism.
    """
    original_split = split
    limit_at = -1
    if split == 'toy':
        split = 'train'
        limit_at = 100
    
    # Locate videos
    search_path = os.path.join(dataset_path, split, '*', '')
    video_folders = natsorted(glob(search_path))
    
    if limit_at > 0:
        video_folders = video_folders[:limit_at]

    print(f"Found {len(video_folders)} videos in {original_split} split")

    if output_dir is None:
        output_dir = os.path.join(dataset_path, 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    all_windows = []
    video_stats = []

    # Prepare arguments for parallel workers
    job_args = [(folder, accumulation_time_ms) for folder in video_folders]

    # Use ProcessPoolExecutor to utilize all CPU cores
    # We use a context manager to ensure proper cleanup
    print(f"Processing with {os.cpu_count() if max_workers is None else max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map returns results in order
        results = list(tqdm(executor.map(process_single_video_job, job_args), total=len(video_folders), desc="Processing"))

    # Aggregate results
    for res in results:
        if res is None or not res:
            continue
            
        windows = res
        video_id = windows[0]['video_id']
        
        windows_with_ann = sum(1 for w in windows if w['has_annotations'])
        total_events = sum(w['num_events'] for w in windows)
        
        video_stats.append({
            'video_id': video_id,
            'total_windows': len(windows),
            'windows_with_annotations': windows_with_ann,
            'total_events': total_events
        })
        all_windows.extend(windows)

    # Save output
    output_file = os.path.join(output_dir, f'{original_split}_windows_{int(accumulation_time_ms)}ms.json')
    print(f"\nSaving preprocessed indices to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump({
            'split': original_split,
            'accumulation_time_ms': accumulation_time_ms,
            'total_videos': len(video_stats),
            'total_windows': len(all_windows),
            'windows_with_annotations': sum(1 for w in all_windows if w['has_annotations']),
            'video_stats': video_stats,
            'windows': all_windows
        }, f, indent=2)

    return all_windows, output_file

# --- Helper functions for visualization (unchanged logic) ---
def load_preprocessed_indices(index_file):
    with open(index_file, 'r') as f:
        return json.load(f)

def load_event_window(hdf5_path, start_idx, end_idx):
    with h5py.File(hdf5_path, "r") as f:
        events_raw = f["CD"]["events"][start_idx:end_idx]
        events = np.zeros(len(events_raw), dtype=[
            ('x', np.uint16), ('y', np.uint16), ('t', np.uint32), ('p', np.uint8)
        ])
        events['x'] = events_raw['x']
        events['y'] = events_raw['y']
        events['t'] = events_raw['t']
        events['p'] = events_raw['p']
    return events

if __name__ == "__main__":
    # Ensure this block is used to prevent recursive spawning issues on Windows/Linux
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dataset_path = "/home/gmagrini/datasets/FRED_split"
    
    # Run the optimized processing
    all_windows, index_file = preprocess_dataset(
        dataset_path=dataset_path,
        split='test',
        accumulation_time_ms=33.333
    )
    
    # Quick verification code
    print("\nVerifying data...")
    if len(all_windows) > 0:
        w = all_windows[0]
        print(f"Sample window from {w['video_id']}: {w['num_events']} events")