import motmetrics as mm
import os
import pandas as pd
import numpy as np

# Compat NumPy 2.0: motmetrics chiama np.asfarray (rimosso in NumPy 2.0). Shim locale, così
# non serve fare downgrade di numpy (romperebbe l'ABI di torch/pandas nell'ambiente FRED++).
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def load_mot_data(file_path):
    """
    Loads MOTChallenge formatted data from a text file.
    Returns a DataFrame with columns: frameid, id, x, y, w, h
    """
    data = pd.read_csv(
        file_path,
        sep=',',
        names=['frameid', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'confidence', 'x_3d', 'y_3d', 'z_3d'],
        index_col=False
    )
    data = data[['frameid', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].copy()
    data['frameid'] = data['frameid'].astype(int)
    data['id'] = data['id'].astype(int)
    return data

def compute_tracking_metrics(gt_dir, tracks_dir, output_file=None):
    accs = [] # List to store MOTAccumulator objects for each video
    names = [] # List to store names of the videos (for summary)

    print(f"Loading ground truth from: {gt_dir}")
    print(f"Loading tracking results from: {tracks_dir}")

    gt_files_in_dir = {f for f in os.listdir(gt_dir) if f.endswith('.txt')}
    track_files_in_dir = {f for f in os.listdir(tracks_dir) if f.endswith('.txt')}

    common_base_names = gt_files_in_dir.intersection(track_files_in_dir)

    if not common_base_names:
        print("No matching ground truth and tracking files found with the same base names. Please check your directories and naming conventions.")
        return

    mh = mm.metrics.create()
    IOU_THRESHOLD = 0.5 # A common threshold for MOTChallenge evaluation

    # Temporary list to store individual sequence summary DataFrames
    individual_summaries_dfs = [] 

    for base_name in sorted(list(common_base_names)):
        gt_path = os.path.join(gt_dir, base_name)
        tracks_path = os.path.join(tracks_dir, base_name)

        video_name_for_summary = os.path.splitext(base_name)[0]

        print(f"\nProcessing video file: {base_name}")
        try:
            gt_data = load_mot_data(gt_path)
            tracks_data = load_mot_data(tracks_path)
        except Exception as e:
            print(f"Error loading data for {base_name}: {e}. Skipping.")
            continue

        print(len(tracks_data), "tracks loaded")
        print(len(gt_data), "ground truth boxes loaded")
        acc = mm.MOTAccumulator(auto_id=False) # auto_id=False for 1.4.0 when frameid is provided

        all_frames = pd.concat([gt_data['frameid'], tracks_data['frameid']]).unique()
        if len(all_frames) == 0:
            print(f"No frames found in data for {base_name}. Skipping.")
            continue

        min_frame_id = all_frames.min()
        max_frame_id = all_frames.max()

        for frameid in range(min_frame_id, max_frame_id + 1):
            gt_frame = gt_data[gt_data['frameid'] == frameid]
            tracks_frame = tracks_data[tracks_data['frameid'] == frameid]

            gt_boxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
            track_boxes = tracks_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values

            gt_ids = gt_frame['id'].values
            track_ids = tracks_frame['id'].values

            if len(gt_boxes) > 0 and len(track_boxes) > 0:
                distances = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=IOU_THRESHOLD)
            else:
                distances = np.empty((len(gt_boxes), len(track_boxes)))

            acc.update(
                gt_ids,
                track_ids,
                distances,
                frameid=frameid
            )
        
        # Store the accumulator for this sequence (for overall summary later)
        accs.append(acc)
        names.append(video_name_for_summary)

        # --- Compute metrics for THIS INDIVIDUAL SEQUENCE (1.4.0 API) ---
        # mh.compute takes a single accumulator for per-sequence metrics
        seq_summary_df = mh.compute(
            acc, # Pass a single accumulator
            metrics=[ "num_frames", "num_objects", "mostly_tracked", "partially_tracked", "mostly_lost", "num_fragmentations", "mota", "motp", "num_detections", 'num_matches', 'num_switches', "num_false_positives", "num_misses", 'idfp', 'idfn'],
            # No 'names' or 'generate_overall' here
        )
        # Assign the sequence name to the index of this DataFrame
        seq_summary_df.index = pd.Index([video_name_for_summary], name='SequenceName')
        individual_summaries_dfs.append(seq_summary_df)
        print(seq_summary_df)

        print(f"Metrics computed for {base_name}")

    if not accs: # If no accumulators were created due to missing files
        print("No tracking data available to compute metrics.")
        return

    # --- Combine individual summaries into one DataFrame ---
    summary_all_sequences = pd.concat(individual_summaries_dfs)

    print(summary_all_sequences.mean())

    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )


    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


    if output_file:
        print(f"\nSaving metrics summary to: {output_file}")
        summary.to_csv(output_file)

if __name__ == "__main__":
    split = 'challenging'
    modality = 'rgb'
    model = 'L2' # leave blank for yolo ''
    GT_DIR = f"gt_tracks/{split}/"
    if model == 'yolo':
        TRACKS_DIR = f"output_tracks/{modality}_{split}/"
    elif model == 'rtdetr':
        TRACKS_DIR = f"output_tracks/{model}_{modality}_{split}/"
    elif model == 'erdetr':
        TRACKS_DIR = f"output_tracks/erdetr_{modality}_{split}/"
    elif model == 'faster':
        TRACKS_DIR = f"output_tracks/faster_{modality}_{split}/"
    elif model == 'L2':
        TRACKS_DIR = "../drone_detection_fred/output_tracks/multimodal_20251029_182326/"
    elif model == 'JEPA':
        TRACKS_DIR = "../drone_detection_fred/output_tracks/multimodal_20251111_121407/"
    OUTPUT_METRICS_FILE = f"tracking_metrics_summary_{model}_{modality}_{split}.csv"

    compute_tracking_metrics(GT_DIR, TRACKS_DIR, OUTPUT_METRICS_FILE)

    print("\nAll metric computation tasks completed.")