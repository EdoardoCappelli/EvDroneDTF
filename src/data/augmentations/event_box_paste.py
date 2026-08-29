import random
from collections import defaultdict

import numpy as np


class EventBoxPaste:
    """
    Trajectory-aware copy-paste augmentation for event cameras.

    At init, builds a per-track trajectory index from the window list:
        (hdf5_path, track_id) -> [(t_end, annotation), ...] sorted by time.

    At call time:
      1. Picks a random source track from a random window.
      2. Retrieves all trajectory points for that track within the
         max_duration_us window ending at t_end_B.
      3. For each 33ms time slice [t_{k-1}, t_k], crops events whose
         (x, y) fall inside the object's box at t_k.
      4. Applies a single rigid spatial offset so the object's final
         position lands at the chosen paste location.
      5. Remaps all pasted timestamps into sample A's temporal window,
         preserving relative order.

    Because augmentation happens in the raw event domain, the result is
    valid for any downstream representation (frames, voxels, time surfaces).
    """

    def __init__(self, windows, sensor_size, max_duration_us, p=0.5, min_events=30):
        """
        Args:
            windows: list of window dicts (same list used by the dataset).
            sensor_size: (W, H) tuple.
            max_duration_us: longest accumulation window in microseconds.
            p: probability of applying the augmentation per sample.
            min_events: minimum total pasted events required to proceed.
        """
        self.windows = windows
        self.W, self.H = sensor_size
        self.max_duration_us = max_duration_us
        self.p = p
        self.min_events = min_events

        self._annotated_windows = [w for w in windows if w.get("has_annotations", False) and w.get("annotations")]
        self._traj_index = self._build_trajectory_index(windows)
        n_tracks = len(self._traj_index)
        print(f"EventBoxPaste: trajectory index built — {n_tracks} tracks, {len(self._annotated_windows)} annotated source windows")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trajectory_index(windows):
        index = defaultdict(list)
        for w in windows:
            if not w.get("has_annotations", False):
                continue
            t_end = w["window_end_time"]
            hdf5 = w["hdf5_path"]
            for ann in w["annotations"]:
                if len(ann) < 6:
                    continue
                track_id = ann[5]
                index[(hdf5, track_id)].append((t_end, ann))
        for key in index:
            index[key].sort(key=lambda x: x[0])
        return dict(index)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def __call__(self, events_A, annotations_A, t_end_A, file_pool):
        """
        Args:
            events_A: structured numpy array with fields x, y, t, p.
            annotations_A: list of [x1, y1, x2, y2, ...] pixel-coord annotations.
            t_end_A: end timestamp of sample A (microseconds).
            file_pool: HDF5FilePool from the calling dataset.

        Returns:
            (events_aug, annotations_aug) — same types as inputs.
        """
        if random.random() > self.p:
            return events_A, annotations_A

        if events_A is None or len(events_A) == 0:
            return events_A, annotations_A

        # --- pick source window and track --------------------------------
        window_B = random.choice(self._annotated_windows)
        ann_B = random.choice(window_B["annotations"])
        if len(ann_B) < 6:
            return events_A, annotations_A

        t_end_B = window_B["window_end_time"]
        hdf5_B = window_B["hdf5_path"]
        track_id = ann_B[5]

        traj_key = (hdf5_B, track_id)
        if traj_key not in self._traj_index:
            return events_A, annotations_A

        # --- collect trajectory within max_duration_us window ----------
        t_window_start = t_end_B - self.max_duration_us
        traj = [
            (t, ann)
            for t, ann in self._traj_index[traj_key]
            if t_window_start <= t <= t_end_B
        ]
        if not traj:
            return events_A, annotations_A

        # --- final box defines paste size and rigid offset --------------
        _, ann_final = traj[-1]
        x1_f = float(ann_final[0])
        y1_f = float(ann_final[1])
        x2_f = float(ann_final[2])
        y2_f = float(ann_final[3])
        w_box = int(x2_f - x1_f)
        h_box = int(y2_f - y1_f)

        if w_box <= 0 or h_box <= 0 or w_box >= self.W or h_box >= self.H:
            return events_A, annotations_A

        new_x1 = random.randint(0, self.W - w_box)
        new_y1 = random.randint(0, self.H - h_box)
        new_x2 = new_x1 + w_box
        new_y2 = new_y1 + h_box

        dx = float(new_x1) - x1_f
        dy = float(new_y1) - y1_f

        # --- load full event stream for B once --------------------------
        events_B = file_pool.load_events(hdf5_B, t_window_start, t_end_B)
        if events_B is None or len(events_B) == 0:
            return events_A, annotations_A

        xB = events_B["x"].astype(np.float32)
        yB = events_B["y"].astype(np.float32)
        tB = events_B["t"].astype(np.int64)
        obj_dtype = events_A.dtype

        # --- crop per time slice using the box at the slice's end -------
        # slice k: [t_boundaries[k], t_boundaries[k+1]] using ann at traj[k]
        t_boundaries = [t_window_start] + [t for t, _ in traj]

        chunks = []
        for k, (t_ann, ann_k) in enumerate(traj):
            t_lo = t_boundaries[k]
            t_hi = t_boundaries[k + 1]

            x1_k = float(ann_k[0])
            y1_k = float(ann_k[1])
            x2_k = float(ann_k[2])
            y2_k = float(ann_k[3])

            mask = (
                (tB >= t_lo) & (tB < t_hi)
                & (xB >= x1_k) & (xB < x2_k)
                & (yB >= y1_k) & (yB < y2_k)
            )
            if not mask.any():
                continue

            new_x = events_B["x"][mask].astype(np.float64) + dx
            new_y = events_B["y"][mask].astype(np.float64) + dy
            in_bounds = (new_x >= 0) & (new_x < self.W) & (new_y >= 0) & (new_y < self.H)
            if not in_bounds.any():
                continue

            n = int(in_bounds.sum())
            chunk = np.empty(n, dtype=obj_dtype)
            chunk["x"] = new_x[in_bounds].astype(obj_dtype["x"].type)
            chunk["y"] = new_y[in_bounds].astype(obj_dtype["y"].type)
            chunk["t"] = events_B["t"][mask][in_bounds]
            chunk["p"] = events_B["p"][mask][in_bounds]
            chunks.append(chunk)

        if not chunks:
            return events_A, annotations_A

        obj_events = np.concatenate(chunks)

        if len(obj_events) < self.min_events:
            return events_A, annotations_A

        # --- shift timestamps into A's temporal window ------------------
        # Use a constant offset (t_end_A - t_end_B) so the pasted object's
        # temporal density and trajectory speed are preserved exactly.
        # A 66ms trail in B occupies the last 66ms of A's window — not
        # stretched to fill the full 330ms.
        t_offset = np.int64(t_end_A) - np.int64(t_end_B)
        obj_events["t"] = (
            obj_events["t"].astype(np.int64) + t_offset
        ).astype(obj_dtype["t"].type)

        # --- merge and sort by timestamp --------------------------------
        events_aug = np.concatenate([events_A, obj_events])
        events_aug = events_aug[np.argsort(events_aug["t"])]

        # label: translated final box, preserving any extra fields (class, track)
        new_ann = list(ann_final)
        new_ann[0] = float(new_x1)
        new_ann[1] = float(new_y1)
        new_ann[2] = float(new_x2)
        new_ann[3] = float(new_y2)
        annotations_aug = list(annotations_A) + [new_ann]

        return events_aug, annotations_aug
