from data.datasets.event_dataset_forecasting import FREDMultiDurationTensorDatasetTracking as tracking_dataset
from data.datasets.event_dataset import FREDMultiDurationTensorDataset as detection_dataset


def get_dataset(name, split="train", modality='detection', config=None):
    """Factory function to create datasets with proper pipelines.

    Args:
        name (str): Dataset name ('FRED', etc.)
        split (str): Dataset split ('train', 'val', 'test'). Default: 'train'
        modality (str): Task modality ('detection', 'tracking'). Default: 'detection'
        domain (str): Domain information (optional, for future use)
        config (dict): Configuration object with dataset parameters

    Returns:
        Dataset instance
    """
    dataset = None
    # Il val non ha un file dedicato quindi si ricava dal train. Vedi seq_split.
    file_split = 'train' if split == 'val' else split
    index_file = config.index_path + f"{file_split}_windows_33ms.json"


    if modality == 'detection':

        if name == "FRED":
            dataset = detection_dataset(
                index_file=index_file,
                width=config.img_width,
                height=config.img_height,
                durations_ms=config.durations,
                render_mode=config.render_mode,
                subsample=config.subsample,
                use_only_annotated=config.use_only_annotated,
            )

            use_aug = getattr(config, 'use_event_box_paste', False)
            if split == 'train' and use_aug:
                max_duration_us = int(max(config.durations) * 1000)
                augmentation = EventBoxPaste(
                    windows=dataset.windows,
                    sensor_size=(config.img_width, config.img_height),
                    max_duration_us=max_duration_us,
                    p=getattr(config, 'event_box_paste_p', 0.5),
                    min_events=getattr(config, 'event_box_paste_min_events', 30),
                )
                dataset.augmentation = augmentation
                print(f"EventBoxPaste enabled (p={augmentation.p}, min_events={augmentation.min_events})")



    elif modality == 'tracking':
        if name == "FRED":
            dataset = tracking_dataset(
                index_file=index_file,
                width=config.img_width,
                height=config.img_height,
                durations_ms=config.durations,
                render_mode=config.render_mode,
                subsample=config.subsample,
                num_past_annotations=getattr(config, 'num_past_annotations', 0),
                num_future_annotations=getattr(config, 'num_future_annotations', 0),
            )

    # ── Split train/val ──
    # train e val caricano lo STESSO train_windows, poi qui si tengono sequenze disgiunte.
    if split in ('train', 'val') and dataset is not None:
        from data.seq_split import sequence_holdout
        _n0 = len(dataset.windows)
        dataset.windows = sequence_holdout(
            dataset.windows, split,
            val_frac=getattr(config, 'val_seq_frac', 0.15),
            seed=getattr(config, 'val_seq_seed', 42),
        )
        print(f"  [seq-split] split={split}: {_n0} -> {len(dataset.windows)} finestre "
              f"(held-out per sequenza, val_frac={getattr(config, 'val_seq_frac', 0.15)})")

    return dataset


if __name__ == "__main__":
    print("TODO.")
