"""
Split train / val a livello di SEQUENZA, deterministico e senza leak temporale.
"""
import hashlib


def sequence_holdout(windows, split, val_frac=0.15, seed=42):
    """Tiene le finestre delle sequenze di train o di val (split per hdf5_path)."""
    if split not in ('train', 'val'):
        return windows

    def _bucket(seq):
        h = int(hashlib.md5(f"{seed}:{seq}".encode()).hexdigest(), 16)
        return (h % 1_000_000) / 1_000_000.0      # valore stabile in [0,1) per sequenza

    val_seqs = {w['hdf5_path'] for w in windows if _bucket(w['hdf5_path']) < val_frac}
    if split == 'val':
        return [w for w in windows if w['hdf5_path'] in val_seqs]
    return [w for w in windows if w['hdf5_path'] not in val_seqs]
