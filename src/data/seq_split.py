"""
Split train / val a livello di SEQUENZA, deterministico e senza leak temporale.
"""
import hashlib


def sequence_holdout(windows, split, val_frac=0.15, seed=42):
    """Filtra `windows` (lista di dict con chiave 'hdf5_path') tenendo train o val per SEQUENZA.

    Args:
        windows: lista di finestre (ognuna con 'hdf5_path' = la sequenza).
        split: 'train' → sequenze NON-val ; 'val' → sequenze val ; altro → invariato 
        val_frac: frazione (attesa) delle sequenze uniche assegnate al val 

    Returns:
        la sotto-lista di `windows` appartenente allo split richiesto.
    """
    if split not in ('train', 'val'):
        return windows

    def _bucket(seq):
        h = int(hashlib.md5(f"{seed}:{seq}".encode()).hexdigest(), 16)
        return (h % 1_000_000) / 1_000_000.0      # valore stabile in [0,1) per sequenza

    val_seqs = {w['hdf5_path'] for w in windows if _bucket(w['hdf5_path']) < val_frac}
    if split == 'val':
        return [w for w in windows if w['hdf5_path'] in val_seqs]
    return [w for w in windows if w['hdf5_path'] not in val_seqs]
