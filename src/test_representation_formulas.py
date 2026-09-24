"""
test_representation_formulas.py — verifica GROUND-TRUTH delle formule UFFICIALI.

Valori attesi calcolati A MANO dalle formule dei paper:
  - time-surface HOTS (Lagorce et al., TPAMI 2017):  S_p = exp(-(t_ref - T_p)/tau)
  - tencode (Huang et al., WACV 2023): G = (t_max - t)/(t_max - t_min), R/B = polarita

Gira sul cluster (serve numpy + numba):
  python test_representation_formulas.py
"""
import numpy as np
from data.datasets.event_dataset_forecasting import (
    render_time_surface_numba, render_tencode_numba,
)


def approx(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


# 3 eventi: (x, y, p, age_us) con age = t_ref - t in microsecondi (newest = age minore).
#   (1,1) ON  age=0  -> evento piu recente in assoluto
#   (1,1) OFF age=1  -> stesso pixel, un filo piu vecchio
#   (2,2) ON  age=4  -> evento piu vecchio in assoluto
x = np.array([1, 1, 2], dtype=np.int32)
y = np.array([1, 1, 2], dtype=np.int32)
p = np.array([1, 0, 1], dtype=np.int32)
age = np.array([0.0, 1.0, 4.0], dtype=np.float32)
H = W = 4
tau = np.float32(2.0)

# ── TIME-SURFACE HOTS: [S_on, S_off, 0], S_p = exp(-age_p/tau) ──
ts = render_time_surface_numba(x, y, p, age, tau, H, W)
assert ts.shape == (4, 4, 3), ts.shape
assert approx(ts[1, 1, 0], 1.0), ts[1, 1, 0]              # S_on = exp(-0/2) = 1
assert approx(ts[1, 1, 1], np.exp(-0.5)), ts[1, 1, 1]    # S_off = exp(-1/2) = 0.6065
assert approx(ts[1, 1, 2], 0.0)                          # zero-pad (3o canale)
assert approx(ts[2, 2, 0], np.exp(-2.0)), ts[2, 2, 0]    # exp(-4/2) = 0.1353
assert approx(ts[2, 2, 1], 0.0)
assert approx(ts[0, 0, 0], 0.0) and approx(ts[0, 0, 1], 0.0)   # pixel senza eventi
print("time_surface OK  ->  TS[1,1]=", ts[1, 1], " TS[2,2]=", ts[2, 2])

# invariante HOTS: evento a distanza esattamente tau -> S = 1/e
z = np.zeros(1, dtype=np.int32)
ts2 = render_time_surface_numba(z, z, np.array([1], np.int32),
                                np.array([float(tau)], np.float32), tau, H, W)
assert approx(ts2[0, 0, 0], np.exp(-1.0)), ts2[0, 0, 0]  # exp(-tau/tau) = 1/e = 0.3679
print("invariante S(tau)=1/e OK  ->", float(ts2[0, 0, 0]))

# ── TENCODE: newest -> G=0, oldest -> G=1 ; ON -> R, OFF -> B ──
tc = render_tencode_numba(x, y, p, age, H, W)
assert tc.shape == (4, 4, 3), tc.shape
# (1,1): ultimo evento = ON age=0 (il piu recente) -> R=1, G=(0-0)/4=0, B=0
assert approx(tc[1, 1, 0], 1.0) and approx(tc[1, 1, 1], 0.0) and approx(tc[1, 1, 2], 0.0), tc[1, 1]
# (2,2): ON age=4 (il piu vecchio) -> R=1, G=(4-0)/4=1, B=0
assert approx(tc[2, 2, 0], 1.0) and approx(tc[2, 2, 1], 1.0) and approx(tc[2, 2, 2], 0.0), tc[2, 2]
assert approx(tc[0, 0, 0], 0.0) and approx(tc[0, 0, 1], 0.0) and approx(tc[0, 0, 2], 0.0)
print("tencode OK       ->  TC[1,1]=", tc[1, 1], " TC[2,2]=", tc[2, 2])

# tencode: pixel con OFF come ultimo evento -> B=1, R=0
tc2 = render_tencode_numba(np.array([0], np.int32), np.array([0], np.int32),
                           np.array([0], np.int32), np.array([2.0], np.float32), H, W)
assert approx(tc2[0, 0, 0], 0.0) and approx(tc2[0, 0, 2], 1.0), tc2[0, 0]   # OFF -> B
print("tencode OFF->B OK ->  TC[0,0]=", tc2[0, 0])

print("\nTUTTI I TEST PASSATI - formule ufficiali (HOTS + TENCODE) corrette.")
