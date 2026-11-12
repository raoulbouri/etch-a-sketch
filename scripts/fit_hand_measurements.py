"""Fit mm-per-degree and backlash for one knob from measured data.

CSV with a header and two columns: cumulative knob angle (deg), pen position along that axis (mm).
Sweep forward in equal steps, then back, at least twice. Run:
    uv run python scripts/fit_hand_measurements.py hardware/x_sweep.csv
"""
import sys

import numpy as np

from etch_sim import fit_axis

data = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
f = fit_axis(data[:, 0], data[:, 1])
print(f"k = {f.k_mm_per_deg:.4f} mm/deg   delta = {f.delta_deg:.2f} deg (a reversal costs {2 * f.delta_deg:.1f} deg)")
print(f"residual rms = {f.rms_mm:.2f} mm, max = {f.max_abs_mm:.2f} mm   ({f.n_used} samples used, {f.n_dropped} dropped)")
