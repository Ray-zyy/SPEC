"""Minimal SPEC example with synthetic streamflow."""
import sys
from pathlib import Path
import numpy as np

# Allow `python examples/quickstart.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec import spec

rng = np.random.default_rng(7)

# Synthetic non-negative daily streamflow (n > 30 so the default minimum segment is valid).
y_obs = rng.lognormal(mean=0.0, sigma=1.0, size=1000)
y_sim = np.clip(y_obs * np.exp(rng.normal(0.0, 0.30, size=y_obs.size)), 0.0, None)

out = spec(y_obs, y_sim)
summary = out["summary"]

for key in ["A_up", "A_lo", "E5_up", "E5_lo", "B5_up", "B5_lo", "delta_bar"]:
    print(f"{key:>10s}: {summary[key]:.4f}")
