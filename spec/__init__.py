"""SPEC: Sweep of Percentile-resolved Error Curves."""
from .core import (spec, curves, discrepancy, signed_discrepancy, delta_from_ratio,
                   marginal_density, pelt_mean_shift, local_sign_entropy,
                   curve_area, zero_crossing, skill_score, ALPHA_GRID, CURVE_FEATURES)
from .metrics import all_conventional, CONVENTIONAL
from .bootstrap import bootstrap_curves, pointwise_band, simultaneous_band, paired_difference
from .variants import variant_curves
__all__ = [n for n in dir() if not n.startswith("_")]
__version__ = "0.1.0"
