"""核心性质单元测试 (论文 Propositions). 运行: python -m pytest tests/test_core.py -q"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec.core import (discrepancy, signed_discrepancy, delta_from_ratio, curves,
                       curve_area, zero_crossing, ALPHA_GRID)

rng = np.random.default_rng(0)
y = np.abs(rng.lognormal(0, 1.2, 4000)); y[y < 0.2] = 0
yh = np.clip(y * np.exp(rng.normal(0, .4, len(y))), 0, None)


def test_bounded():
    d = discrepancy(y, yh); b = signed_discrepancy(y, yh)
    assert d.min() >= 0 and d.max() <= 1 and b.min() >= -1 and b.max() <= 1
    assert np.allclose(d, np.abs(b))


def test_zero_convention():
    assert discrepancy([0.], [0.])[0] == 0.0          # 双零
    assert discrepancy([0.], [3.])[0] == 1.0          # 单零
    assert signed_discrepancy([3.], [0.])[0] == -1.0


def test_exchange_symmetry():
    assert np.allclose(discrepancy(y, yh), discrepancy(yh, y))
    assert np.allclose(signed_discrepancy(y, yh), -signed_discrepancy(yh, y))


def test_scale_invariance():
    assert np.allclose(discrepancy(y, yh), discrepancy(7.3 * y, 7.3 * yh))


def test_tanh_identity():
    rho = np.array([2.0, 0.5, 3.0, 1 / 3, 1.5, 1 / 1.5])
    assert np.allclose(delta_from_ratio(rho), np.abs((rho - 1) / (rho + 1)))
    assert np.isclose(delta_from_ratio(2.0), 1 / 3) and np.isclose(delta_from_ratio(0.5), 1 / 3)


def test_multiplicative_symmetry():
    for r in (1.2, 2.0, 5.0):
        assert np.isclose(delta_from_ratio(r), delta_from_ratio(1 / r))


def test_closure_property():
    c = curves(y, yh)
    assert np.isclose(c.E_up[-1], c.E_lo[-1]) and np.isclose(c.B_up[-1], c.B_lo[-1])
    assert np.isclose(c.E_up[-1], np.mean(discrepancy(y, yh)))


def test_bounded_influence():
    """单个灾难性点对 alpha=100% 段的贡献 <= 1/n."""
    c0 = curves(y, yh); yh2 = yh.copy(); yh2[0] = 1e9
    c1 = curves(y, yh2)
    assert abs(c1.E_up[-1] - c0.E_up[-1]) <= 1.0 / len(y) + 1e-12


def test_min_segment_suppression():
    c = curves(y[:100], yh[:100], min_m=30)
    assert np.isnan(c.E_up[0])                          # m(1%)=1 < 30 -> 抑制


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
