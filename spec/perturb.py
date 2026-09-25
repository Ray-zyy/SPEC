"""
spec.perturb —— E1 受控扰动 P1–P8 (论文 E1 procedure)
每个函数: perturb(y, rng, **par) -> yhat.  PERTURBATIONS 给出 (名称, 函数, 参数网格).
P5 用一维求根把 PBIAS 精确调到 0 —— 论文的卖点案例.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def _q(y, p):
    return np.quantile(y[np.isfinite(y)], p)


def p1_peak_damping(y, rng=None, k=0.7):
    q90 = _q(y, 0.90)
    return np.where(y < q90, y, q90 + k * (y - q90))


def p2_peak_inflation(y, rng=None, k=1.15):
    return p1_peak_damping(y, rng, k)


def p3_baseflow_inflation(y, rng=None, c=0.10):
    q70, q50 = _q(y, 0.70), _q(y, 0.50)
    return np.where(y < q70, y + c * q50, y)


def p4_baseflow_depletion(y, rng=None, kappa=0.5):
    q70 = _q(y, 0.70)
    return np.where(y < q70, kappa * y, y)


def p5_compensating(y, rng=None, k=0.7):
    """P1(k) + P3(c), c 由求根确定使 PBIAS == 0 (精确补偿)."""
    y = np.asarray(y, float)
    q90, q70, q50 = _q(y, .90), _q(y, .70), _q(y, .50)
    lost = np.sum(np.where(y >= q90, (1 - k) * (y - q90), 0.0))     # 削峰损失的水量
    ndry = np.sum(y < q70)
    if ndry == 0 or q50 == 0:
        return p1_peak_damping(y, rng, k)
    c = lost / (ndry * q50)                                        # 解析解: c*q50*ndry = lost
    yh = np.where(y < q90, y, q90 + k * (y - q90))
    yh = np.where(y < q70, yh + c * q50, yh)
    return yh


def p6_timing_shift(y, rng=None, tau=1):
    y = np.asarray(y, float)
    out = np.empty_like(y); out[:tau] = y[0]; out[tau:] = y[:-tau]
    return out


def p7_variance_compression(y, rng=None, gamma=0.8):
    mu = np.nanmean(y)
    return np.clip(mu + gamma * (y - mu), 0, None)


def p8_multiplicative_noise(y, rng=None, s=0.1):
    rng = rng or np.random.default_rng(0)
    return np.asarray(y, float) * np.exp(rng.normal(0, s, size=len(y)))


PERTURBATIONS = [
    ("P1_peak_damp_0.70", p1_peak_damping, dict(k=0.70)),
    ("P1_peak_damp_0.85", p1_peak_damping, dict(k=0.85)),
    ("P2_peak_infl_1.15", p2_peak_inflation, dict(k=1.15)),
    ("P2_peak_infl_1.30", p2_peak_inflation, dict(k=1.30)),
    ("P3_base_infl_0.10", p3_baseflow_inflation, dict(c=0.10)),
    ("P3_base_infl_0.25", p3_baseflow_inflation, dict(c=0.25)),
    ("P4_base_depl_0.50", p4_baseflow_depletion, dict(kappa=0.50)),
    ("P4_base_depl_0.75", p4_baseflow_depletion, dict(kappa=0.75)),
    ("P5_compensating",   p5_compensating,      dict(k=0.70)),
    ("P6_shift_1d", p6_timing_shift, dict(tau=1)),
    ("P6_shift_3d", p6_timing_shift, dict(tau=3)),
    ("P7_var_compress_0.80", p7_variance_compression, dict(gamma=0.80)),
    ("P7_var_compress_0.90", p7_variance_compression, dict(gamma=0.90)),
    ("P8_mult_noise_0.10", p8_multiplicative_noise, dict(s=0.10)),
    ("P8_mult_noise_0.30", p8_multiplicative_noise, dict(s=0.30)),
]

CLASS_OF = {name: name.split("_")[0] for name, _, _ in PERTURBATIONS}   # P1..P8 八类标签


def rating_curve_noise(y, rng, hi_sd=0.125, lo_sd=0.25, q_split=0.7):
    """E5(d) 观测不确定性: 异方差乘性率定误差 (高流 ~12.5%, 低流 ~25%)."""
    thr = _q(y, q_split)
    sd = np.where(y >= thr, hi_sd, lo_sd)
    return np.clip(y * np.exp(rng.normal(0, sd)), 0, None)
