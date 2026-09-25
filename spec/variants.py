"""
spec.variants —— E2 用的四种核函数变体 (两支都算):
  V1 体积归一化 (原 ATPE, eq. atpe): sum|e| / sum y        -> 分母为 0 时 NaN (未定义)
  V2 段内 MAPE: mean(|e|/y)                                  -> y=0 时 NaN
  V3 段内 MAE (有量纲): mean|e|
  V4 本文 delta: mean delta
"""
from __future__ import annotations
import numpy as np
from .core import clean_pairs, obs_orderings, segment_sizes, discrepancy, ALPHA_GRID

VARIANTS = ("V1_volume", "V2_mape", "V3_mae", "V4_delta", "V5_delta_ymax")


def variant_curves(y, yhat, alphas=ALPHA_GRID, tie_seed=0, min_m=30):
    y, yhat, _ = clean_pairs(y, yhat); n = len(y)
    up, lo = obs_orderings(y, tie_seed)
    m = segment_sizes(n, alphas)
    res = {}
    for br, order in (("up", up), ("lo", lo)):
        yo, ys = y[order], yhat[order]
        ae = np.abs(ys - yo)
        cs_ae, cs_y = np.cumsum(ae), np.cumsum(yo)
        with np.errstate(divide="ignore", invalid="ignore"):
            v1 = np.where(cs_y[m - 1] > 0, cs_ae[m - 1] / cs_y[m - 1], np.nan)
            ape = np.where(yo > 0, ae / yo, np.nan)
            cum_ape = np.cumsum(np.where(np.isnan(ape), np.nan, ape))
            v2 = cum_ape[m - 1] / m            # 任一 y=0 -> NaN 传播 = 未定义
            v3 = cs_ae[m - 1] / m
            v4 = np.cumsum(discrepancy(yo, ys, reference=0.0))[m - 1] / m
            rmax = float(np.nanmax(y)) if len(y) else 0.0
            v5 = np.cumsum(discrepancy(yo, ys, reference=rmax))[m - 1] / m
        for name, v in zip(VARIANTS, (v1, v2, v3, v4, v5)):
            v = np.array(v, float); v[m < min_m] = np.nan
            res[f"{name}_{br}"] = v
    return res
