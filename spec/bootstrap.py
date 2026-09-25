"""
spec.bootstrap —— 平稳块 bootstrap (Politis & Romano 1994) + 逐点/同时置信带 + 配对差异
关键: 每个重抽样内部 **重新排序、重新分段** (curves() 在重抽样序列上从头算).
"""
from __future__ import annotations
import numpy as np
from .core import curves, ALPHA_GRID, curve_area, zero_crossing
from .metrics import decorrelation_time, nse


def choose_block_length(y, yhat, floor=30):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    tau = decorrelation_time(yhat[ok] - y[ok])
    return int(max(floor, np.ceil(3 * tau)))


def stationary_bootstrap_indices(n, L, B, seed=0):
    """返回 (B, n) 的索引矩阵. 块长 ~ Geometric(1/L), 起点均匀, 循环取模."""
    rng = np.random.default_rng(seed)
    p = 1.0 / L
    idx = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            length = min(length, n - pos)
            idx[b, pos:pos + length] = (start + np.arange(length)) % n
            pos += length
    return idx


def bootstrap_curves(y, yhat, B=1000, L=None, seed=0, alphas=ALPHA_GRID, indices=None, **kw):
    """
    返回 dict: E_up, E_lo, B_up, B_lo 各为 (B, K) 数组, 以及 indices (供配对差异复用).
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat); y, yhat = y[ok], yhat[ok]
    n = len(y)
    if L is None:
        L = choose_block_length(y, yhat)
    if indices is None:
        indices = stationary_bootstrap_indices(n, L, B, seed)
    K = len(alphas)
    out = {k: np.empty((B, K)) for k in ("E_up", "E_lo", "B_up", "B_lo")}
    for b in range(B):
        c = curves(y[indices[b]], yhat[indices[b]], alphas, **kw)   # 重新排序+分段
        for k in out:
            out[k][b] = getattr(c, k)
    out["indices"] = indices; out["L"] = L
    return out


def pointwise_band(ens, q=(2.5, 97.5)):
    return np.nanpercentile(ens, q[0], axis=0), np.nanpercentile(ens, q[1], axis=0)


def simultaneous_band(ens, level=0.95):
    """max-t 同时带: c 使 95% 的曲线满足 sup_a |E_b - mean|/s <= c."""
    mu = np.nanmean(ens, axis=0); s = np.nanstd(ens, axis=0) + 1e-12
    t = np.nanmax(np.abs(ens - mu) / s, axis=1)
    c = np.nanquantile(t, level)
    return mu - c * s, mu + c * s, c


def paired_difference(y, yA, yB, B=1000, L=None, seed=0, alphas=ALPHA_GRID, **kw):
    """共用随机块的配对差异 Delta(a)=E_A-E_B; 返回 ens 差异, 以及同时带排除零的 alpha 范围."""
    bootA = bootstrap_curves(y, yA, B, L, seed, alphas, **kw)
    bootB = bootstrap_curves(y, yB, B, L, seed, alphas, indices=bootA["indices"], **kw)
    res = {}
    for k in ("E_up", "E_lo", "B_up", "B_lo"):
        d = bootA[k] - bootB[k]
        lo, hi, _ = simultaneous_band(d)
        res[k] = {"ens": d, "sim_lo": lo, "sim_hi": hi,
                  "excludes_zero": (lo > 0) | (hi < 0)}
    # 同一 bootstrap 下 NSE 差异是否显著 (E5c 对照)
    yv = y[np.isfinite(y) & np.isfinite(yA) & np.isfinite(yB)]
    idx = bootA["indices"]
    dn = np.array([nse(yv[i], yA[i]) - nse(yv[i], yB[i]) for i in idx])
    lo, hi = np.percentile(dn, [2.5, 97.5])
    res["NSE_diff_significant"] = bool(lo > 0 or hi < 0)
    return res


def bootstrap_summaries(boot):
    """从曲线集合得到 A_up, A_lo, alpha* 的 bootstrap 分布."""
    a = ALPHA_GRID
    A_up = np.array([curve_area(a, e) for e in boot["E_up"]])
    A_lo = np.array([curve_area(a, e) for e in boot["E_lo"]])
    ast = np.array([zero_crossing(a, b) for b in boot["B_up"]])
    return {"A_up": A_up, "A_lo": A_lo, "alpha_star": ast}


def changepoint_frequency(boot, cp_fn, tol=2, alphas=ALPHA_GRID):
    """每个 alpha 处 ±tol 个百分位点内检测到突变点的 bootstrap 频率."""
    from .core import marginal_density
    freq = np.zeros(len(alphas))
    Bn = boot["E_up"].shape[0]
    for b in range(Bn):
        e = marginal_density(alphas, boot["E_up"][b])
        cps = cp_fn(e)
        for c in cps:
            lo, hi = max(0, c - tol), min(len(alphas), c + tol + 1)
            freq[lo:hi] += 1
    return freq / Bn
