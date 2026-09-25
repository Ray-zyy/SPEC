"""
spec.metrics —— 传统指标全套 (E1(b), E3 全表) + 水文签名 (Eckhardt BFI, RB 闪变, 退水常数)
所有函数输入 (y_obs, y_sim) 一维数组, 内部剔除 NaN 配对.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

from .core import clean_pairs


def _c(y, yh):
    y, yh, _ = clean_pairs(y, yh); return y, yh


def nse(y, yh):
    y, yh = _c(y, yh); return 1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)


def log_nse(y, yh, eps=None):
    y, yh = _c(y, yh)
    if eps is None:
        eps = 0.01 * np.mean(y)          # Pushpalatha 2012 惯例
    return nse(np.log(y + eps), np.log(yh + eps))


def kge(y, yh):
    y, yh = _c(y, yh)
    r = np.corrcoef(y, yh)[0, 1]; a = np.std(yh) / np.std(y); b = np.mean(yh) / np.mean(y)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def kge_prime(y, yh):
    """Kling 2012: 变差项用变异系数比."""
    y, yh = _c(y, yh)
    r = np.corrcoef(y, yh)[0, 1]
    g = (np.std(yh) / np.mean(yh)) / (np.std(y) / np.mean(y)); b = np.mean(yh) / np.mean(y)
    return 1 - np.sqrt((r - 1) ** 2 + (g - 1) ** 2 + (b - 1) ** 2)


def kge_np(y, yh):
    """Pool 2018 非参数 KGE: Spearman r + 归一化 FDC 差异."""
    y, yh = _c(y, yh)
    r = spearmanr(y, yh).correlation
    fdc_o = np.sort(y / (len(y) * y.mean())); fdc_s = np.sort(yh / (len(yh) * yh.mean()))
    a = 1 - 0.5 * np.sum(np.abs(fdc_s - fdc_o)); b = np.mean(yh) / np.mean(y)
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def pbias(y, yh):
    y, yh = _c(y, yh); return 100 * np.sum(yh - y) / np.sum(y)


def rmse(y, yh):
    y, yh = _c(y, yh); return np.sqrt(np.mean((yh - y) ** 2))


def rsr(y, yh):
    y, yh = _c(y, yh); return rmse(y, yh) / np.std(y)


def volumetric_efficiency(y, yh):
    y, yh = _c(y, yh); return 1 - np.sum(np.abs(yh - y)) / np.sum(y)


def mape(y, yh):
    y, yh = _c(y, yh); ok = y > 0
    return 100 * np.mean(np.abs(yh[ok] - y[ok]) / y[ok]) if ok.any() else np.nan


def smape(y, yh):
    y, yh = _c(y, yh); den = np.abs(y) + np.abs(yh)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, np.abs(yh - y) / np.where(den > 0, den, 1.0), 0.0)
    return 200 * np.mean(r)


def _fdc(x):
    return np.sort(x)[::-1]


def bias_fhv(y, yh, h=0.02):
    """Yilmaz 2008 高流量段体积偏差 (FDC 前 h)."""
    y, yh = _c(y, yh); k = max(int(np.ceil(h * len(y))), 1)
    fo, fs = _fdc(y)[:k], _fdc(yh)[:k]
    return 100 * np.sum(fs - fo) / np.sum(fo)


def bias_flv(y, yh, l=0.7):
    """Yilmaz 2008 低流量段 (FDC 后 30%) 对数体积偏差."""
    y, yh = _c(y, yh); n = len(y); s = int(np.floor(l * n))
    fo, fs = _fdc(y)[s:], _fdc(yh)[s:]
    eps = 1e-6 + 0.0 * fo
    lo, ls = np.log(fo + eps), np.log(fs + eps)
    num = np.sum(ls - ls.min()); den = np.sum(lo - lo.min())
    return -100 * (num - den) / (den + 1e-12)


def bias_fms(y, yh, lo_q=0.2, hi_q=0.7):
    """Yilmaz 2008 中段 FDC 斜率偏差."""
    y, yh = _c(y, yh); n = len(y)
    i1, i2 = int(lo_q * n), int(hi_q * n)
    fo, fs = _fdc(y), _fdc(yh)
    so = np.log(fo[i1] + 1e-6) - np.log(fo[i2] + 1e-6)
    ss = np.log(fs[i1] + 1e-6) - np.log(fs[i2] + 1e-6)
    return 100 * (ss - so) / (so + 1e-12)


def peak_flow_bias(y, yh, q=0.99):
    y, yh = _c(y, yh); t = np.quantile(y, q); idx = y >= t
    return 100 * (np.mean(yh[idx]) - np.mean(y[idx])) / np.mean(y[idx])


def quantile_bias(y, yh, q=0.05):
    """Q95 (exceedance 95% == quantile 0.05) 偏差, 百分比. q 为分位数."""
    y, yh = _c(y, yh); qo, qs = np.quantile(y, q), np.quantile(yh, q)
    return 100 * (qs - qo) / (qo + 1e-12)


def mam7(y):
    """平均年最低 7 日流量 (要求 y 为日序列, 用 365 天分年)."""
    y = np.asarray(y, float); yrs = len(y) // 365
    mins = []
    for k in range(yrs):
        seg = y[k * 365:(k + 1) * 365]
        r = np.convolve(seg, np.ones(7) / 7, mode="valid")
        mins.append(np.nanmin(r))
    return float(np.mean(mins)) if mins else np.nan


def eckhardt_bfi(q, alpha=0.98, bfi_max=0.80):
    """Eckhardt 2005 两参数递归滤波; 返回 (BFI, baseflow)."""
    q = np.asarray(q, float); q = np.where(np.isfinite(q), q, 0.0)
    b = np.zeros_like(q); b[0] = q[0] * bfi_max
    for t in range(1, len(q)):
        b[t] = ((1 - bfi_max) * alpha * b[t - 1] + (1 - alpha) * bfi_max * q[t]) / (1 - alpha * bfi_max)
        b[t] = min(b[t], q[t])
    return float(np.sum(b) / (np.sum(q) + 1e-12)), b


def bfi_bias(y, yh, **kw):
    y, yh = _c(y, yh)
    return 100 * (eckhardt_bfi(yh, **kw)[0] - eckhardt_bfi(y, **kw)[0]) / (eckhardt_bfi(y, **kw)[0] + 1e-12)


def richards_baker_flashiness(q):
    q = np.asarray(q, float); q = q[np.isfinite(q)]
    return float(np.sum(np.abs(np.diff(q))) / (np.sum(q[1:]) + 1e-12))


def recession_constant(q, min_len=5):
    """退水常数 k: 连续下降段上 ln(q_t/q_{t-1}) 的中位数取负指数 -> q_t = k q_{t-1}."""
    q = np.asarray(q, float)
    ratios = []
    run = 0
    for t in range(1, len(q)):
        if np.isfinite(q[t]) and np.isfinite(q[t - 1]) and 0 < q[t] < q[t - 1]:
            run += 1
            if run >= min_len:
                ratios.append(q[t] / q[t - 1])
        else:
            run = 0
    return float(np.median(ratios)) if ratios else np.nan


def decorrelation_time(res, max_lag=200):
    """残差序列 lag-1 去相关时间 tau = -1/ln(r1) (若 r1<=0 返回 1)."""
    res = np.asarray(res, float); res = res[np.isfinite(res)] - np.nanmean(res)
    r1 = np.corrcoef(res[:-1], res[1:])[0, 1]
    return 1.0 if not (0 < r1 < 1) else float(-1.0 / np.log(r1))


CONVENTIONAL = {
    "NSE": nse, "lnNSE": log_nse, "KGE": kge, "KGEp": kge_prime, "KGEnp": kge_np,
    "PBIAS": pbias, "RMSE": rmse, "RSR": rsr, "VE": volumetric_efficiency,
    "MAPE": mape, "SMAPE": smape,
    "BiasFHV": bias_fhv, "BiasFLV": bias_flv, "BiasFMS": bias_fms,
    "PeakBias": peak_flow_bias, "Q95Bias": lambda y, yh: quantile_bias(y, yh, 0.05),
    "Q90Bias": lambda y, yh: quantile_bias(y, yh, 0.10), "BFIBias": bfi_bias,
}


def all_conventional(y, yh):
    out = {}
    for k, f in CONVENTIONAL.items():
        try:
            out[k] = float(f(y, yh))
        except Exception:
            out[k] = np.nan
    return out
