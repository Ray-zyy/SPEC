"""
spec.core  ——  SPEC 双向百分位误差曲线的核心实现
==========================================================
对应论文 Methods 3.1–3.5：
    delta_i = |yhat-y| / (|yhat|+|y|)              (eq. delta)   双零 -> 0
    beta_i  = (yhat-y) / (|yhat|+|y|)              (eq. beta)
    E^up(a) / E^lo(a) : 按观测排序后前 m(a)=ceil(a*n) 个点的 delta 均值   (eq. E)
    B^up(a) / B^lo(a) : 同上, 用 beta                                  (eq. B)
    A^up, A^lo, E5, alpha*, Delta_asym, SPECSS                          (3.3 派生标量)
    e(a) = d/da [a E(a)]  边际密度, PELT 突变点, 局部符号熵 H(a)          (3.5)
一行接口:  spec(y_obs, y_sim) -> dict

约定 (论文 3.7 Practical implementation):
    * 只按观测排序; 并列值用固定随机种子 (tie_seed) 打破, 所有模型共用;
    * 缺测配对剔除后 n 为有效配对数;
    * m(alpha) < min_m (默认 30) 的 alpha 处曲线置 NaN;
    * 零流量不加 epsilon.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

ALPHA_GRID = np.round(np.arange(1, 101) / 100.0, 4)      # 1%..100%


# ----------------------------------------------------------------------------- 逐点核
def discrepancy(y, yhat, reference=0.0):
    """delta_i in [0,1]. 双零 -> 0; 恰一个为零 -> 1."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y-reference) + np.abs(yhat-reference)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(den > 0, np.abs(yhat - y) / den, 0.0)
    return d


def signed_discrepancy(y, yhat, reference=0.0):
    """beta_i in [-1,1], >0 为高估."""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y-reference) + np.abs(yhat-reference)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(den > 0, (yhat - y) / den, 0.0)
    return b


def delta_from_ratio(rho):
    """Prop. tanh: delta = |tanh(0.5 ln rho)| = |rho-1|/(rho+1)."""
    rho = np.asarray(rho, float)
    return np.abs(np.tanh(0.5 * np.log(rho)))


# ----------------------------------------------------------------------------- 排序
def clean_pairs(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    return y[ok], yhat[ok], ok


def obs_orderings(y, tie_seed: int = 0):
    """返回 (sigma_up, sigma_lo): 观测降序 / 升序索引, 随机(固定种子)破并列."""
    n = len(y)
    perm = np.random.default_rng(tie_seed).permutation(n)
    up = np.lexsort((perm, -y))   # 主键 -y 升序 == y 降序, 次键随机
    lo = np.lexsort((perm, y))
    return up, lo


def segment_sizes(n, alphas=ALPHA_GRID):
    m = np.ceil(np.asarray(alphas) * n).astype(int)
    return np.clip(m, 1, n)


# ----------------------------------------------------------------------------- 曲线
@dataclass
class SpecCurves:
    alphas: np.ndarray
    E_up: np.ndarray
    E_lo: np.ndarray
    B_up: np.ndarray
    B_lo: np.ndarray
    n: int
    m: np.ndarray
    zero_frac: float
    delta_bar: float = field(init=False)
    beta_bar: float = field(init=False)

    def __post_init__(self):
        self.delta_bar = float(self.E_up[-1]); self.beta_bar = float(self.B_up[-1])

    # -- 派生标量 (论文 3.3 / 3.4)
    def area(self, branch="up"):
        return curve_area(self.alphas, getattr(self, f"E_{branch}"))

    def at(self, alpha, which="E_up"):
        k = int(np.argmin(np.abs(self.alphas - alpha)))
        return float(getattr(self, which)[k])

    def alpha_star(self, branch="up"):
        return zero_crossing(self.alphas, getattr(self, f"B_{branch}"))

    def summary(self):
        return dict(
            A_up=self.area("up"), A_lo=self.area("lo"),
            E5_up=self.at(0.05, "E_up"), E5_lo=self.at(0.05, "E_lo"),
            B5_up=self.at(0.05, "B_up"), B5_lo=self.at(0.05, "B_lo"),
            E1_up=self.at(0.01, "E_up"), E1_lo=self.at(0.01, "E_lo"),
            E20_up=self.at(0.20, "E_up"), E20_lo=self.at(0.20, "E_lo"),
            alpha_star_up=self.alpha_star("up"), alpha_star_lo=self.alpha_star("lo"),
            delta_bar=self.delta_bar, beta_bar=self.beta_bar,
            Delta_asym=self.area("up") - self.area("lo"),
            n=self.n, m1=int(self.m[0]), zero_frac=self.zero_frac,
        )


def _running_mean_ordered(vals_ordered, m):
    cs = np.cumsum(vals_ordered)
    return cs[m - 1] / m


def curves(y, yhat, alphas=ALPHA_GRID, tie_seed=0, min_m=30,
           rank_by="obs", zero_rule="delta0", reference=0.0) -> SpecCurves:
    """
    计算四条曲线.  rank_by ∈ {obs, sim, mean} 仅供 E7(d) 稳健性用, 默认 obs.
    zero_rule ∈ {delta0, exclude}: 双零日置 0 (默认) 或剔除 (E7c).
    """
    y, yhat, _ = clean_pairs(y, yhat)
    if zero_rule == "exclude":
        keep = ~((y == 0) & (yhat == 0)); y, yhat = y[keep], yhat[keep]
    n = len(y)
    alphas = np.asarray(alphas, float)
    if n == 0:
        nan = np.full(len(alphas), np.nan)
        return SpecCurves(alphas, nan, nan, nan, nan, 0, np.zeros(len(alphas), int), np.nan)
    key = {"obs": y, "sim": yhat, "mean": 0.5 * (y + yhat)}[rank_by]
    up, lo = obs_orderings(key, tie_seed)
    d = discrepancy(y, yhat, reference=reference); b = signed_discrepancy(y, yhat, reference=reference)
    m = segment_sizes(n, alphas)
    E_up = _running_mean_ordered(d[up], m); E_lo = _running_mean_ordered(d[lo], m)
    B_up = _running_mean_ordered(b[up], m); B_lo = _running_mean_ordered(b[lo], m)
    small = m < min_m
    for arr in (E_up, E_lo, B_up, B_lo):
        arr[small] = np.nan
    return SpecCurves(alphas, E_up, E_lo, B_up, B_lo, n, m, float(np.mean(y == 0)))


# ----------------------------------------------------------------------------- 标量摘要
def curve_area(alphas, E):
    """A = ∫_0^1 E(a) da, 梯形积分, NaN(小 m 段)忽略; 左端以第一个有效值补齐."""
    ok = np.isfinite(E)
    if ok.sum() < 2:
        return np.nan
    a, e = alphas[ok], E[ok]
    return float(np.trapezoid(e, a) + e[0] * a[0])   # 左侧 [0,a_first] 用 e[0] 补


def zero_crossing(alphas, B):
    """第一次符号翻转处的线性插值 alpha*; 无交叉 -> NaN."""
    ok = np.isfinite(B); a, b = alphas[ok], B[ok]
    if len(b) < 2:
        return np.nan
    s = np.sign(b)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return np.nan if not np.any(b == 0) else float(a[np.argmax(b == 0)])
    i = idx[0]
    return float(a[i] + (a[i + 1] - a[i]) * (-b[i]) / (b[i + 1] - b[i]))


def skill_score(A_model, A_bench):
    """SPECSS = 1 - A_model/A_bench (eq. skill)."""
    return 1.0 - A_model / A_bench


# ----------------------------------------------------------------------------- 边际密度 / 突变点 / 熵
def marginal_density(alphas, E, smooth=True, window=5, polyorder=2):
    """e(a) = d/da [a E(a)], 前向差分 (最后一点后向), 可选 Savitzky-Golay 平滑 (±2 点)."""
    aE = alphas * E
    e = np.full_like(E, np.nan)
    ok = np.isfinite(aE)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return e
    seg = aE[idx]; a = alphas[idx]
    de = np.gradient(seg, a)
    if smooth and len(de) >= window:
        from scipy.signal import savgol_filter
        de = savgol_filter(de, window, polyorder)
    e[idx] = de
    return e


def pelt_mean_shift(x, penalty=None, min_size=5, max_bkps=6):
    """
    PELT / optimal partitioning (Killick 2012) with an l2 mean-shift cost.
    Returns changepoint positions (indices into the *original* array, segment starts),
    excluding 0 and n.  penalty=None -> BIC-type 2*sigma^2*log(n), sigma robustly
    estimated from first differences (MAD/sqrt(2)).
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    xv = x[ok]
    n = len(xv)
    if n < 2 * min_size:
        return np.array([], int)
    if penalty is None:
        dif = np.diff(xv)
        sigma = 1.4826 * np.median(np.abs(dif - np.median(dif))) / np.sqrt(2)
        sigma = max(sigma, 1e-9)
        penalty = 2.0 * sigma ** 2 * np.log(n)
    cs = np.concatenate([[0.0], np.cumsum(xv)])
    cs2 = np.concatenate([[0.0], np.cumsum(xv ** 2)])

    def cost(s, t):                       # l2 cost of segment [s, t)
        L = t - s
        return cs2[t] - cs2[s] - (cs[t] - cs[s]) ** 2 / L

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, int)
    cands = [0]
    for t in range(min_size, n + 1):
        best, best_s = np.inf, 0
        keep = []
        for s in cands:
            if t - s < min_size:
                keep.append(s)
                continue
            v = F[s] + cost(s, t)
            if v + penalty < best:
                best, best_s = v + penalty, s
            keep.append((s, v))
        F[t], last[t] = best, best_s
        # PELT pruning: drop s with F[s]+cost(s,t) > F[t]
        cands = [s if not isinstance(s, tuple) else s[0]
                 for s in keep if (not isinstance(s, tuple)) or s[1] <= F[t]]
        cands.append(t)
        cands = sorted(set(cands))
    bps = []
    t = n
    while t > 0:
        s = last[t]
        if s > 0:
            bps.append(s)
        t = s
    bps = sorted(bps)
    if len(bps) > max_bkps:                # keep the strongest jumps only
        jumps = []
        edges = [0] + bps + [n]
        for i, b in enumerate(bps):
            m1 = (cs[b] - cs[edges[i]]) / (b - edges[i])
            m2 = (cs[edges[i + 2]] - cs[b]) / (edges[i + 2] - b)
            jumps.append(abs(m2 - m1))
        bps = [b for _, b in sorted(zip(jumps, bps), reverse=True)[:max_bkps]]
        bps = sorted(bps)
    idx_ok = np.where(ok)[0]
    return idx_ok[np.array(bps, int)] if bps else np.array([], int)


def local_sign_entropy(y, yhat, alphas=ALPHA_GRID, branch="up", width=5, tie_seed=0, min_m=30):
    """
    H(a): 以 a 为中心、宽 width 个百分位点的窗口内, beta>0 的比例 p 的二元香农熵.
    双零 (beta=0) 排除; 返回 (H, n_eff).
    """
    y, yhat, _ = clean_pairs(y, yhat); n = len(y)
    up, lo = obs_orderings(y, tie_seed)
    order = up if branch == "up" else lo
    b = signed_discrepancy(y, yhat)[order]
    m = segment_sizes(n, alphas)
    H = np.full(len(alphas), np.nan); neff = np.zeros(len(alphas), int)
    half = width / 2.0
    for k, a in enumerate(alphas):
        lo_a, hi_a = max(a - half / 100, 0.0), min(a + half / 100, 1.0)
        s = int(np.floor(lo_a * n)); e = int(np.ceil(hi_a * n))
        w = b[s:max(e, s + 1)]
        w = w[w != 0]
        neff[k] = len(w)
        if len(w) < min_m:
            continue
        p = np.mean(w > 0)
        H[k] = 0.0 if p in (0.0, 1.0) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return H, neff


# ----------------------------------------------------------------------------- 一行接口
def spec(y_obs, y_sim, alphas=ALPHA_GRID, tie_seed=0, min_m=30, entropy_width=5,
         changepoints=True, **kw):
    """
    一行接口. 返回 dict:
      curves: SpecCurves; summary: 标量; density: {e_up,e_lo,b_up,b_lo};
      changepoints: {up: alphas, lo: alphas}; entropy: {H_up,H_lo}
    """
    c = curves(y_obs, y_sim, alphas, tie_seed, min_m, **kw)
    out = {"curves": c, "summary": c.summary()}
    dens = {f"e_{br}": marginal_density(c.alphas, getattr(c, f"E_{br}")) for br in ("up", "lo")}
    dens.update({f"b_{br}": marginal_density(c.alphas, getattr(c, f"B_{br}")) for br in ("up", "lo")})
    out["density"] = dens
    if changepoints:
        out["changepoints"] = {br: c.alphas[pelt_mean_shift(dens[f"e_{br}"])] for br in ("up", "lo")}
        for br in ("up", "lo"):
            cp = out["changepoints"][br]
            out["summary"][f"cp1_{br}"] = float(cp[0]) if len(cp) else np.nan
            out["summary"][f"ncp_{br}"] = int(len(cp))
    H_up, _ = local_sign_entropy(y_obs, y_sim, alphas, "up", entropy_width, tie_seed, min_m)
    H_lo, _ = local_sign_entropy(y_obs, y_sim, alphas, "lo", entropy_width, tie_seed, min_m)
    out["entropy"] = {"H_up": H_up, "H_lo": H_lo}
    out["summary"]["H_mean_up"] = float(np.nanmean(H_up)) if np.isfinite(H_up).any() else np.nan
    out["summary"]["H_mean_lo"] = float(np.nanmean(H_lo)) if np.isfinite(H_lo).any() else np.nan
    out["summary"]["H5_up"] = float(np.nanmean(H_up[:5])) if np.isfinite(H_up[:5]).any() else np.nan
    out["summary"]["H5_lo"] = float(np.nanmean(H_lo[:5])) if np.isfinite(H_lo[:5]).any() else np.nan
    return out


CURVE_FEATURES = ["E5_up", "E5_lo", "A_up", "A_lo", "B5_up", "B5_lo", "alpha_star_up",
                  "alpha_star_lo", "cp1_up", "cp1_lo", "H_mean_up", "H_mean_lo", "H5_up", "H5_lo",
                  "Delta_asym", "beta_bar"]
