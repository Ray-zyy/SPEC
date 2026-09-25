"""
E8 —— 决策相关性 (审稿人最认可的实验)

python experiments/e8_decision.py --dataset camels_aus --regime temporal --workers 16

(a) 洪水预警: Q95/Q99/Q99.5 阈值下 POD/FAR/CSI; 用 E5_up, B5_up 与 NSE 分别回归比解释力
(b) 生态流量: Q90/Q95/MAM7 偏差 vs E5_lo, B5_lo 与 lnNSE
(c) 诚实负结果: 峰现时间误差 Δt_p 无法从 SPEC 曲线恢复 -> 二者互补, 应联合报告
产出: results/e8_*.csv, figures/F10_decision_regressions.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
_FONT_PATH = Path(__file__).resolve().parents[1] / "DejaVuMathTeXGyre.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_FONT_PATH)).get_name()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (RES, FIG, save_table, evaluate_pair)          # noqa
from experiments.e3_benchmark import collect                                  # noqa
from spec.metrics import mam7, quantile_bias                                  # noqa
from spec.core import CURVE_FEATURES                                          # noqa


def contingency(y, yh, q):
    thr = np.quantile(y, q)
    o, s = y >= thr, yh >= thr
    hits = float(np.sum(o & s)); miss = float(np.sum(o & ~s)); fa = float(np.sum(~o & s))
    pod = hits / (hits + miss) if hits + miss else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + miss + fa) if hits + miss + fa else np.nan
    return pod, far, csi


def peak_time_error(y, yh, q=0.99, win=5):
    """每个观测峰事件附近 ±win 天内模拟峰的时间偏移中位数 |Δt_p|."""
    thr = np.quantile(y, q)
    peaks = [i for i in range(win, len(y) - win) if y[i] >= thr and y[i] == y[i - win:i + win + 1].max()]
    if not peaks:
        return np.nan
    d = [abs(int(np.argmax(yh[i - win:i + win + 1])) - win) for i in peaks]
    return float(np.median(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--regime", default="temporal")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    groups = collect(a.dataset, a.regime)
    rows = []
    for name, g in groups.items():
        for b, gg in g["pred"].groupby("basin"):
            gg = gg.sort_values("date").dropna(subset=["sim"])
            if len(gg) < 730:
                continue
            y, yh = gg.obs.values, gg.sim.values
            s, _ = evaluate_pair(y, yh)
            rec = dict(basin=str(b), model=name, **{k: s[k] for k in
                       CURVE_FEATURES + ["NSE", "lnNSE", "KGE", "PBIAS", "BiasFHV", "BiasFLV"] if k in s})
            for q, tag in [(0.95, "Q95"), (0.99, "Q99"), (0.995, "Q995")]:
                pod, far, csi = contingency(y, yh, q)
                rec |= {f"POD_{tag}": pod, f"FAR_{tag}": far, f"CSI_{tag}": csi}
            rec["Q90_bias"] = quantile_bias(y, yh, 0.10)
            rec["Q95_bias"] = quantile_bias(y, yh, 0.05)
            mo, ms = mam7(y), mam7(yh)
            rec["MAM7_bias"] = 100 * (ms - mo) / (mo + 1e-9)
            rec["dtp"] = peak_time_error(y, yh)
            rows.append(rec)
    df = pd.DataFrame(rows)
    save_table(df, f"e8_decision_{a.dataset}_{a.regime}.csv")

    # ---------------- 回归比较解释力
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from scipy.stats import spearmanr

    def compare(target, predictor_sets):
        """解释力对比. 目标做 1/99 分位 winsorize (偏差类目标有极端离群);
        交叉验证按站点分组 (同一站不同时出现在训练与测试), 单变量另报 Spearman."""
        out = []
        d = df.dropna(subset=[target]).copy()
        lo, hi = np.nanpercentile(d[target], [1, 99])
        d[target] = d[target].clip(lo, hi)
        for nm, cols in predictor_sets.items():
            cols = [c for c in cols if c in d.columns]
            if not cols:
                continue
            X = d[cols].replace([np.inf, -np.inf], np.nan).values
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
            r2 = cross_val_score(pipe, X, d[target].values, cv=GroupKFold(5),
                                 groups=d.basin.values, scoring="r2").mean()
            r2_model = cross_val_score(pipe, X, d[target].values, cv=GroupKFold(5),
                                       groups=d.model.values, scoring="r2").mean()
            rho = (spearmanr(d[cols[0]], d[target], nan_policy="omit").correlation
                   if len(cols) == 1 else np.nan)
            out.append(dict(target=target, predictors=nm, n=len(d), cv_r2_by_basin=float(r2),
                            cv_r2_by_model=float(r2_model), spearman=rho,
                            abs_spearman=abs(rho) if rho == rho else np.nan))
        return out

    res = []
    for tgt in ["POD_Q95", "CSI_Q99", "CSI_Q995", "FAR_Q99"]:
        res += compare(tgt, {"NSE": ["NSE"], "KGE": ["KGE"], "E5_up": ["E5_up"],
                             "B5_up": ["B5_up"], "peak_branch_features": ["E5_up", "B5_up", "A_up"]})
    for tgt in ["Q90_bias", "Q95_bias", "MAM7_bias"]:
        res += compare(tgt, {"lnNSE": ["lnNSE"], "NSE": ["NSE"], "E5_lo": ["E5_lo"],
                             "B5_lo": ["B5_lo"], "low_branch_features": ["E5_lo", "B5_lo", "A_lo"]})
    # (c) 诚实负结果: Δt_p 能否从曲线特征恢复
    res += compare("dtp", {"all_curve_features": [c for c in CURVE_FEATURES],
                           "NSE": ["NSE"], "KGE": ["KGE"]})
    reg = pd.DataFrame(res); save_table(reg, f"e8_regressions_{a.dataset}_{a.regime}.csv")
    print(reg.round(3).to_string(index=False))

    # ---------------- F10
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axs[0].scatter(df.E5_up, df.CSI_Q99, s=8, alpha=.5, label=r"$E^{\uparrow}_5$")
    axs[0].set_xlabel(r"$E^{\uparrow}_5$"); axs[0].set_ylabel(r"CSI at $Q_{99}$")
    ax0b = axs[0].twiny(); ax0b.scatter(df.NSE, df.CSI_Q99, s=8, alpha=.35, color="tab:orange")
    ax0b.set_xlabel("NSE", color="tab:orange")
    axs[0].set_title("(a) flood-warning skill", fontsize=10, pad=8)
    axs[1].scatter(df.E5_lo, df.MAM7_bias.clip(-200, 200), s=8, alpha=.5)
    axs[1].set_xlabel(r"$E^{\downarrow}_5$"); axs[1].set_ylabel("MAM7 bias (%)")
    axs[1].axhline(0, lw=.6, c="k"); axs[1].set_title("(b) environmental low flows", fontsize=10, pad=8)
    sub = reg[reg.target == "dtp"]
    axs[2].bar(sub.predictors, sub.cv_r2_by_basin.clip(lower=0)); axs[2].set_ylabel(r"CV $R^2$ for $\Delta t_p$")
    axs[2].tick_params(axis="x", rotation=20, labelsize=7)
    axs[2].set_title("(c) negative result: curves carry no timing information", fontsize=10, pad=8)
    fig.savefig(FIG / f"F10_decision_{a.dataset}_{a.regime}.png"); plt.close(fig)
    print("E8 完成")


if __name__ == "__main__":
    main()

