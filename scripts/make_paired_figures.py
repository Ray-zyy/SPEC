"""
scripts/make_paired_figures.py —— "两条曲线" 对照图套件 (论文 F11-F20 与补充材料)

本文件专门产出双曲线对照图: 每张图要么是同一模型的 **峰值支 vs 基流支**,
要么是 **两个对象 (模型/损失/制度/率定目标) 的同一条曲线**, 并尽量附上
成对 bootstrap 差异曲线 —— 差异曲线带跨过 0 才是"两者无显著差别"的证据.

用法:
  python scripts/make_paired_figures.py --all                     # 按 runs/ 里已有的运行自动出全套
  python scripts/make_paired_figures.py --pair aus_lstm_nse_s0 aus_gr4j_kge
  python scripts/make_paired_figures.py --pair aus_gr4j_kge aus_gr4j_lnnse --basin 105101A --B 500
  python scripts/make_paired_figures.py --branch aus_lstm_nse_s0 --basin 105101A
  python scripts/make_paired_figures.py --prefix fake_            # 用伪造运行自检

产出目录: figures/paired/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spec import spec, ALPHA_GRID
from spec.core import curves
from spec.bootstrap import bootstrap_curves, pointwise_band, simultaneous_band, paired_difference
from spec.metrics import nse, log_nse, kge
from spec.plotting import PALETTE, plot_dual, style_error_axis, style_bias_axis, add_branch_legend
from experiments.common import RUNS, FIG, list_runs, load_preds, run_meta, seed_median_predictions

OUT = FIG / "paired"
OUT.mkdir(parents=True, exist_ok=True)
X = ALPHA_GRID * 100


def _logx(ax):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel(r"$\alpha$ (%)")


def _median_curves(pred: pd.DataFrame, min_n=365):
    """跨站中位数与 IQR: 返回 dict[key] = (median, q25, q75), key in E_up/E_lo/B_up/B_lo."""
    acc = {k: [] for k in ("E_up", "E_lo", "B_up", "B_lo")}
    for b, g in pred.groupby("basin"):
        g = g.sort_values("date").dropna(subset=["obs", "sim"])
        if len(g) < min_n:
            continue
        c = curves(g.obs.values, g.sim.values)
        for k in acc:
            acc[k].append(getattr(c, k))
    out = {}
    for k, v in acc.items():
        a = np.array(v)
        out[k] = (np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0))
    out["_n"] = len(acc["E_up"])
    return out


def _one_basin(pred, basin=None):
    """取某站 (默认取有效样本最多的站) 的 (obs, sim, dates, basin)."""
    if basin is None:
        basin = pred.dropna(subset=["sim"]).groupby("basin").size().idxmax()
    g = pred[pred.basin == basin].sort_values("date").dropna(subset=["obs", "sim"])
    return g.obs.values, g.sim.values, g.date.values, basin


# ==========================================================================  P1 分支对照
def fig_branch_pair(run, basin=None, B=300, fname=None):
    """P1: 同一模型 峰值支 vs 基流支 —— 误差、偏差、边际密度、局部熵 四联双曲线."""
    pred = load_preds(RUNS / run)
    y, yh, _, b = _one_basin(pred, basin)
    r = spec(y, yh)
    boot = bootstrap_curves(y, yh, B=B, seed=0)
    c = r["curves"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.1))
    plot_dual(axs[0], c.alphas, c.E_up, c.E_lo, PALETTE[0],
              band_up=pointwise_band(boot["E_up"]), band_lo=pointwise_band(boot["E_lo"]))
    style_error_axis(axs[0], f"{b} / {run}: error"); add_branch_legend(axs[0])
    plot_dual(axs[1], c.alphas, c.B_up, c.B_lo, PALETTE[3],
              band_up=pointwise_band(boot["B_up"]), band_lo=pointwise_band(boot["B_lo"]))
    style_bias_axis(axs[1], "bias  (with 95% pointwise band)")
    for br, ls in (("up", "-"), ("lo", "--")):
        a = r["summary"][f"alpha_star_{br}"]
        if np.isfinite(a):
            axs[1].axvline(a * 100, ls=":", lw=0.8, color="k")
            axs[1].text(a * 100, 0.75 if br == "up" else -0.9, rf"$\alpha^*_{{{br}}}$={a*100:.0f}%", fontsize=7)
    axs[2].plot(X, r["density"]["e_up"], "-", color=PALETTE[2], label=r"$e^{\uparrow}$")
    axs[2].plot(X, r["density"]["e_lo"], "--", color=PALETTE[2], label=r"$e^{\downarrow}$")
    for cp in r["changepoints"]["up"]:
        axs[2].axvline(cp * 100, lw=0.7, color=PALETTE[2], alpha=0.6)
    for cp in r["changepoints"]["lo"]:
        axs[2].axvline(cp * 100, lw=0.7, color=PALETTE[2], alpha=0.6, ls="--")
    axs[2].set_ylabel(r"$e(\alpha)$"); axs[2].set_title("marginal density + PELT", fontsize=9)
    axs[2].legend(fontsize=7, frameon=False)
    axs[3].plot(X, r["entropy"]["H_up"], "-", color="k", label=r"$H^{\uparrow}$")
    axs[3].plot(X, r["entropy"]["H_lo"], "--", color="k", label=r"$H^{\downarrow}$")
    axs[3].axhline(1.0, lw=0.6, color="grey", ls=":")
    axs[3].set_ylim(0, 1.08); axs[3].set_ylabel(r"$H(\alpha)$")
    axs[3].set_title("local sign entropy (1 = random, 0 = systematic)", fontsize=8)
    axs[3].legend(fontsize=7, frameon=False)
    for ax in axs:
        _logx(ax)
    fig.tight_layout()
    fname = fname or OUT / f"P1_branch_{run}_{b}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P2 两模型对照
def fig_two_models(runA, runB, basin=None, B=300, aggregate=False, fname=None, labels=None):
    """
    P2: 两个模型的同一条曲线并排 + 成对 bootstrap 差异曲线 (共用随机块).
    aggregate=True 时画跨站中位数 (只出前两栏, 差异栏改为逐站 ΔA 的分布).
    """
    la, lb = labels or (runA, runB)
    pa, pb = load_preds(RUNS / runA), load_preds(RUNS / runB)
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.2))
    if aggregate:
        ma, mb = _median_curves(pa), _median_curves(pb)
        for ax, key, styl in ((axs[0], "E", style_error_axis), (axs[1], "B", style_bias_axis)):
            for m, lab, col in ((ma, la, PALETTE[0]), (mb, lb, PALETTE[1])):
                plot_dual(ax, ALPHA_GRID, m[f"{key}_up"][0], m[f"{key}_lo"][0], col, label=lab,
                          band_up=(m[f"{key}_up"][1], m[f"{key}_up"][2]),
                          band_lo=(m[f"{key}_lo"][1], m[f"{key}_lo"][2]))
            styl(ax, f"{key} curves: median over {ma['_n']} basins (IQR)")
            add_branch_legend(ax)
        # 逐站 A 的配对差异
        rows = []
        common = sorted(set(pa.basin) & set(pb.basin))
        for b in common:
            ga = pa[pa.basin == b].sort_values("date"); gb = pb[pb.basin == b].sort_values("date")
            ca = curves(ga.obs.values, ga.sim.values); cb = curves(gb.obs.values, gb.sim.values)
            rows.append(dict(basin=b, dA_up=ca.area("up") - cb.area("up"),
                             dA_lo=ca.area("lo") - cb.area("lo")))
        d = pd.DataFrame(rows)
        axs[2].axhline(0, color="k", lw=0.7)
        axs[2].boxplot([d.dA_up.dropna(), d.dA_lo.dropna()], labels=[r"$\Delta A^{\uparrow}$", r"$\Delta A^{\downarrow}$"])
        axs[2].set_title(f"per-basin difference  {la} − {lb}\n(<0 means {la} better)", fontsize=8)
        axs[2].set_ylabel(r"$\Delta A$")
    else:
        ya, yha, _, b = _one_basin(pa, basin)
        gb = pb[pb.basin == b].sort_values("date").dropna(subset=["obs", "sim"])
        yhb = gb.sim.values
        n = min(len(ya), len(yhb))
        ya, yha, yhb = ya[:n], yha[:n], yhb[:n]
        ca, cb = curves(ya, yha), curves(ya, yhb)
        ba = bootstrap_curves(ya, yha, B=B, seed=0)
        bb = bootstrap_curves(ya, yhb, B=B, seed=0)
        for ax, key, styl in ((axs[0], "E", style_error_axis), (axs[1], "B", style_bias_axis)):
            for c, bo, lab, col in ((ca, ba, la, PALETTE[0]), (cb, bb, lb, PALETTE[1])):
                plot_dual(ax, c.alphas, getattr(c, f"{key}_up"), getattr(c, f"{key}_lo"), col, label=lab,
                          band_up=pointwise_band(bo[f"{key}_up"]), band_lo=pointwise_band(bo[f"{key}_lo"]))
            styl(ax, f"{b}: {key} curves"); add_branch_legend(ax)
        dif = paired_difference(ya, yha, yhb, B=B, seed=0)
        for key, ls in (("E_up", "-"), ("E_lo", "--")):
            ens = dif[key] if key in dif else None
            if ens is None:
                continue
            med = np.nanmedian(ens, 0)
            lo, hi = pointwise_band(ens)
            axs[2].plot(X, med, ls, color=PALETTE[2], lw=1.5,
                        label=rf"$\Delta E^{{\{'uparrow' if key.endswith('up') else 'downarrow'}}}$")
            axs[2].fill_between(X, lo, hi, color=PALETTE[2], alpha=0.16, lw=0)
        axs[2].axhline(0, color="k", lw=0.8)
        axs[2].set_ylabel(rf"$E_{{{la}}} - E_{{{lb}}}$")
        axs[2].set_title("paired bootstrap difference (band crossing 0 = n.s.)", fontsize=8)
        axs[2].legend(fontsize=7, frameon=False)
        _logx(axs[2])
    fig.tight_layout()
    fname = fname or OUT / f"P2_{runA}_vs_{runB}{'_agg' if aggregate else ''}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P3 过程线/FDC 对照
def fig_hydrograph_fdc(run, basin=None, years=3, fname=None):
    """P3: obs vs sim 两条曲线 —— 时间域过程线 + 频率域 FDC (对数轴) + 逐日 delta 剖面."""
    pred = load_preds(RUNS / run)
    y, yh, dates, b = _one_basin(pred, basin)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.1))
    sl = slice(0, min(len(y), int(years * 365)))
    axs[0].plot(dates[sl], y[sl], "k", lw=0.8, label="observed")
    axs[0].plot(dates[sl], yh[sl], color=PALETTE[0], lw=0.8, label=run)
    axs[0].set_ylabel("Q (mm/d)"); axs[0].legend(fontsize=7, frameon=False)
    axs[0].set_title(f"{b}: hydrograph ({years} yr excerpt)", fontsize=9)
    ex = np.linspace(0, 100, len(y))
    axs[1].semilogy(ex, np.sort(y)[::-1] + 1e-3, "k", lw=1.0, label="observed")
    axs[1].semilogy(ex, np.sort(yh)[::-1] + 1e-3, color=PALETTE[0], lw=1.0, label=run)
    axs[1].set_xlabel("exceedance (%)"); axs[1].set_ylabel("Q (mm/d)")
    axs[1].set_title("flow duration curves", fontsize=9); axs[1].legend(fontsize=7, frameon=False)
    from spec.core import discrepancy, obs_orderings
    d = discrepancy(y, yh)
    up, lo = obs_orderings(y, 0)
    axs[2].plot(np.arange(1, len(y) + 1) / len(y) * 100, np.cumsum(d[up]) / np.arange(1, len(y) + 1),
                "-", color=PALETTE[0], label=r"$E^{\uparrow}(\alpha)$ (running)")
    axs[2].plot(np.arange(1, len(y) + 1) / len(y) * 100, np.cumsum(d[lo]) / np.arange(1, len(y) + 1),
                "--", color=PALETTE[0], label=r"$E^{\downarrow}(\alpha)$ (running)")
    axs[2].set_ylim(0, 1); axs[2].set_ylabel(r"$E(\alpha)$"); _logx(axs[2])
    axs[2].set_title("bidirectional sweep", fontsize=9); axs[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fname = fname or OUT / f"P3_hydro_fdc_{run}_{b}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P4 率定目标对照
def fig_objective_tradeoff(run_kge, run_lnnse, fname=None):
    """
    P4 (全文最直观的一张): 同一概念模型分别按 KGE 与 lnNSE 率定 —— 两支曲线应当交叉.
    左: 跨站中位数误差曲线; 中: 偏差曲线; 右: 逐站 (E5_up, E5_lo) 的配对箭头图.
    """
    pk, pl = load_preds(RUNS / run_kge), load_preds(RUNS / run_lnnse)
    mk, ml = _median_curves(pk), _median_curves(pl)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for ax, key, styl in ((axs[0], "E", style_error_axis), (axs[1], "B", style_bias_axis)):
        for m, lab, col in ((mk, "calibrated on KGE", PALETTE[0]), (ml, "calibrated on lnNSE", PALETTE[1])):
            plot_dual(ax, ALPHA_GRID, m[f"{key}_up"][0], m[f"{key}_lo"][0], col, label=lab,
                      band_up=(m[f"{key}_up"][1], m[f"{key}_up"][2]),
                      band_lo=(m[f"{key}_lo"][1], m[f"{key}_lo"][2]))
        styl(ax, f"{key} curves (median over basins)")
        add_branch_legend(ax)
    rows = []
    for b in sorted(set(pk.basin) & set(pl.basin)):
        gk = pk[pk.basin == b].sort_values("date"); gl = pl[pl.basin == b].sort_values("date")
        ck = curves(gk.obs.values, gk.sim.values); cl = curves(gl.obs.values, gl.sim.values)
        rows.append(dict(basin=b, k_up=ck.at(0.05, "E_up"), k_lo=ck.at(0.05, "E_lo"),
                         l_up=cl.at(0.05, "E_up"), l_lo=cl.at(0.05, "E_lo")))
    d = pd.DataFrame(rows).dropna()
    for _, r in d.iterrows():
        axs[2].annotate("", xy=(r.l_up, r.l_lo), xytext=(r.k_up, r.k_lo),
                        arrowprops=dict(arrowstyle="->", lw=0.6, color="grey", alpha=0.7))
    axs[2].scatter(d.k_up, d.k_lo, s=12, color=PALETTE[0], label="KGE", zorder=3)
    axs[2].scatter(d.l_up, d.l_lo, s=12, color=PALETTE[1], label="lnNSE", zorder=3)
    axs[2].set_xlabel(r"$E^{\uparrow}(5\%)$  peak-branch error")
    axs[2].set_ylabel(r"$E^{\downarrow}(5\%)$  baseflow-branch error")
    axs[2].set_title("per-basin shift when the objective changes", fontsize=8)
    axs[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fname = fname or OUT / f"P4_objective_{run_kge}_vs_{run_lnnse}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P5 多损失 / 多模型叠加
def fig_overlay(runs, labels=None, title="", fname=None, kind="E"):
    """P5: 任意多组运行的双支曲线叠加 (E6 五种损失 / E3 五个架构)."""
    labels = labels or runs
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for i, (r, lab) in enumerate(zip(runs, labels)):
        m = _median_curves(load_preds(RUNS / r))
        col = PALETTE[i % 10]
        axs[0].plot(X, m[f"{kind}_up"][0], "-", color=col, lw=1.5, label=lab)
        axs[0].fill_between(X, m[f"{kind}_up"][1], m[f"{kind}_up"][2], color=col, alpha=0.12, lw=0)
        axs[1].plot(X, m[f"{kind}_lo"][0], "--", color=col, lw=1.5, label=lab)
        axs[1].fill_between(X, m[f"{kind}_lo"][1], m[f"{kind}_lo"][2], color=col, alpha=0.12, lw=0)
    for ax, ttl in ((axs[0], r"peak branch  $%s^{\uparrow}$" % kind),
                    (axs[1], r"baseflow branch  $%s^{\downarrow}$" % kind)):
        (style_error_axis if kind == "E" else style_bias_axis)(ax, ttl)
        _logx(ax); ax.legend(fontsize=7, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fname = fname or OUT / f"P5_overlay_{kind}_{len(runs)}runs.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P6 种子离散度
def fig_seed_spread(prefix, label="", fname=None):
    """P6: 单个种子的曲线 (细灰线) vs 种子中位数 (粗线) —— 模型差异是否大于种子波动."""
    paths = [p for p in list_runs(f"{prefix}*") if "pub" not in p.name]
    if len(paths) < 2:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for p in paths:
        m = _median_curves(load_preds(p))
        axs[0].plot(X, m["E_up"][0], "-", color="grey", lw=0.7, alpha=0.8)
        axs[1].plot(X, m["E_lo"][0], "--", color="grey", lw=0.7, alpha=0.8)
    med = _median_curves(seed_median_predictions(paths))
    axs[0].plot(X, med["E_up"][0], "-", color=PALETTE[3], lw=2.0, label="seed median")
    axs[1].plot(X, med["E_lo"][0], "--", color=PALETTE[3], lw=2.0, label="seed median")
    for ax, ttl in ((axs[0], r"peak branch, %d seeds" % len(paths)),
                    (axs[1], r"baseflow branch, %d seeds" % len(paths))):
        style_error_axis(ax, f"{label or prefix}: {ttl}")
        _logx(ax); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fname = fname or OUT / f"P6_seedspread_{prefix.strip('_')}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P7 制度对照
def fig_regime(run_temporal, runs_pub, fname=None):
    """P7: temporal vs PUB (五折合并) 的双支曲线 —— 陌生流域先丢峰值还是先丢基流?"""
    pt = load_preds(RUNS / run_temporal)
    pubs = [load_preds(RUNS / r) for r in runs_pub if (RUNS / r).exists()]
    if not pubs:
        return None
    pp = pd.concat(pubs)
    mt, mp = _median_curves(pt), _median_curves(pp)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for ax, key, styl in ((axs[0], "E", style_error_axis), (axs[1], "B", style_bias_axis)):
        for m, lab, col in ((mt, "temporal (gauged)", PALETTE[0]), (mp, "PUB (leave-region-out)", PALETTE[1])):
            plot_dual(ax, ALPHA_GRID, m[f"{key}_up"][0], m[f"{key}_lo"][0], col, label=lab,
                      band_up=(m[f"{key}_up"][1], m[f"{key}_up"][2]),
                      band_lo=(m[f"{key}_lo"][1], m[f"{key}_lo"][2]))
        styl(ax, f"{key} curves: temporal vs PUB")
        add_branch_legend(ax)
    fig.tight_layout()
    fname = fname or OUT / f"P7_regime_{run_temporal}.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  P8 标量 vs 曲线排序
def fig_rank_disagreement(runs, fname=None):
    """P8: 逐站 NSE 排名 vs E5_up 排名的两条排序曲线 —— 排序翻转的可视化."""
    rows = []
    for r in runs:
        p = load_preds(RUNS / r)
        for b, g in p.groupby("basin"):
            g = g.sort_values("date").dropna(subset=["obs", "sim"])
            if len(g) < 365:
                continue
            c = curves(g.obs.values, g.sim.values)
            rows.append(dict(run=r, basin=b, NSE=nse(g.obs.values, g.sim.values),
                             lnNSE=log_nse(g.obs.values, g.sim.values),
                             KGE=kge(g.obs.values, g.sim.values),
                             E5_up=c.at(0.05, "E_up"), E5_lo=c.at(0.05, "E_lo")))
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="basin", columns="run", values=["NSE", "E5_up", "E5_lo"])
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.2))
    for ax, (a, b, invert) in zip(axs, [("NSE", "E5_up", True), ("NSE", "E5_lo", True), ("E5_up", "E5_lo", False)]):
        ra = d.groupby("basin")[a].rank(ascending=not invert)
        ax.scatter(d[a], d[b], s=8, c=[PALETTE[list(runs).index(r) % 10] for r in d.run], alpha=0.6)
        ax.set_xlabel(a); ax.set_ylabel(b)
    axs[0].set_title("scalar vs curve feature (each point = basin × model)", fontsize=8)
    rho = d[["NSE", "lnNSE", "KGE", "E5_up", "E5_lo"]].corr(method="spearman")
    rho.to_csv(ROOT / "results" / "paired_rank_spearman.csv")
    fig.tight_layout()
    fname = fname or OUT / "P8_rank_disagreement.png"
    fig.savefig(fname); plt.close(fig)
    return fname


# ==========================================================================  自动全套
def auto_all(prefix="aus_", basin=None, B=300):
    made = []
    runs = [p.name for p in list_runs(f"{prefix}*")]
    if not runs:
        print(f"runs/ 下没有 {prefix}* 的运行, 先训练或先跑 tests/make_fake_runs.py")
        return made
    have = set(runs)

    def pick(*cands):
        for c in cands:
            if c in have:
                return c
        return None

    dl = pick(f"{prefix}lstm_nse_s0", *[r for r in runs if "lstm" in r])
    g_kge = pick(f"{prefix}gr4j_kge", *[r for r in runs if "gr4j" in r and "kge" in r])
    g_ln = pick(f"{prefix}gr4j_lnnse", *[r for r in runs if "gr4j" in r and "lnnse" in r])
    hbv = pick(f"{prefix}hbv_kge", *[r for r in runs if "hbv" in r])

    if dl:
        made += [fig_branch_pair(dl, basin, B), fig_hydrograph_fdc(dl, basin)]
    if dl and g_kge:
        made += [fig_two_models(dl, g_kge, basin, B),
                 fig_two_models(dl, g_kge, aggregate=True)]
    if g_kge and g_ln:
        made.append(fig_objective_tradeoff(g_kge, g_ln))
    if dl and hbv:
        made.append(fig_two_models(dl, hbv, aggregate=True))
    # 五个架构叠加
    archs = [pick(f"{prefix}{a}_nse_s0") for a in ("lstm", "gru", "transformer", "patchtst", "tcn")]
    archs = [a for a in archs if a]
    if len(archs) >= 2:
        made.append(fig_overlay(archs, [a.split("_")[1] for a in archs],
                                "architectures (median over basins)", OUT / "P5_architectures.png"))
    # 五种损失叠加
    losses = [pick(f"{prefix}lstm_{l}_s0") for l in ("mse", "mae", "nse", "logmse", "delta")]
    losses = [l for l in losses if l]
    if len(losses) >= 2:
        made.append(fig_overlay(losses, [l.split("_")[2] for l in losses],
                                "training losses (LSTM)", OUT / "P5_losses.png"))
    # 种子离散度
    if dl:
        made.append(fig_seed_spread(f"{prefix}lstm_nse_s"))
    # 制度对照
    pubs = [r for r in runs if "lstm" in r and "pub" in r]
    if dl and pubs:
        made.append(fig_regime(dl, pubs))
    # 排序分歧
    cmp_runs = [r for r in (dl, g_kge, hbv) if r]
    if len(cmp_runs) >= 2:
        made.append(fig_rank_disagreement(cmp_runs))
    made = [m for m in made if m]
    print(f"共产出 {len(made)} 张对照图 -> {OUT}")
    for m in made:
        print("  ", m)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prefix", default="aus_")
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--hydro", default=None)
    ap.add_argument("--objective", nargs=2, default=None)
    ap.add_argument("--overlay", nargs="+", default=None)
    ap.add_argument("--basin", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args()
    if a.all:
        auto_all(a.prefix, a.basin, a.B)
    if a.pair:
        print(fig_two_models(a.pair[0], a.pair[1], a.basin, a.B, aggregate=a.aggregate))
    if a.branch:
        print(fig_branch_pair(a.branch, a.basin, a.B))
    if a.hydro:
        print(fig_hydrograph_fdc(a.hydro, a.basin))
    if a.objective:
        print(fig_objective_tradeoff(*a.objective))
    if a.overlay:
        print(fig_overlay(a.overlay))
