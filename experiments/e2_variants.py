"""
E2 —— 为什么必须换核函数 (打掉原 ATPE)

python experiments/e2_variants.py --dataset camels_aus --basins_file data/processed/basins_aus.txt \
       --runs "aus_gr4j_kge,aus_lstm_nse_s0" --B 200 --workers 16

分析 (对应论文 E2 (a)-(e)):
 (a) 低流量支未定义 / 超出 [0,3] 的站-模型对占比, 按零流量比例分层
 (b) bootstrap 变异系数 CV(alpha)
 (c) 留一影响: 删掉最极端一点后曲线变化幅度
 (d) alpha=100% 处跨站方差
 (e) 两支之间的秩相关
产出: results/e2_*.csv, figures/F4_variant_failure.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RUNS, RES, FIG, load_preds, save_table, pmap    # noqa
from spec.variants import variant_curves, VARIANTS                             # noqa
from spec.core import ALPHA_GRID                                               # noqa
from spec.bootstrap import stationary_bootstrap_indices, choose_block_length   # noqa
import spec.plotting as sp                                                     # noqa

PLOT_MAX = 3.0


def analyse_pair(task):
    basin, run, y, yh, B = task
    y = np.asarray(y); yh = np.asarray(yh)
    ok = np.isfinite(y) & np.isfinite(yh); y, yh = y[ok], yh[ok]
    if len(y) < 365:
        return None
    v = variant_curves(y, yh)
    zf = float(np.mean(y == 0))
    rows = []
    # (b) bootstrap CV
    L = choose_block_length(y, yh)
    idx = stationary_bootstrap_indices(len(y), L, B, seed=0)
    boot = {k: np.empty((B, len(ALPHA_GRID))) for k in v}
    for b in range(B):
        vb = variant_curves(y[idx[b]], yh[idx[b]])
        for k in v:
            boot[k][b] = vb[k]
    # (c) 留一影响: 去掉观测最大 & 最小的那一点
    drop = np.ones(len(y), bool); drop[np.argmax(y)] = False; drop[np.argmin(y)] = False
    v_loo = variant_curves(y[drop], yh[drop])
    for name in VARIANTS:
        for br in ("up", "lo"):
            key = f"{name}_{br}"
            c = v[key]
            fin = np.isfinite(c)
            rows.append(dict(
                basin=basin, run=run, variant=name, branch=br, zero_frac=zf,
                undefined_frac=float(1 - fin.mean()),
                exceeds_range=float(np.nanmean(np.abs(c[fin]) > PLOT_MAX)) if fin.any() else np.nan,
                value_alpha5=float(c[4]), value_alpha100=float(c[-1]),
                cv_alpha5=float(np.nanstd(boot[key][:, 4]) / (abs(np.nanmean(boot[key][:, 4])) + 1e-12)),
                cv_alpha100=float(np.nanstd(boot[key][:, -1]) / (abs(np.nanmean(boot[key][:, -1])) + 1e-12)),
                loo_max_change=float(np.nanmax(np.abs(v_loo[key] - c))),
                loo_rel_change=float(np.nanmax(np.abs((v_loo[key] - c) / (np.abs(c) + 1e-12)))),
            ))
    return pd.DataFrame(rows), {k: v[k] for k in v}, basin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="逗号分隔的 runs/ 子目录名")
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--basins_file", default=None)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    keep = set(l.strip() for l in open(a.basins_file)) if a.basins_file else None
    tasks, examples = [], {}
    for run in a.runs.split(","):
        pred = load_preds(RUNS / run)
        for b, g in pred.groupby("basin"):
            if keep and str(b) not in keep:
                continue
            g = g.sort_values("date")
            tasks.append((str(b), run, g.obs.values, g.sim.values, a.B))
    res = [r for r in pmap(analyse_pair, tasks, a.workers, "E2") if r]
    df = pd.concat([r[0] for r in res], ignore_index=True)
    save_table(df, "e2_variant_diagnostics.csv")

    # (a) 分层汇总
    df["zero_stratum"] = pd.cut(df.zero_frac, [-.001, .001, .05, .2, 1.0],
                                labels=["0%", "0-5%", "5-20%", ">20%"])
    tabA = (df[df.branch == "lo"].groupby(["variant", "zero_stratum"], observed=True)
            .agg(n=("basin", "size"), undefined=("undefined_frac", "mean"),
                 out_of_range=("exceeds_range", "mean")).reset_index())
    save_table(tabA, "e2_a_undefined_by_zero_stratum.csv")
    # (b)(c)(d)
    tabB = df.groupby(["variant", "branch"]).agg(
        cv5=("cv_alpha5", "median"), cv100=("cv_alpha100", "median"),
        loo_abs=("loo_max_change", "median"), loo_rel=("loo_rel_change", "median"),
        var_alpha100_across_basins=("value_alpha100", "var")).reset_index()
    save_table(tabB, "e2_bcd_stability.csv")
    # (e) 两支秩相关
    from scipy.stats import spearmanr
    rows = []
    for var in VARIANTS:
        u = df[(df.variant == var) & (df.branch == "up")].set_index(["basin", "run"]).value_alpha5
        l = df[(df.variant == var) & (df.branch == "lo")].set_index(["basin", "run"]).value_alpha5
        j = pd.concat([u, l], axis=1, keys=["up", "lo"]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(variant=var, n=len(j),
                         spearman_branches=float(spearmanr(j.up, j.lo).correlation) if len(j) > 5 else np.nan))
    save_table(pd.DataFrame(rows), "e2_e_branch_correlation.csv")

    # ---- F4: 四变体在同一站上的两支曲线 (原式在基流支爆掉)
    ex = res[0][1]; basin = res[0][2]
    fig, axs = plt.subplots(1, 5, figsize=(16, 2.9))
    for j, var in enumerate(VARIANTS):
        ax = axs[j]
        sp.plot_dual(ax, ALPHA_GRID, ex[f"{var}_up"], ex[f"{var}_lo"], sp.PALETTE[j])
        ax.set_title(var, fontsize=9); ax.set_ylabel("curve value")
        finite = np.concatenate([ex[f"{var}_up"], ex[f"{var}_lo"]])
        finite = finite[np.isfinite(finite)]
        if var == "V4_delta":
            ax.set_ylim(0, 1)
        elif len(finite):
            ax.set_yscale("symlog", linthresh=1)
        nan_lo = np.mean(~np.isfinite(ex[f"{var}_lo"]))
        ax.text(.02, .95, f"low-branch undefined: {nan_lo:.0%}", transform=ax.transAxes, fontsize=7, va="top")
    axs[0].legend(*sp.plt.gca().get_legend_handles_labels(), fontsize=6)
    fig.suptitle(f"E2 / F4  basin {basin}: V1--V3 unbounded/undefined; V4 r=0 and V5 r=ymax are bounded reference-point variants", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "F4_variant_failure.png"); plt.close(fig)
    print("E2 完成")


if __name__ == "__main__":
    main()
