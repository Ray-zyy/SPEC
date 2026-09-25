"""
E6 —— 损失函数消融: 误差剖面是架构的性质还是训练目标的性质?

先训练 (见 scripts/run_e6.sh): LSTM 与 Transformer × {mse, mae, nse, logmse, delta} × 多种子
再分析:
  python experiments/e6_loss_ablation.py --dataset camels_aus --archs lstm,transformer

产出: results/e6_pareto.csv, figures/F9_pareto_front.png, F9b_curves_by_loss.png
诚实报告: delta-loss 是否让 NSE 变差 (有界损失对严重错误样本梯度消失).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (RUNS, RES, FIG, list_runs, run_meta,     # noqa
                                seed_median_predictions, evaluate_run, save_table)
from spec.core import ALPHA_GRID                                          # noqa
import spec.plotting as sp                                                # noqa

LOSSES = ["mse", "mae", "nse", "logmse", "delta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--archs", default="lstm,transformer")
    ap.add_argument("--regime", default="temporal")
    a = ap.parse_args()
    archs = a.archs.split(",")

    groups = {}
    for p in list_runs():
        m = run_meta(p)
        if m["dataset"] != a.dataset or m["regime"] != a.regime:
            continue
        if m["model"] in archs and m["loss"] in LOSSES:
            groups.setdefault((m["model"], m["loss"]), []).append(p)
    # 对 logmse/delta 优先使用物理流量空间重新训练的运行，避免与旧错误空间结果混合。
    for key in list(groups):
        arch, loss = key
        if loss in ("logmse", "delta"):
            phys = [p for p in groups[key] if "_phys_" in p.name]
            if phys:
                groups[key] = phys

    summ, curve_med = [], {}
    for (arch, loss), paths in groups.items():
        pred = seed_median_predictions(paths)
        df, curves = evaluate_run(pred, dict(arch=arch, loss=loss, n_seeds=len(paths)))
        summ.append(df)
        curve_med[f"{arch}[{loss}]"] = (
            np.nanmedian(np.vstack([c.E_up for c in curves.values()]), 0),
            np.nanmedian(np.vstack([c.E_lo for c in curves.values()]), 0))
    summ = pd.concat(summ, ignore_index=True)
    save_table(summ, "e6_summary.csv")

    pareto = summ.groupby(["arch", "loss"])[["E5_up", "E5_lo", "A_up", "A_lo", "NSE", "lnNSE",
                                             "KGE", "PBIAS", "delta_bar"]].median().reset_index()
    save_table(pareto, "e6_pareto.csv")
    print(pareto.round(3).to_string(index=False))

    # F9 Pareto 前沿
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, arch in enumerate(archs):
        sub = pareto[pareto.arch == arch]
        axs[0].scatter(sub.E5_up, sub.E5_lo, s=45, marker="os"[i % 2], label=arch)
        for _, r in sub.iterrows():
            axs[0].annotate(r.loss, (r.E5_up, r.E5_lo), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    axs[0].set_xlabel(r"$E^{\uparrow}_5$ (peak segment, lower is better)")
    axs[0].set_ylabel(r"$E^{\downarrow}_5$ (baseflow segment)")
    axs[0].set_title("peak-baseflow competence Pareto front", fontsize=9); axs[0].legend(fontsize=7, frameon=False)
    # NSE 代价 (诚实负结果)
    for i, arch in enumerate(archs):
        sub = pareto[pareto.arch == arch].set_index("loss").reindex(LOSSES)
        axs[1].plot(range(len(LOSSES)), sub.NSE.values, marker="o", label=f"{arch} NSE")
        axs[1].plot(range(len(LOSSES)), sub.lnNSE.values, marker="s", ls="--", label=f"{arch} lnNSE")
    axs[1].set_xticks(range(len(LOSSES))); axs[1].set_xticklabels(LOSSES)
    axs[1].set_ylabel("median score"); axs[1].legend(fontsize=6, frameon=False)
    axs[1].set_title(r"cost of the $\delta$-loss (reported honestly)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "F9_pareto_front.png"); plt.close(fig)

    sp.plot_models_overlay(curve_med, None, "E6: dual curves under five training losses",
                           FIG / "F9b_curves_by_loss.png")
    print("E6 完成")


if __name__ == "__main__":
    main()
