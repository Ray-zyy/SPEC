"""
E1 —— 受控扰动: 框架能否看见它应当看见的东西 (无需训练任何模型, 最先跑)

python experiments/e1_perturbation.py --dataset camels_aus --basins_file data/processed/basins_aus.txt \
       --n_basins 60 --workers 16

产出:
  results/e1_metrics.csv          每 (站, 扰动) 的曲线特征 + 传统指标
  results/e1_identifiability.csv  三种特征集的 balanced accuracy (T4)
  results/e1_confusion_*.csv      混淆矩阵
  figures/F3_perturbation_grid.png   扰动签名网格
  figures/F3b_P5_compensating.png    P5 卖点图 (PBIAS=0 但曲线定位出 alpha*)
  figures/F3c_entropy_contrast.png   P8 vs P3/P4 熵对照
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RES, FIG, save_table, pmap                       # noqa
from spec import spec, ALPHA_GRID, CURVE_FEATURES                               # noqa
from spec.core import curves, local_sign_entropy                                # noqa
from spec.metrics import all_conventional                                       # noqa
from spec.perturb import PERTURBATIONS, CLASS_OF                                # noqa
from spec.data import load_basin, slice_period                                  # noqa
import spec.plotting as sp                                                      # noqa


def one_basin(task):
    dataset, basin, seed, contaminate = task
    try:
        df = load_basin(dataset, basin)
        y = slice_period(df, dataset, "test")["q"].values.astype(float)
        y = y[np.isfinite(y)]
        if len(y) < 1000:
            return None
        rng = np.random.default_rng(seed)
        rows, sig = [], {}
        for name, fn, par in PERTURBATIONS:
            yh = np.clip(fn(y, rng, **par), 0, None)
            if contaminate > 0 and not name.startswith("P8"):
                # 叠加与 P8 同量级的随机噪声, 使各扰动类型不再平凡可分
                yh = np.clip(yh * np.exp(rng.normal(0, contaminate, len(yh))), 0, None)
                # Preserve the analytic P5 compensation after contamination so its
                # aggregate PBIAS remains exactly zero while its shape is noisy.
                if name == "P5_compensating":
                    den = float(np.sum(yh))
                    if den > 0:
                        yh = yh * (float(np.sum(y)) / den)
            r = spec(y, yh)
            row = dict(basin=basin, perturbation=name, pclass=CLASS_OF[name],
                       **r["summary"], **all_conventional(y, yh))
            rows.append(row)
            c = r["curves"]
            sig[name] = dict(E_up=c.E_up, E_lo=c.E_lo, B_up=c.B_up, B_lo=c.B_lo,
                             H_up=r["entropy"]["H_up"], H_lo=r["entropy"]["H_lo"])
        return pd.DataFrame(rows), sig, basin
    except Exception as e:
        print("skip", basin, e); return None


def identifiability(df):
    """(i) 传统指标向量 (ii) 曲线特征 (iii) 二者并集 -> 多分类还原扰动类型, 跨站分组 CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_predict, GroupKFold
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

    conv = ["NSE", "lnNSE", "KGE", "PBIAS", "RMSE", "BiasFHV", "BiasFLV", "BiasFMS", "VE", "RSR"]
    sets = {"conventional": conv, "spec_curve": CURVE_FEATURES, "both": conv + CURVE_FEATURES}
    y = df["pclass"].astype(str).to_numpy(); groups = df["basin"].astype(str).to_numpy()
    out, cms = [], {}
    for nm, cols in sets.items():
        X = df[[c for c in cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).values
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=4000, C=1.0))
        pred = cross_val_predict(clf, X, y, groups=groups, cv=GroupKFold(5), n_jobs=-1)
        ba = balanced_accuracy_score(y, pred)
        labs = sorted(set(y))
        rec = recall_score(y, pred, labels=labs, average=None, zero_division=0)
        out.append(dict(feature_set=nm, n_features=X.shape[1], balanced_accuracy=ba,
                        **{f"recall_{l}": float(r) for l, r in zip(labs, rec)}))
        cms[nm] = pd.DataFrame(confusion_matrix(y, pred, labels=labs), index=labs, columns=labs)
        print(f"  {nm:12s} balanced acc = {ba:.3f}")
    return pd.DataFrame(out), cms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--basins_file", required=True)
    ap.add_argument("--n_basins", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--contaminate", type=float, default=0.15,
                    help="叠加在每种结构性扰动上的乘性噪声 sd; 0 表示纯净扰动(分类将平凡可分)")
    a = ap.parse_args()

    basins = [l.strip() for l in open(a.basins_file) if l.strip()][: a.n_basins]
    res = [r for r in pmap(one_basin, [(a.dataset, b, i, a.contaminate) for i, b in enumerate(basins)],
                           a.workers, "E1") if r]
    df = pd.concat([r[0] for r in res], ignore_index=True)
    save_table(df, "e1_metrics.csv")

    # ---- 签名网格 (F3): 每个扰动在各站上的中位数曲线
    sig = {}
    for name, _, _ in PERTURBATIONS:
        sig[name] = {k: np.vstack([r[1][name][k] for r in res]) for k in ("E_up", "E_lo", "B_up", "B_lo")}
    sp.plot_perturbation_grid(sig, FIG / "F3_perturbation_grid.png")

    # ---- P5 卖点图
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    nm = "P5_compensating"
    med = {k: np.nanmedian(sig[nm][k], 0) for k in sig[nm]}
    sp.plot_dual(axs[0], ALPHA_GRID, med["E_up"], med["E_lo"], sp.PALETTE[0]); sp.style_error_axis(axs[0], "P5: error curves")
    sp.add_branch_legend(axs[0])
    sp.plot_dual(axs[1], ALPHA_GRID, med["B_up"], med["B_lo"], sp.PALETTE[3]); sp.style_bias_axis(axs[1], "P5: bias curves")
    d5 = df[df.perturbation == nm]
    ast = np.nanmedian(d5["alpha_star_up"]) * 100
    axs[1].axvline(ast, ls=":", color="k"); axs[1].text(ast, .8, rf"median $\alpha^*$={ast:.0f}%", fontsize=7)
    axs[2].boxplot([df[df.perturbation == nm]["PBIAS"].dropna(), df[df.perturbation == nm]["NSE"].dropna(),
                    df[df.perturbation == nm]["B5_up"].dropna(), df[df.perturbation == nm]["B5_lo"].dropna()],
                   tick_labels=["PBIAS", "NSE", r"$B^{\uparrow}(5\%)$", r"$B^{\downarrow}(5\%)$"])
    axs[2].axhline(0, color="k", lw=.6); axs[2].set_title("P5: invisible to PBIAS, visible to the bias curves", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "F3b_P5_compensating.png"); plt.close(fig)

    # ---- 熵对照 (P8 -> H≈1, P3/P4 -> H≈0)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for nm2, col in [("P8_mult_noise_0.30", sp.PALETTE[2]), ("P3_base_infl_0.25", sp.PALETTE[1]),
                     ("P4_base_depl_0.50", sp.PALETTE[4])]:
        H = np.vstack([r[1][nm2]["H_lo"] for r in res])
        ax.plot(ALPHA_GRID * 100, np.nanmedian(H, 0), label=nm2, color=col)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel(r"$\alpha$ (%)"); ax.set_ylabel(r"$H(\alpha)$ (low branch)")
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(FIG / "F3c_entropy_contrast.png"); plt.close(fig)

    # ---- 可识别性 (T4)
    tab, cms = identifiability(df)
    save_table(tab, "e1_identifiability.csv")
    for k, cm in cms.items():
        cm.to_csv(RES / f"e1_confusion_{k}.csv")
    print("E1 完成. P5 中位 alpha* =", round(float(np.nanmedian(d5['alpha_star_up'])), 3),
          " 中位 PBIAS =", round(float(np.nanmedian(d5['PBIAS'])), 4))


if __name__ == "__main__":
    main()
