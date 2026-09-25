"""
E3 —— 大样本主实验 (全模型 × 两数据集 × 两制度)

python experiments/e3_benchmark.py --dataset camels_aus --regime temporal --workers 16

自动扫描 runs/ 下 dataset+regime 匹配的运行, 多种子取中位数, 然后:
 (a) 各模型中位数曲线 + IQR 带 (F5 头图)
 (b) 排序翻转率 (NSE vs A_up / A_lo / alpha=5%)
 (c) 冗余分析: Spearman 矩阵 + PCA (F6)
 (d) Friedman + Nemenyi + 临界差异图
 (e) 属性回归: 随机森林 + 置换重要性/SHAP
产出: results/e3_*.csv, figures/F5_*.png, F6_*.png, T3_summary.csv
"""
from __future__ import annotations
import argparse, itertools, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import (RUNS, RES, FIG, list_runs, run_meta, load_preds,      # noqa
                                seed_median_predictions, evaluate_run, save_table)
from spec.core import ALPHA_GRID, CURVE_FEATURES                                      # noqa
import spec.plotting as sp                                                            # noqa


def collect(dataset, regime):
    """把 runs/ 按 (model, loss) 分组, 组内多种子取预测中位数."""
    groups = defaultdict(list)
    for p in list_runs():
        m = run_meta(p)
        if m["dataset"] != dataset or m["regime"] != regime:
            continue
        groups[(m["model"], m["loss"])].append(p)
    out = {}
    for (model, loss), paths in groups.items():
        name = model if not loss else f"{model}[{loss}]"
        try:
            pred = seed_median_predictions(paths)
        except ImportError:
            print(f"skip {name}: parquet reader unavailable")
            continue
        out[name] = dict(paths=paths, pred=pred, n_seeds=len(paths))
    return out


def seed_spread(paths, basin_sample=40):
    """种子离散度: 同一模型不同种子的 A_up/A_lo 波动 (论文要求与模型差异并列展示)."""
    from experiments.common import evaluate_run
    rows = []
    for p in paths:
        df, _ = evaluate_run(load_preds(p), dict(run=p.name))
        if len(df):
            rows.append(df.set_index("basin")[["A_up", "A_lo", "NSE"]])
    if len(rows) < 2:
        return None
    stack = pd.concat(rows, keys=range(len(rows)), names=["seed"])
    return stack.groupby("basin").agg(["median", "std"])


def ranking_flip(summ, models, key_pairs):
    """对每个站点、每个模型对: NSE 排序与曲线量排序是否一致."""
    rows = []
    piv = {k: summ.pivot_table(index="basin", columns="model", values=k) for k in
           ["NSE", "A_up", "A_lo", "E5_up", "E5_lo", "E20_up", "delta_bar"]}
    for m1, m2 in itertools.combinations(models, 2):
        for key, better_low in key_pairs:
            a, b = piv["NSE"], piv[key]
            common = a[[m1, m2]].join(b[[m1, m2]], rsuffix="_k").dropna()
            if len(common) == 0:
                continue
            nse_pref_m1 = common[m1] > common[m2]                       # NSE 越大越好
            key_pref_m1 = (common[f"{m1}_k"] < common[f"{m2}_k"]) if better_low else (common[f"{m1}_k"] > common[f"{m2}_k"])
            flip = float(np.mean(nse_pref_m1 != key_pref_m1))
            rows.append(dict(model_a=m1, model_b=m2, criterion=key, n_basins=len(common), flip_rate=flip))
    return pd.DataFrame(rows)


def redundancy(summ, conv_cols, curve_cols):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    cols = [c for c in conv_cols + curve_cols if c in summ.columns]
    X = summ[cols].replace([np.inf, -np.inf], np.nan)
    corr = X.corr(method="spearman")
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    pca = PCA(n_components=min(8, Z.shape[1])).fit(Z)
    load = pd.DataFrame(pca.components_.T, index=cols,
                        columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    load["explained_var_of_PC1"] = pca.explained_variance_ratio_[0]
    return corr, load, pca


def friedman_nemenyi(summ, value, models):
    from scipy.stats import friedmanchisquare, rankdata
    piv = summ.pivot_table(index="basin", columns="model", values=value)[models].dropna()
    # Friedman is undefined for fewer than three independent model groups.
    # Keep the summary pipeline usable for partial conceptual-model runs.
    if len(piv) < 5 or len(models) < 3:
        return None, None, None
    stat, p = friedmanchisquare(*[piv[m].values for m in models])
    ranks = np.apply_along_axis(rankdata, 1, piv.values)                 # 小的更好
    avg = ranks.mean(0)
    k, N = len(models), len(piv)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}.get(k, 3.3)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    return dict(stat=float(stat), p=float(p), N=N), pd.Series(avg, index=models), float(cd)


def plot_cd(avg_ranks, cd, title, fname):
    fig, ax = plt.subplots(figsize=(8.5, max(3.6, 1.4 + 0.32 * len(avg_ranks))))
    s = avg_ranks.sort_values()
    ax.errorbar(s.values, np.arange(len(s)), xerr=cd / 2, fmt="o", color="k", capsize=3)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel(f"average rank (CD={cd:.2f})"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


def attribute_regression(summ, dataset, targets=("A_up", "A_lo", "alpha_star_up")):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score, GroupKFold
    from spec.data import load_attributes
    att = load_attributes(dataset).apply(pd.to_numeric, errors="coerce")
    att = att.loc[:, att.notna().mean() > 0.8].dropna(axis=1, how="all")
    rows, shap_rows = [], []
    for tgt in targets:
        d = summ.dropna(subset=[tgt]).copy()
        d.index = d.basin.astype(str)
        X = att.reindex(d.index).fillna(att.median(numeric_only=True))
        X = X.loc[:, X.std() > 0]
        if X.empty:
            continue
        y = d[tgt].values
        rf = RandomForestRegressor(500, min_samples_leaf=3, n_jobs=-1, random_state=0)
        # 按站点分组: 同一流域不同时进入训练与测试, 否则属性回归的 R2 被泄漏抬高
        cv = cross_val_score(rf, X.values, y, cv=GroupKFold(5), groups=d.index.values, scoring="r2")
        rf.fit(X.values, y)
        pi = permutation_importance(rf, X.values, y, n_repeats=10, random_state=0, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
        rows.append(dict(target=tgt, cv_r2_mean=float(cv.mean()), cv_r2_std=float(cv.std()),
                         top_attributes="; ".join(f"{k}={v:.3f}" for k, v in imp.head(8).items())))
        try:
            import shap
            sv = shap.TreeExplainer(rf).shap_values(X.values, check_additivity=False)
            shap_rows.append(pd.DataFrame(dict(target=tgt, attribute=X.columns,
                                               mean_abs_shap=np.abs(sv).mean(0))))
        except Exception:
            shap_rows.append(pd.DataFrame(dict(target=tgt, attribute=imp.index,
                                               mean_abs_shap=imp.values)))
    return pd.DataFrame(rows), pd.concat(shap_rows) if shap_rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--regime", default="temporal")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip_attributes", action="store_true")
    a = ap.parse_args()

    groups = collect(a.dataset, a.regime)
    print("模型组:", {k: v["n_seeds"] for k, v in groups.items()})
    summaries, curve_store = [], {}
    for name, g in groups.items():
        df, curves = evaluate_run(g["pred"], dict(model=name, dataset=a.dataset, regime=a.regime))
        summaries.append(df); curve_store[name] = curves
        print(f"  {name}: {len(df)} basins, median NSE={df.NSE.median():.3f}, "
              f"A_up={df.A_up.median():.3f}, A_lo={df.A_lo.median():.3f}")
    summ = pd.concat(summaries, ignore_index=True)
    save_table(summ, f"e3_summary_{a.dataset}_{a.regime}.csv")

    models = sorted(groups)
    # ---- (a) F5 头图
    cd_, bd_ = {}, {}
    for name in models:
        cs = curve_store[name]
        Eu = np.vstack([c.E_up for c in cs.values()]); El = np.vstack([c.E_lo for c in cs.values()])
        cd_[name] = (np.nanmedian(Eu, 0), np.nanmedian(El, 0))
        bd_[name] = ((np.nanpercentile(Eu, 25, 0), np.nanpercentile(Eu, 75, 0)),
                     (np.nanpercentile(El, 25, 0), np.nanpercentile(El, 75, 0)))
    sp.plot_models_overlay(cd_, bd_, f"{a.dataset} / {a.regime}: median SPEC error curves (IQR)",
                           FIG / f"F5_headline_{a.dataset}_{a.regime}.png")
    cdb = {}
    for name in models:
        cs = curve_store[name]
        cdb[name] = (np.nanmedian(np.vstack([c.B_up for c in cs.values()]), 0),
                     np.nanmedian(np.vstack([c.B_lo for c in cs.values()]), 0))
    sp.plot_models_overlay(cdb, None, f"{a.dataset} / {a.regime}: median SPEC bias curves",
                           FIG / f"F5b_bias_{a.dataset}_{a.regime}.png", kind="B")

    # ---- (b) 排序翻转
    flips = ranking_flip(summ, models, [("A_up", True), ("A_lo", True), ("E5_up", True),
                                        ("E5_lo", True), ("delta_bar", True)])
    save_table(flips, f"e3_ranking_flips_{a.dataset}_{a.regime}.csv")
    print(flips.groupby("criterion").flip_rate.median().round(3).to_string())

    # ---- (c) 冗余
    conv = ["NSE", "lnNSE", "KGE", "KGEp", "KGEnp", "PBIAS", "RMSE", "RSR", "VE",
            "BiasFHV", "BiasFLV", "BiasFMS", "PeakBias", "Q95Bias", "BFIBias"]
    corr, load, pca = redundancy(summ, conv, CURVE_FEATURES)
    corr.to_csv(RES / f"e3_spearman_{a.dataset}_{a.regime}.csv")
    load.to_csv(RES / f"e3_pca_loadings_{a.dataset}_{a.regime}.csv")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.05, 1]})
    im = axs[0].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    axs[0].set_xticks(range(len(corr))); axs[0].set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    axs[0].set_yticks(range(len(corr))); axs[0].set_yticklabels(corr.index, fontsize=8)
    plt.colorbar(im, ax=axs[0], shrink=.8); axs[0].set_title("Spearman |metric x metric|", fontsize=9)
    axs[1].scatter(load.PC1, load.PC2, s=12, c=["tab:red" if i in CURVE_FEATURES else "tab:blue" for i in load.index])
    for i, r in load.iterrows():
        axs[1].annotate(i, (r.PC1, r.PC2), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axs[1].axhline(0, lw=.5, c="k"); axs[1].axvline(0, lw=.5, c="k")
    axs[1].set_xlabel("PC1"); axs[1].set_ylabel("PC2")
    axs[1].set_title(f"PCA biplot (red = curve features), PC1 var={pca.explained_variance_ratio_[0]:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"F6_redundancy_{a.dataset}_{a.regime}.png"); plt.close(fig)
    print("E5_lo 与传统指标的最大 |Spearman| =",
          round(float(corr.loc["E5_lo", [c for c in conv if c in corr.columns]].abs().max()), 3))

    # ---- (d) Friedman + Nemenyi
    rows = []
    for val in ("A_up", "A_lo"):
        info, avg, cd = friedman_nemenyi(summ, val, models)
        if info:
            rows.append(dict(value=val, **info, ranks=avg.round(2).to_dict(), CD=cd))
            plot_cd(avg, cd, f"Nemenyi CD — {val} ({a.dataset}/{a.regime})",
                    FIG / f"F6b_CD_{val}_{a.dataset}_{a.regime}.png")
    save_table(pd.DataFrame(rows), f"e3_friedman_{a.dataset}_{a.regime}.csv")

    # ---- (e) 属性回归
    if not a.skip_attributes:
        try:
            reg, shap_tab = attribute_regression(summ, a.dataset)
            save_table(reg, f"e3_attribute_rf_{a.dataset}_{a.regime}.csv")
            save_table(shap_tab, f"e3_attribute_shap_{a.dataset}_{a.regime}.csv")
        except Exception as e:
            print("属性回归跳过:", e)

    # ---- T3 主表
    t3 = summ.groupby("model")[["A_up", "A_lo", "E5_up", "E5_lo", "alpha_star_up",
                                "NSE", "KGE", "lnNSE", "PBIAS"]].median().round(3)
    t3.to_csv(RES / f"T3_summary_{a.dataset}_{a.regime}.csv"); print(t3.to_string())


if __name__ == "__main__":
    main()

