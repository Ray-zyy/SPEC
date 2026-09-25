"""
E5 —— 抽样不确定性与记录长度 (正面回应 Clark et al. 2021)

python experiments/e5_uncertainty.py --dataset camels_aus --runs "aus_lstm_nse_s0,aus_gr4j_kge" \
       --core_file data/processed/core_catchments_aus.csv --B 1000 --workers 16

(a) 块长敏感性 L ∈ {15,30,60,90} + 自适应 L=max(30,3τ); 逐点带与 max-t 同时带宽度 vs alpha
(b) 记录长度: 1/2/3/5/10 年子样本 -> A_up, A_lo, alpha* 的离散度; 给出稳定所需最短年数
(c) 模型差异显著性: 配对 bootstrap (共用随机块) vs 同一 bootstrap 下 NSE 差异显著比例
(d) 观测不确定性: 异方差乘性率定误差传播
产出: results/e5_*.csv, figures/F8_bands_and_recordlength.png
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RUNS, RES, FIG, load_preds, save_table, pmap       # noqa
from spec.core import curves, ALPHA_GRID, curve_area, zero_crossing               # noqa
from spec.bootstrap import (bootstrap_curves, pointwise_band, simultaneous_band,  # noqa
                            paired_difference, choose_block_length)
from spec.perturb import rating_curve_noise                                       # noqa
import spec.plotting as sp                                                        # noqa

BLOCKS = [15, 30, 60, 90, None]     # None = 自适应 max(30, 3*tau)
YEARS = [1, 2, 3, 5, 10]


def task_bands(t):
    basin, run, y, yh, B = t
    rows = []
    for L in BLOCKS:
        boot = bootstrap_curves(y, yh, B=B, L=L)
        for br in ("up", "lo"):
            plo, phi = pointwise_band(boot[f"E_{br}"])
            slo, shi, c = simultaneous_band(boot[f"E_{br}"])
            for k, al in [(0, 0.01), (4, 0.05), (19, 0.20), (99, 1.00)]:
                rows.append(dict(basin=basin, run=run, branch=br, L=(L or boot["L"]),
                                 L_mode=("fixed" if L else "adaptive"), alpha=al,
                                 pointwise_width=float(phi[k] - plo[k]),
                                 simultaneous_width=float(shi[k] - slo[k]), maxt_c=float(c)))
    return pd.DataFrame(rows)


def task_recordlength(t):
    basin, run, y, yh, seed = t
    rng = np.random.default_rng(seed)
    n = len(y); rows = []
    full = curves(y, yh)
    ref = dict(A_up=curve_area(ALPHA_GRID, full.E_up), A_lo=curve_area(ALPHA_GRID, full.E_lo),
               E5_up=full.E_up[4], E5_lo=full.E_lo[4], alpha_star=zero_crossing(ALPHA_GRID, full.B_up))
    for yr in YEARS:
        L = yr * 365
        if L >= n:
            continue
        for rep in range(20):
            s = int(rng.integers(0, n - L))
            c = curves(y[s:s + L], yh[s:s + L])
            rows.append(dict(basin=basin, run=run, years=yr, rep=rep,
                             A_up=curve_area(ALPHA_GRID, c.E_up), A_lo=curve_area(ALPHA_GRID, c.E_lo),
                             E5_up=c.E_up[4], E5_lo=c.E_lo[4],
                             alpha_star=zero_crossing(ALPHA_GRID, c.B_up),
                             **{f"ref_{k}": v for k, v in ref.items()}))
    return pd.DataFrame(rows)


def task_rating(t):
    basin, run, y, yh, R, seed = t
    rng = np.random.default_rng(seed)
    base = curves(y, yh)
    A_up, A_lo, ast = [], [], []
    for _ in range(R):
        yp = rating_curve_noise(y, rng)
        c = curves(yp, yh)
        A_up.append(curve_area(ALPHA_GRID, c.E_up)); A_lo.append(curve_area(ALPHA_GRID, c.E_lo))
        ast.append(zero_crossing(ALPHA_GRID, c.B_up))
    return pd.DataFrame([dict(basin=basin, run=run,
                              A_up=curve_area(ALPHA_GRID, base.E_up),
                              A_up_p05=np.nanpercentile(A_up, 5), A_up_p95=np.nanpercentile(A_up, 95),
                              A_lo=curve_area(ALPHA_GRID, base.E_lo),
                              A_lo_p05=np.nanpercentile(A_lo, 5), A_lo_p95=np.nanpercentile(A_lo, 95),
                              alpha_star=zero_crossing(ALPHA_GRID, base.B_up),
                              alpha_star_p05=np.nanpercentile(ast, 5), alpha_star_p95=np.nanpercentile(ast, 95))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--core_file", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    core = [str(b) for b in pd.read_csv(a.core_file, dtype={"basin": str}).basin]
    runs = a.runs.split(",")
    data = {}
    for r in runs:
        p = load_preds(RUNS / r); p["basin"] = p.basin.astype(str)
        for b in core:
            g = p[p.basin == b].sort_values("date").dropna(subset=["sim"])
            if len(g) > 365:
                data[(b, r)] = (g.obs.values, g.sim.values)

    # (a)
    bands = pd.concat(pmap(task_bands, [(b, r, y, yh, a.B) for (b, r), (y, yh) in data.items()],
                           a.workers, "E5a"))
    save_table(bands, "e5_a_band_width.csv")
    # (b)
    rl = pd.concat(pmap(task_recordlength, [(b, r, y, yh, i) for i, ((b, r), (y, yh)) in enumerate(data.items())],
                        a.workers, "E5b"))
    save_table(rl, "e5_b_record_length.csv")
    conv = (rl.assign(err_A_up=(rl.A_up - rl.ref_A_up).abs(),
                      err_E5_up=(rl.E5_up - rl.ref_E5_up).abs(),
                      err_E5_lo=(rl.E5_lo - rl.ref_E5_lo).abs())
            .groupby("years")[["err_A_up", "err_E5_up", "err_E5_lo"]]
            .quantile(0.9).reset_index())
    save_table(conv, "e5_b_convergence.csv")
    tol = 0.05
    ok = conv[conv.err_E5_up <= tol]
    print(f"→ 使 E^up(5%) 的 90% 误差 ≤ {tol} 所需最短记录: "
          f"{int(ok.years.min()) if len(ok) else '>10'} 年")

    # (c) 配对 bootstrap
    rows = []
    for b in core:
        for r1, r2 in itertools.combinations(runs, 2):
            if (b, r1) not in data or (b, r2) not in data:
                continue
            y1, s1 = data[(b, r1)]; y2, s2 = data[(b, r2)]
            n = min(len(y1), len(y2))
            res = paired_difference(y1[:n], s1[:n], s2[:n], B=min(a.B, 500))
            for br in ("E_up", "E_lo"):
                ez = res[br]["excludes_zero"]
                rows.append(dict(basin=b, model_a=r1, model_b=r2, branch=br,
                                 frac_alpha_significant=float(np.nanmean(ez)),
                                 sig_at_5pct=bool(ez[4]), sig_at_100pct=bool(ez[-1]),
                                 NSE_diff_significant=res["NSE_diff_significant"]))
    sig = pd.DataFrame(rows); save_table(sig, "e5_c_paired_significance.csv")
    if len(sig):
        print("曲线差异显著(α=5%)的站点比例:", round(sig[sig.branch == 'E_up'].sig_at_5pct.mean(), 3),
              " NSE 差异显著比例:", round(sig.NSE_diff_significant.mean(), 3))
    # (d)
    rat = pd.concat(pmap(task_rating, [(b, r, y, yh, 200, i) for i, ((b, r), (y, yh)) in enumerate(data.items())],
                         a.workers, "E5d"))
    save_table(rat, "e5_d_rating_uncertainty.csv")

    # ---- F8
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.1))
    ex = list(data)[0]; y, yh = data[ex]
    boot = bootstrap_curves(y, yh, B=a.B)
    c = curves(y, yh)
    sp.plot_dual(axs[0], ALPHA_GRID, c.E_up, c.E_lo, sp.PALETTE[0],
                 band_up=pointwise_band(boot["E_up"]), band_lo=pointwise_band(boot["E_lo"]))
    slo, shi, _ = simultaneous_band(boot["E_up"])
    axs[0].plot(ALPHA_GRID * 100, slo, ":", c=sp.PALETTE[0], lw=.8)
    axs[0].plot(ALPHA_GRID * 100, shi, ":", c=sp.PALETTE[0], lw=.8)
    sp.style_error_axis(axs[0], f"{ex[0]} / {ex[1]}: 95% pointwise (shaded) and simultaneous (dotted)")
    for L, sub in bands[bands.branch == "up"].groupby("L"):
        m = sub.groupby("alpha").pointwise_width.median()
        axs[1].plot(m.index * 100, m.values, marker="o", ms=3, label=f"L={L}")
    axs[1].set_xscale("log"); axs[1].set_xlabel(r"$\alpha$ (%)"); axs[1].set_ylabel("band width")
    axs[1].legend(fontsize=6, frameon=False); axs[1].set_title("(a) block-length sensitivity", fontsize=9)
    for col, lab in [("err_E5_up", r"$E^{\uparrow}_5$"), ("err_E5_lo", r"$E^{\downarrow}_5$"), ("err_A_up", r"$A^{\uparrow}$")]:
        axs[2].plot(conv.years, conv[col], marker="s", ms=3, label=lab)
    axs[2].axhline(tol, ls=":", c="k"); axs[2].set_xlabel("record length (years)")
    axs[2].set_ylabel("90th pct |error|"); axs[2].legend(fontsize=7, frameon=False)
    axs[2].set_title("(b) record-length convergence", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "F8_bands_and_recordlength.png"); plt.close(fig)
    print("E5 完成")


if __name__ == "__main__":
    main()
