"""
E7 —— 稳健性与敏感性 (补充材料表)

python experiments/e7_robustness.py --dataset camels_aus --runs "aus_lstm_nse_s0,aus_gr4j_kge" \
       --basins_file data/processed/basins_aus.txt --workers 16

(a) alpha 网格 1% / 0.5% / 5%;  (b) 并列值种子;  (c) 零流量约定;
(d) 按观测/模拟/均值排序;      (e)(f) 时间步聚合 (日/3日/月);
(g) 跨数据集迁移 (对 camels_gb 或 lamah 重跑 E3 头图, 由 --dataset 控制);
(h) 计算耗时表.
产出: results/e7_*.csv
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RUNS, RES, load_preds, save_table, pmap     # noqa
from spec.core import curves, spec, curve_area, zero_crossing              # noqa
from spec.bootstrap import bootstrap_curves                                # noqa


def grids():
    return {"1pct": np.round(np.arange(1, 101) / 100, 4),
            "0.5pct": np.round(np.arange(0.5, 100.01, 0.5) / 100, 4),
            "5pct": np.round(np.arange(5, 101, 5) / 100, 4)}


def one(t):
    basin, run, y, yh = t
    rows = []
    def rec(setting, value, c, al):
        rows.append(dict(basin=basin, run=run, setting=setting, value=value,
                         A_up=curve_area(al, c.E_up), A_lo=curve_area(al, c.E_lo),
                         E5_up=float(c.E_up[np.argmin(np.abs(al - .05))]),
                         E5_lo=float(c.E_lo[np.argmin(np.abs(al - .05))]),
                         alpha_star=zero_crossing(al, c.B_up)))
    for gname, g in grids().items():
        rec("alpha_grid", gname, curves(y, yh, alphas=g), g)
    a1 = grids()["1pct"]
    for seed in range(5):
        rec("tie_seed", str(seed), curves(y, yh, tie_seed=seed), a1)
    for rule in ("delta0", "exclude"):
        rec("zero_rule", rule, curves(y, yh, zero_rule=rule), a1)
    for rk in ("obs", "sim", "mean"):
        rec("rank_by", rk, curves(y, yh, rank_by=rk), a1)
    for agg, k in (("daily", 1), ("3day", 3), ("monthly", 30)):
        if k == 1:
            rec("aggregation", agg, curves(y, yh), a1)
        else:
            n = (len(y) // k) * k
            ya = y[:n].reshape(-1, k).mean(1); yha = yh[:n].reshape(-1, k).mean(1)
            rec("aggregation", agg, curves(ya, yha), a1)
    for m in (10, 30, 50):
        rec("min_m", str(m), curves(y, yh, min_m=m), a1)
    return pd.DataFrame(rows)


def timing(y, yh):
    rows = []
    t0 = time.perf_counter(); [curves(y, yh) for _ in range(100)]
    rows.append(dict(op="curves x100", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter(); spec(y, yh)
    rows.append(dict(op="spec() full (density+PELT+entropy)", seconds=time.perf_counter() - t0))
    t0 = time.perf_counter(); bootstrap_curves(y, yh, B=100)
    rows.append(dict(op="bootstrap B=100", seconds=time.perf_counter() - t0))
    return pd.DataFrame(rows).assign(n=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--basins_file", default=None); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_basins", type=int, default=60)
    a = ap.parse_args()
    keep = set(l.strip() for l in open(a.basins_file)) if a.basins_file else None
    tasks = []
    for run in a.runs.split(","):
        p = load_preds(RUNS / run); p["basin"] = p.basin.astype(str)
        bs = [b for b in p.basin.unique() if (keep is None or b in keep)][: a.max_basins]
        for b in bs:
            g = p[p.basin == b].sort_values("date").dropna(subset=["sim"])
            if len(g) > 365:
                tasks.append((b, run, g.obs.values, g.sim.values))
    df = pd.concat(pmap(one, tasks, a.workers, "E7"))
    save_table(df, f"e7_sensitivity_{a.dataset}.csv")
    ref = df[(df.setting == "alpha_grid") & (df.value == "1pct")].set_index(["basin", "run"])
    rows = []
    for (setting, value), sub in df.groupby(["setting", "value"]):
        s = sub.set_index(["basin", "run"])
        j = s.join(ref, rsuffix="_ref").dropna(subset=["A_up", "A_up_ref"])
        rows.append(dict(setting=setting, value=value, n=len(j),
                         dA_up_median=float((j.A_up - j.A_up_ref).abs().median()),
                         dE5_up_median=float((j.E5_up - j.E5_up_ref).abs().median()),
                         dE5_lo_median=float((j.E5_lo - j.E5_lo_ref).abs().median()),
                         dalpha_star_median=float((j.alpha_star - j.alpha_star_ref).abs().median())))
    save_table(pd.DataFrame(rows), f"e7_sensitivity_summary_{a.dataset}.csv")
    y, yh = tasks[0][2], tasks[0][3]
    save_table(timing(y, yh), "e7_h_timing.csv")
    print("E7 完成")


if __name__ == "__main__":
    main()
