"""
tests/make_fake_runs.py —— 用合成数据伪造 runs/*/preds_test.parquet, 端到端跑通 E2/E3/E5/E7/E8
真实数据到位前先跑这个, 确认整条分析链没有 bug.

python tests/make_fake_runs.py && bash tests/smoke_test.sh
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"
from spec.io import save_preds

rng = np.random.default_rng(0)
n_basins, n_days = 24, 3650
dates = pd.date_range("2005-01-01", periods=n_days, freq="D")
basins = [f"F{i:03d}" for i in range(n_basins)]

obs = {}
for i, b in enumerate(basins):
    base = np.abs(rng.lognormal(0, 1.0 + 0.3 * rng.random(), n_days)) * np.exp(0.7 * np.sin(np.arange(n_days) / 58 + i))
    if i % 3 == 0:                                   # 三分之一为间歇性河流 (含零流量)
        base[base < np.quantile(base, 0.18)] = 0.0
    obs[b] = base

MODELS = {
    "lstm":  dict(peak=0.75, base=+0.12, noise=0.20),   # 削峰 + 基流抬升 (典型深度模型)
    "gru":   dict(peak=0.80, base=+0.08, noise=0.22),
    "gr4j":  dict(peak=0.95, base=-0.20, noise=0.35),   # 峰值好, 基流偏低
    "hbv":   dict(peak=0.88, base=-0.05, noise=0.30),
}


def simulate(y, peak, base, noise, rng):
    q90, q70, q50 = np.quantile(y, .9), np.quantile(y, .7), np.median(y)
    yh = np.where(y >= q90, q90 + peak * (y - q90), y)
    yh = np.where(y < q70, np.clip(yh + base * q50, 0, None), yh)
    return np.clip(yh * np.exp(rng.normal(0, noise, len(y))), 0, None)


def write_run(name, cfg, sim_fn):
    d = RUNS / name; d.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in basins:
        y = obs[b]
        rows.append(pd.DataFrame(dict(basin=b, date=dates, obs=y, sim=sim_fn(y))))
    save_preds(pd.concat(rows), d)
    json.dump(cfg, open(d / "config.json", "w"))


for arch, par in MODELS.items():
    for seed in range(3):
        r = np.random.default_rng(100 + seed)
        write_run(f"fake_{arch}_nse_s{seed}",
                  dict(dataset="fake", arch=arch, loss="nse", seed=seed, regime="temporal"),
                  lambda y, p=par, r=r: simulate(y, p["peak"], p["base"], p["noise"], r))

# E6: 五种损失 (delta-loss 剖面最平)
LOSS_PAR = {"mse": (0.72, +0.15, 0.20), "mae": (0.80, +0.08, 0.22), "nse": (0.75, +0.12, 0.20),
            "logmse": (0.92, -0.02, 0.30), "delta": (0.85, +0.02, 0.24)}
for loss, p in LOSS_PAR.items():
    for seed in range(2):
        r = np.random.default_rng(200 + seed)
        write_run(f"fake_lstm_{loss}_s{seed}",
                  dict(dataset="fake", arch="lstm", loss=loss, seed=seed, regime="temporal"),
                  lambda y, p=p, r=r: simulate(y, *p, rng=r))

# 基准 (SPECSS 分母)
write_run("fake_climatology", dict(dataset="fake", model="climatology", regime="temporal"),
          lambda y: np.full_like(y, np.mean(y)))
write_run("fake_persistence", dict(dataset="fake", model="persistence", regime="temporal"),
          lambda y: np.concatenate([[y[0]], y[:-1]]))

pd.DataFrame(dict(basin=basins, stratum=["test"] * len(basins))).to_csv(
    ROOT / "data/processed/core_catchments_fake.csv", index=False)
Path(ROOT / "data/processed/basins_fake.txt").write_text("\n".join(basins))
print("伪造运行写入", RUNS, "共", len(list(RUNS.glob('fake_*'))), "个")
