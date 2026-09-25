"""
experiments/common.py —— 各实验共用: 运行目录登记、预测读取、逐(站,模型)评价、并行封装
所有实验脚本读取 runs/*/preds_test.parquet, 与训练解耦 —— 训练一次, 分析可反复重跑.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec import spec, ALPHA_GRID, CURVE_FEATURES            # noqa
from spec.io import read_preds, save_preds, preds_path        # noqa
from spec.metrics import all_conventional                    # noqa

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RES = ROOT / "results"
FIG = ROOT / "figures"
for d in (RES, FIG):
    d.mkdir(exist_ok=True, parents=True)


def list_runs(pattern="*"):
    return sorted(p for p in RUNS.glob(pattern) if p.is_dir() and preds_path(p) is not None)


def run_meta(p: Path):
    cfg = json.load(open(p / "config.json")) if (p / "config.json").exists() else {}
    name = p.name
    model = cfg.get("arch", cfg.get("model", name.split("_")[1] if "_" in name else name))
    loss = cfg.get("loss", cfg.get("obj", ""))
    # 概念模型配置使用 obj 字段；对旧配置按运行名兜底，避免 KGE/lnNSE 被合并。
    if not loss:
        if name.endswith("_kge"): loss = "kge"
        elif name.endswith("_lnnse"): loss = "lnnse"
    return dict(run=name, path=str(p),
                dataset=cfg.get("dataset", "unknown"), model=model,
                loss=loss, seed=cfg.get("seed", 0),
                regime=cfg.get("regime", "temporal"))


def load_preds(p: Path) -> pd.DataFrame:
    return read_preds(Path(p)).dropna(subset=["obs"])


def seed_median_predictions(paths):
    """多种子: 逐 (basin,date) 取模拟中位数 —— 曲线报告种子中位数."""
    dfs = [load_preds(p)[["basin", "date", "obs", "sim"]] for p in paths]
    cat = pd.concat(dfs)
    return cat.groupby(["basin", "date"], as_index=False).agg(obs=("obs", "first"), sim=("sim", "median"))


def evaluate_pair(y, yh, with_conventional=True, **kw):
    r = spec(y, yh, **kw)
    out = dict(r["summary"])
    if with_conventional:
        out.update(all_conventional(y, yh))
    return out, r


def evaluate_run(pred: pd.DataFrame, label: dict, min_n=365):
    rows, curves = [], {}
    for b, g in pred.groupby("basin"):
        g = g.sort_values("date").dropna(subset=["sim"])
        if len(g) < min_n:
            continue
        s, r = evaluate_pair(g.obs.values, g.sim.values)
        rows.append(dict(basin=b, **label, **s))
        curves[b] = r["curves"]
    return pd.DataFrame(rows), curves


def pmap(fn, items, workers=None, desc=""):
    workers = workers or min(os.cpu_count(), 16)
    out = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for k, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if k % 20 == 0:
                print(f"  {desc} {k}/{len(items)}", flush=True)
    return out


def save_table(df, name):
    p = RES / name
    df.to_csv(p, index=False)
    print("wrote", p, df.shape)
    return p
