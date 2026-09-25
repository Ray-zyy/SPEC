"""
数据准备 (CAMELS-AUS, 一次性, 纯 CPU, 约 2-5 分钟)

  python scripts/prepare_data.py

产出 (全部在 data/processed/):
  camels_aus_all.pkl        全部站点的日序缓存 (训练读取用, ~120 MB)
  qc_aus.csv                质控明细 (论文要报告保留站数与剔除原因)
  basins_aus.txt            质控后保留的站点列表 (后续所有脚本的 --basins_file)
  attributes_aus.csv        派生属性 (clim_* 由 forcing 导出, hyd_* 由实测 q 导出; 仅率定期 1980-2004)
  attributes_climate_aus.csv 只含 clim_*, 训练时的静态输入 (PUB 也合法)
  signatures_aus.csv        分层用签名
  pub_folds_aus.csv         留区域 5 折 (按 AWRC drainage division 分组)
  core_catchments_aus.csv   T2 核心站点 (四类各 n_per_class 个)
  figures/F0_stratification.png  站点分层散点 (干旱指数-BFI-零流量), 论文可放补充材料
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from spec.data import (PROC, build_cache, compute_attributes, qc_basins, make_pub_folds,
                       catchment_signatures, select_core_catchments, basin_region)

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_class", type=int, default=4)
ap.add_argument("--k_folds", type=int, default=5)
ap.add_argument("--min_test_nonzero", type=float, default=0.02,
                help="测试期非零流量日比例下限; 低于此值的站无法定义 FDC/lnNSE")
ap.add_argument("--force_cache", action="store_true")
a = ap.parse_args()

PROC.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 0. 缓存
build_cache(force=a.force_cache)

# ---------------------------------------------------------------- 1. 质控
basins, qc = qc_basins(min_test_nonzero=a.min_test_nonzero)
qc.to_csv(PROC / "qc_aus.csv", index=False)
Path(PROC / "basins_aus.txt").write_text("\n".join(map(str, basins)))

# ---------------------------------------------------------------- 2. 属性 (只用率定期)
att = compute_attributes(basins=basins)
att.to_csv(PROC / "attributes_aus.csv")
att[[c for c in att.columns if c.startswith("clim_")]].to_csv(PROC / "attributes_climate_aus.csv")

# ---------------------------------------------------------------- 3. PUB 空间折
folds = make_pub_folds(basins, k=a.k_folds)
folds.to_csv(PROC / "pub_folds_aus.csv", index=False)

# ---------------------------------------------------------------- 4. 签名与核心站点
sig = catchment_signatures(basins=basins)
sig.to_csv(PROC / "signatures_aus.csv")
core = select_core_catchments(sig, a.n_per_class)
core.to_csv(PROC / "core_catchments_aus.csv")

# ---------------------------------------------------------------- 5. 概览图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = Path(__file__).resolve().parents[1] / "DejaVuMathTeXGyre.ttf"
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
sc = axs[0].scatter(sig.aridity_PET_P, sig.BFI, c=sig.zero_frac, s=14, cmap="viridis")
axs[0].set_xlabel("aridity  PET/P"); axs[0].set_ylabel("Eckhardt BFI")
plt.colorbar(sc, ax=axs[0], label="zero-flow fraction")
axs[1].scatter(sig.aridity_PET_P, sig.RB_flashiness, s=14, c="grey")
axs[1].scatter(sig.loc[core.index, "aridity_PET_P"], sig.loc[core.index, "RB_flashiness"],
               s=40, c="crimson", label="core catchments")
axs[1].set_xlabel("aridity  PET/P"); axs[1].set_ylabel("RB flashiness"); axs[1].legend(fontsize=7, frameon=False)
axs[2].bar(folds.groupby("fold").size().index.astype(str), folds.groupby("fold").size().values)
axs[2].set_xlabel("PUB fold"); axs[2].set_ylabel("# basins")
fig.tight_layout(); fig.savefig(FIG / "F0_stratification.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 6. 汇总打印
print("\n================ 数据准备结果 ================")
print(f"保留站点      : {len(basins)} / {len(qc)}")
print(f"剔除原因分布  : {qc[~qc.keep].reason.value_counts().to_dict()}")
print(f"drainage div. : {pd.Series([basin_region(b) for b in basins]).value_counts().to_dict()}")
print(f"PUB 折大小    : {folds.groupby('fold').size().to_dict()}")
print(f"零流量比例    : 中位 {sig.zero_frac.median():.3f}, >0.5 的站 {(sig.zero_frac > .5).sum()} 个")
print(f"干旱指数      : 中位 {sig.aridity_PET_P.median():.2f}  [{sig.aridity_PET_P.min():.2f}, {sig.aridity_PET_P.max():.2f}]")
print("\n核心站点 (论文 T2):")
print(core[["stratum", "aridity_PET_P", "BFI", "zero_frac", "RB_flashiness", "region"]].round(3).to_string())
print("\n产出目录:", PROC)
