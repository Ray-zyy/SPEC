"""
E4 —— 核心站点诊断案例 (五联图 + 逐段解读模板自动生成)

python experiments/e4_case_studies.py --dataset camels_aus --runs "aus_lstm_nse_s0,aus_gr4j_kge,aus_gr4j_lnnse" \
       --core_file data/processed/core_catchments_aus.csv --B 1000

产出:
  figures/F7_<basin>_<model>.png          五联图
  results/e4_readings.md                  四段 × 四问 的逐段解读草稿 (供你改写进论文)
  results/e4_crosscheck.csv               每条结论与独立签名 (%BiasFHV / BFI偏差 / Q95偏差 / 退水常数 k) 的一致性
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RUNS, RES, FIG, load_preds, save_table          # noqa
from spec import spec                                                          # noqa
from spec.bootstrap import bootstrap_curves, simultaneous_band                 # noqa
from spec.metrics import (all_conventional, bias_fhv, bfi_bias, quantile_bias,  # noqa
                          recession_constant, eckhardt_bfi)
import spec.plotting as sp                                                     # noqa

SEGMENTS = [(0.01, 0.05, "极端事件"), (0.05, 0.20, "场次峰"),
            (0.20, 0.70, "中流与涨落段"), (0.70, 1.00, "退水/基流")]


def segment_reading(res, boot, branch="up"):
    """四段 × 四问: ①误差水平 ②偏差符号及显著性 ③熵指示 ④隐含缺陷"""
    c = res["curves"]; a = c.alphas
    E = getattr(c, f"E_{branch}"); B = getattr(c, f"B_{branch}")
    e = res["density"][f"e_{branch}"]; H = res["entropy"][f"H_{branch}"]
    lo, hi, _ = simultaneous_band(boot[f"B_{branch}"])
    out = []
    for a0, a1, label in SEGMENTS:
        m = (a > a0 - 1e-9) & (a <= a1 + 1e-9)
        if not np.isfinite(E[m]).any():
            continue
        Em, Bm, Hm, em = (np.nanmean(x[m]) for x in (E, B, H, e))
        sig_pos = bool(np.all(lo[m] > 0)); sig_neg = bool(np.all(hi[m] < 0))
        sign = "显著高估" if sig_pos else ("显著低估" if sig_neg else "偏差不显著")
        struct = "结构性(低熵)" if Hm < 0.5 else ("随机性(高熵)" if Hm > 0.8 else "混合")
        if branch == "up":
            defect = ("汇流/演算存储过度阻尼, 峰值被削平" if (sig_neg and Hm < 0.6) else
                      "峰值系统性偏高, 产流阈值或快速径流分配过大" if sig_pos else
                      "峰值误差以随机成分为主, 可能受降水输入或率定曲线限制")
        else:
            defect = ("缺乏深层地下水补给机制, 干期流量被高估(常伴随干日误报)" if sig_pos else
                      "退水过快/低流量被压低, 慢速蓄水库参数偏小" if sig_neg else
                      "低流量偏差无系统方向")
        out.append(dict(branch=branch, segment=f"[{a0:.0%},{a1:.0%}] {label}",
                        E_mean=round(float(Em), 3), e_mean=round(float(em), 3),
                        equiv_ratio=round(float((1 + Em) / (1 - Em)), 2),   # delta -> rho 换算
                        B_mean=round(float(Bm), 3), significance=sign,
                        H_mean=round(float(Hm), 3), entropy_reading=struct, implied_defect=defect))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--core_file", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()

    core = pd.read_csv(a.core_file, dtype={"basin": str}).set_index("basin")
    md, cross = [], []
    for run in a.runs.split(","):
        pred = load_preds(RUNS / run)
        pred["basin"] = pred["basin"].astype(str)
        for b in core.index:
            g = pred[pred.basin == b].sort_values("date").dropna(subset=["sim"])
            if len(g) < 365:
                continue
            y, yh, dates = g.obs.values, g.sim.values, g.date.values
            res = spec(y, yh)
            boot = bootstrap_curves(y, yh, B=a.B)
            sp.plot_five_panel(dates, y, yh, res, boot, run, b, FIG / f"F7_{b}_{run}.png")
            rd = segment_reading(res, boot, "up") + segment_reading(res, boot, "lo")
            md.append(f"\n## {b} ({core.loc[b].get('stratum','')}) — {run}\n")
            md.append(f"n={res['summary']['n']}, zero_frac={res['summary']['zero_frac']:.1%}, "
                      f"A↑={res['summary']['A_up']:.3f}, A↓={res['summary']['A_lo']:.3f}, "
                      f"α*={res['summary']['alpha_star_up']}\n")
            md.append(pd.DataFrame(rd).to_markdown(index=False))
            conv = all_conventional(y, yh)
            cross.append(dict(basin=b, run=run,
                              B5_up=res["summary"]["B5_up"], BiasFHV=conv["BiasFHV"],
                              agree_peak=np.sign(res["summary"]["B5_up"]) == np.sign(conv["BiasFHV"]),
                              B5_lo=res["summary"]["B5_lo"], Q95Bias=conv["Q95Bias"],
                              agree_low=np.sign(res["summary"]["B5_lo"]) == np.sign(conv["Q95Bias"]),
                              BFIBias=conv["BFIBias"],
                              k_obs=recession_constant(y), k_sim=recession_constant(yh),
                              alpha_star=res["summary"]["alpha_star_up"], PBIAS=conv["PBIAS"]))
            print("done", b, run)
    (RES / "e4_readings.md").write_text("\n".join(md), encoding="utf-8")
    ct = pd.DataFrame(cross); save_table(ct, "e4_crosscheck.csv")
    print("交叉印证一致率: 峰值支", round(ct.agree_peak.mean(), 3), " 基流支", round(ct.agree_low.mean(), 3))


if __name__ == "__main__":
    main()
