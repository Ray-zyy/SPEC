"""
spec.data —— CAMELS-AUS (本地实际目录布局) 读取 / 质控 / 派生属性 / 空间分折 / 分层抽样

本文件针对你机器上的真实目录重写 (原版按 Zenodo 宽表写, 与本地不符):

    $SPEC_DATA  (默认 <project>/data)
      CAMELS-AUS/
        forcing/<basin>.csv    date,prcp,srad,tmax,tmin,vprp,aet,pet   (13879 行, 1977-01-01..2014-12-31)
        runoff/<basin>.csv     date,runoff                             (同上, mm/day)
      processed/               本模块与 scripts/prepare_data.py 的产出

统一接口 (与原实验脚本兼容, dataset 参数接受 'camels_aus'/'aus'/None):
    list_basins(dataset)              -> [str]
    load_basin(dataset, basin)        -> DataFrame(index=date, cols=[prcp,tmax,tmin,srad,vp,pet,aet,q])
    slice_period(df, dataset, split)  -> DataFrame
    load_attributes(dataset, kind)    -> DataFrame(index=basin)   kind in {climate, hydro, all}

设计要点 (论文 implementation 段必须写):
  * 属性一律只用 **率定期 (train+val, 1980-2004)** 计算, 测试期 2005-2014 绝不参与, 避免信息泄漏;
  * 属性分两类: clim_* 只由 forcing 导出 (未测流域也可得, 因此 PUB 制度下可入模);
    hyd_*  由实测 q 导出 (未测流域不可得), 只用于分层抽样 / T2 / E3 事后属性回归, 不入模;
  * PUB 折按 AWRC 站号的 drainage division (首位数字, 字母前缀站另计) 分组,
    同一 division 的站不跨折 —— 这是空间外推, 不是随机划分.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SPEC_DATA", Path(__file__).resolve().parents[1] / "data"))
PROC = ROOT / "processed"

# 1977-1979 作为暖机期 (深度模型 365 天输入窗口 / 概念模型状态起转), 不进入任何评价
SPLITS = {
    "camels_aus": dict(
        warmup=("1977-01-01", "1979-12-31"),
        train=("1980-01-01", "1999-12-31"),
        val=("2000-01-01", "2004-12-31"),
        test=("2005-01-01", "2014-12-31"),
    )
}
SPLITS["aus"] = SPLITS["camels_aus"]

FORCING_COLS = ["prcp", "tmax", "tmin", "srad", "vp", "pet"]
RENAME = {"vprp": "vp", "runoff": "q"}


# --------------------------------------------------------------------------- 路径
def _norm(dataset):
    if dataset in (None, "aus", "camels_aus", "CAMELS-AUS"):
        return "camels_aus"
    raise ValueError(f"本方案只支持 CAMELS-AUS, 收到 dataset={dataset!r}")


def aus_dir() -> Path:
    """兼容 CAMELS-AUS / CAMELS_AUS / camels_aus 三种写法."""
    for name in ("CAMELS-AUS", "CAMELS_AUS", "camels_aus", "camels-aus"):
        p = ROOT / name
        if (p / "forcing").is_dir() and (p / "runoff").is_dir():
            return p
    raise FileNotFoundError(
        f"在 {ROOT} 下找不到含 forcing/ 与 runoff/ 的 CAMELS-AUS 目录。"
        f" 用环境变量 SPEC_DATA 指定数据根, 例如 export SPEC_DATA=/path/to/spec_data"
    )


# --------------------------------------------------------------------------- 读取
def list_basins(dataset="camels_aus"):
    _norm(dataset)
    d = aus_dir()
    f = {p.stem for p in (d / "forcing").glob("*.csv")}
    r = {p.stem for p in (d / "runoff").glob("*.csv")}
    return sorted(f & r)


@lru_cache(maxsize=512)
def _read_one(basin: str) -> pd.DataFrame:
    d = aus_dir()
    fo = pd.read_csv(d / "forcing" / f"{basin}.csv", parse_dates=["date"]).set_index("date")
    ru = pd.read_csv(d / "runoff" / f"{basin}.csv", parse_dates=["date"]).set_index("date")
    df = fo.join(ru, how="outer").rename(columns=RENAME)
    df = df.replace([-99.99, -99.9, -999.0, -9999.0], np.nan)
    if "q" in df.columns:
        df.loc[df["q"] < 0, "q"] = np.nan          # 负流量视为缺测 (本地为 0 条, 保险起见)
    for c in FORCING_COLS:
        if c not in df.columns:
            df[c] = np.nan
    cols = [c for c in ["prcp", "tmax", "tmin", "srad", "vp", "pet", "aet", "q"] if c in df.columns]
    return df[cols].astype("float64").sort_index()


def load_basin(dataset, basin):
    _norm(dataset)
    return _read_one(str(basin))


def slice_period(df, dataset, split):
    a, b = SPLITS[_norm(dataset)][split]
    return df.loc[a:b]


def calib_period(df, dataset="camels_aus"):
    """率定期 = train + val (1980-01-01 .. 2004-12-31); 所有属性只在此期计算."""
    d = SPLITS[_norm(dataset)]
    return df.loc[d["train"][0]: d["val"][1]]


def build_cache(dataset="camels_aus", basins=None, force=False) -> Path:
    """把全部站点打包成一个 pickle, 训练时一次读入, 免去 222 次 CSV 解析."""
    _norm(dataset)
    PROC.mkdir(parents=True, exist_ok=True)
    f = PROC / "camels_aus_all.pkl"
    if f.exists() and not force:
        return f
    basins = basins or list_basins(dataset)
    dat = {b: _read_one(b).astype("float32") for b in basins}
    pd.to_pickle(dat, f)
    print(f"[cache] {f}  ({len(dat)} 站)")
    return f


@lru_cache(maxsize=1)
def load_cache():
    f = PROC / "camels_aus_all.pkl"
    return pd.read_pickle(f) if f.exists() else None


def load_basin_fast(basin):
    """训练用: 优先走 pickle 缓存."""
    c = load_cache()
    if c is not None and basin in c:
        return c[basin]
    return _read_one(basin)


# --------------------------------------------------------------------------- 站号 -> 空间分组
def basin_region(basin: str) -> str:
    """
    AWRC 站号: 102101A -> drainage division '1'; G9030250 -> 'G9'; A5030502 -> 'A5'.
    返回 drainage division 级区域码 (本数据集共 13 个), 作为 PUB 的空间分组单元.
    """
    b = str(basin)
    m = re.match(r"^([A-Za-z]*)(\d+)", b)
    if not m:
        return "X"
    pre, num = m.group(1), m.group(2)
    return (pre + num[:1]) if pre else num[:1]


def basin_river_basin(basin: str) -> str:
    """前三位数字 = river basin (本数据集 102 个), 更细的空间单元, 备用."""
    b = str(basin)
    m = re.match(r"^([A-Za-z]*)(\d+)", b)
    return (m.group(1) + m.group(2)[:3]) if m else "X"


def make_pub_folds(basins, k=5, seed=0) -> pd.DataFrame:
    """
    留区域交叉验证: 按 drainage division 贪心均衡装箱到 k 折, 同一 division 不跨折.
    返回 DataFrame(basin, region, river_basin, fold).
    """
    reg = pd.Series({b: basin_region(b) for b in basins})
    sizes = reg.value_counts()
    rng = np.random.default_rng(seed)
    order = list(sizes.index)
    rng.shuffle(order)
    order = sorted(order, key=lambda r: -sizes[r])          # 大区域先分, 贪心装箱
    load, assign = {i: 0 for i in range(k)}, {}
    for r in order:
        i = min(load, key=lambda j: (load[j], j))
        assign[r] = i
        load[i] += int(sizes[r])
    rows = [dict(basin=b, region=reg[b], river_basin=basin_river_basin(b), fold=assign[reg[b]])
            for b in basins]
    return pd.DataFrame(rows).sort_values("basin").reset_index(drop=True)


# --------------------------------------------------------------------------- 质控
def qc_basins(dataset="camels_aus", min_years=20, max_missing=0.05,
              min_test_nonzero=0.02, verbose=True):
    """
    保留条件 (论文要写清楚):
      (1) 有效记录 >= min_years 年;
      (2) 测试期 q 缺测比例 < max_missing;
      (3) 测试期非零流量日比例 >= min_test_nonzero —— 否则 FDC / lnNSE / 低流量支无从定义;
      (4) 序列非常数.
    返回 (保留站列表, 质控明细表).
    """
    dataset = _norm(dataset)
    rows = []
    for b in list_basins(dataset):
        try:
            df = load_basin(dataset, b)
        except Exception as e:
            rows.append(dict(basin=b, keep=False, reason=f"load_error:{e}"))
            continue
        q = df["q"]
        test = slice_period(df, dataset, "test")["q"]
        full = q.dropna()
        yrs = (full.index.max() - full.index.min()).days / 365.25 if len(full) else 0.0
        miss = float(test.isna().mean()) if len(test) else 1.0
        nz = float((test.dropna() > 0).mean()) if test.notna().any() else 0.0
        const = bool(full.nunique() <= 2) if len(full) else True
        keep = (yrs >= min_years) and (miss < max_missing) and (nz >= min_test_nonzero) and not const
        reason = ";".join(x for x in [
            "" if yrs >= min_years else "short_record",
            "" if miss < max_missing else "test_missing",
            "" if nz >= min_test_nonzero else "test_almost_all_zero",
            "constant" if const else "",
        ] if x)
        rows.append(dict(basin=b, region=basin_region(b), years=round(yrs, 1),
                         test_missing=round(miss, 4), test_nonzero_frac=round(nz, 4),
                         zero_frac_full=round(float((full == 0).mean()), 4) if len(full) else np.nan,
                         keep=keep, reason=reason))
    tab = pd.DataFrame(rows)
    if verbose:
        print(f"CAMELS-AUS 质控: 保留 {int(tab['keep'].sum())} / {len(tab)} 站")
        bad = tab[~tab.keep]
        if len(bad):
            print(bad[["basin", "reason"]].to_string(index=False))
    return tab[tab.keep].basin.tolist(), tab


# --------------------------------------------------------------------------- 派生属性
def _harmonic(series: pd.Series):
    """对日序列拟合年周期一次谐波, 返回 (相对振幅, 相位 rad)."""
    x = series.values.astype(float)
    ok = np.isfinite(x)
    if ok.sum() < 365:
        return np.nan, np.nan
    t = np.asarray(series.index.dayofyear)[ok].astype(float)
    w = 2 * np.pi * t / 365.25
    A = np.column_stack([np.ones(ok.sum()), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(A, x[ok], rcond=None)
    c0, cs, cc = coef
    amp = np.hypot(cs, cc)
    return (float(amp / abs(c0)) if abs(c0) > 1e-9 else np.nan), float(np.arctan2(cc, cs))


def _mean_spell(mask) -> float:
    v = np.asarray(mask).astype(int)
    if v.sum() == 0:
        return 0.0
    d = np.diff(np.concatenate([[0], v, [0]]))
    return float(np.mean(np.where(d == -1)[0] - np.where(d == 1)[0]))


def climate_attributes(df: pd.DataFrame) -> dict:
    """只用 forcing 导出 —— 未测流域同样可得, 因此 PUB 制度下可作模型输入."""
    p, pet = df["prcp"], df["pet"]
    tmean = (df["tmax"] + df["tmin"]) / 2.0
    pm, petm = float(np.nanmean(p)), float(np.nanmean(pet))
    sp, php = _harmonic(p)
    st, pht = _harmonic(tmean)
    seas = float(sp * np.sign(st) * np.cos(php - pht)) if np.isfinite(sp) and np.isfinite(st) else np.nan
    hi, lo = (p > 5 * pm), (p < 1.0)
    yrs = len(p) / 365.25
    ptot = np.nansum(p.values)
    return dict(
        clim_p_mean=pm,
        clim_pet_mean=petm,
        clim_aridity=petm / pm if pm > 0 else np.nan,
        clim_p_seasonality=seas,
        clim_frac_snow=float(np.nansum(p.values[tmean.values < 0]) / ptot) if ptot > 0 else 0.0,
        clim_high_prec_freq=float(hi.sum() / yrs),
        clim_high_prec_dur=_mean_spell(hi),
        clim_low_prec_freq=float(lo.sum() / yrs),
        clim_low_prec_dur=_mean_spell(lo),
        clim_prcp_cv=float(np.nanstd(p) / pm) if pm > 0 else np.nan,
        clim_tmean=float(np.nanmean(tmean)),
        clim_trange=float(np.nanmean(df["tmax"] - df["tmin"])),
        clim_srad_mean=float(np.nanmean(df["srad"])),
        clim_vp_mean=float(np.nanmean(df["vp"])),
        clim_aet_mean=float(np.nanmean(df["aet"])) if "aet" in df else np.nan,
    )


def hydro_attributes(df: pd.DataFrame) -> dict:
    """由实测 q 导出 —— 未测流域不可得, 只用于分层 / T2 / E3 事后回归, 不入模."""
    from .metrics import eckhardt_bfi, richards_baker_flashiness, recession_constant
    q = df["q"].values.astype(float)
    qq = q[np.isfinite(q)]
    p = df["prcp"].values.astype(float)
    yrs = len(q) / 365.25
    if len(qq) == 0:
        return {k: np.nan for k in
                ["hyd_q_mean", "hyd_runoff_ratio", "hyd_zero_q_freq", "hyd_bfi", "hyd_rb_flashiness",
                 "hyd_slope_fdc", "hyd_q5", "hyd_q95", "hyd_high_q_freq", "hyd_high_q_dur",
                 "hyd_low_q_freq", "hyd_low_q_dur", "hyd_hfd_mean", "hyd_recession_k", "hyd_ac1"]}
    med, mean = float(np.median(qq)), float(np.mean(qq))
    hi = qq > 9 * med
    lo = qq < 0.2 * mean
    q33, q66 = np.percentile(qq, [33, 66])
    slope_fdc = float((np.log(q33 + 1e-6) - np.log(q66 + 1e-6)) / 0.33)
    # 半年流量日 (南半球水文年起于 7-1)
    s = df["q"].dropna()
    hfd = np.nan
    if len(s) > 365:
        dts = []
        for _, gg in s.groupby(pd.PeriodIndex(s.index, freq="Y-JUN")):
            if len(gg) < 300:
                continue
            c = np.cumsum(gg.values)
            if c[-1] <= 0:
                continue
            dts.append(int(np.argmax(c >= 0.5 * c[-1])))
        hfd = float(np.mean(dts)) if dts else np.nan
    return dict(
        hyd_q_mean=mean,
        hyd_runoff_ratio=float(mean / np.nanmean(p)) if np.nanmean(p) > 0 else np.nan,
        hyd_zero_q_freq=float(np.mean(qq == 0)),
        hyd_bfi=float(eckhardt_bfi(qq)[0]),
        hyd_rb_flashiness=float(richards_baker_flashiness(qq)),
        hyd_slope_fdc=slope_fdc,
        hyd_q5=float(np.percentile(qq, 5)),
        hyd_q95=float(np.percentile(qq, 95)),
        hyd_high_q_freq=float(hi.sum() / yrs),
        hyd_high_q_dur=_mean_spell(hi),
        hyd_low_q_freq=float(lo.sum() / yrs),
        hyd_low_q_dur=_mean_spell(lo),
        hyd_hfd_mean=hfd,
        hyd_recession_k=float(recession_constant(qq)),
        hyd_ac1=float(pd.Series(qq).autocorr(1)) if len(qq) > 10 else np.nan,
    )


def compute_attributes(dataset="camels_aus", basins=None, verbose=True) -> pd.DataFrame:
    """全部属性 (climate + hydro), 只用率定期 1980-2004."""
    dataset = _norm(dataset)
    basins = basins or list_basins(dataset)
    rows = []
    for i, b in enumerate(basins, 1):
        df = calib_period(load_basin(dataset, b), dataset)
        rows.append(dict(basin=b, region=basin_region(b), river_basin=basin_river_basin(b),
                         **climate_attributes(df), **hydro_attributes(df)))
        if verbose and i % 50 == 0:
            print(f"  属性 {i}/{len(basins)}", flush=True)
    return pd.DataFrame(rows).set_index("basin")


def load_attributes(dataset="camels_aus", kind="all") -> pd.DataFrame:
    """
    读取 scripts/prepare_data.py 生成的属性表; 不存在则现算 (慢).
    kind: 'climate' (入模) / 'hydro' (分析) / 'all'.
    """
    _norm(dataset)
    f = PROC / "attributes_aus.csv"
    att = pd.read_csv(f, index_col=0) if f.exists() else compute_attributes(dataset)
    att.index = att.index.astype(str)
    if kind == "climate":
        return att[[c for c in att.columns if c.startswith("clim_")]]
    if kind == "hydro":
        return att[[c for c in att.columns if c.startswith("hyd_")]]
    return att


# --------------------------------------------------------------------------- 分层与核心站点
def catchment_signatures(dataset="camels_aus", basins=None) -> pd.DataFrame:
    """兼容旧接口: 分层用签名 (干旱指数 / BFI / 零流量比例 / RB 闪变 ...)."""
    att = load_attributes(dataset, "all")
    if basins is not None:
        att = att.loc[att.index.intersection([str(b) for b in basins])]
    return pd.DataFrame({
        "aridity_PET_P": att["clim_aridity"],
        "BFI": att["hyd_bfi"],
        "zero_frac": att["hyd_zero_q_freq"],
        "RB_flashiness": att["hyd_rb_flashiness"],
        "mean_q": att["hyd_q_mean"],
        "runoff_ratio": att["hyd_runoff_ratio"],
        "p_seasonality": att["clim_p_seasonality"],
        "region": att["region"] if "region" in att.columns else "",
    })


def select_core_catchments(sig: pd.DataFrame, n_per_class=4, seed=0) -> pd.DataFrame:
    """
    T2 核心站点: 四类各 n_per_class 个, 尽量分散在不同 drainage division.
      humid_high_BFI        湿润高基流 (持久基流, 低流量支应稳定)
      humid_flashy          湿润强闪变 (峰值支主导)
      semiarid_intermittent 半干旱间歇性 (高零流量比例 —— 零流量约定的展示站)
      dry_perennial         干旱但常年有流 (蒸发主导, 低流量支易系统偏高)
    """
    rng = np.random.default_rng(seed)
    s = sig.dropna(subset=["BFI", "RB_flashiness", "aridity_PET_P"]).copy()
    med = s["aridity_PET_P"].median()
    humid, dry = s[s.aridity_PET_P <= med], s[s.aridity_PET_P > med]
    pools = {
        "humid_high_BFI": humid.sort_values("BFI", ascending=False).head(20),
        "humid_flashy": humid.sort_values("RB_flashiness", ascending=False).head(20),
        "semiarid_intermittent": dry.sort_values("zero_frac", ascending=False).head(20),
        "dry_perennial": dry[dry.zero_frac < 0.02].sort_values("BFI", ascending=False).head(20),
    }
    out, used = [], set()
    for k, pool in pools.items():
        pool = pool[~pool.index.isin(used)]
        if len(pool) == 0:
            continue
        picked, regions = [], set()
        for b in pool.sample(frac=1.0, random_state=int(rng.integers(1e6))).index:
            if len(picked) >= n_per_class:
                break
            r = str(pool.loc[b].get("region", ""))
            if r in regions and len(picked) >= n_per_class // 2:
                continue
            picked.append(b); regions.add(r)
        for b in picked:
            used.add(b)
            out.append(dict(basin=b, stratum=k, **sig.loc[b].to_dict()))
    return pd.DataFrame(out).drop_duplicates("basin").set_index("basin")
