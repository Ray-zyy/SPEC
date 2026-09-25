"""
models/conceptual/run_conceptual.py —— GR4J / HBV + SCE-UA 逐站率定 (KGE 与 lnNSE 各一次)
                                       + 三个基准 (气候态 / 持续性 / 线性 ARX)

用法:
  python models/conceptual/run_conceptual.py --dataset camels_aus --model gr4j --obj kge \
      --basins_file data/processed/basins_aus.txt --out runs/aus_gr4j_kge --workers 16

纯 NumPy, CPU 多进程 (深度模型占 GPU 时并行跑, 互不干扰).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spec.data import load_basin, SPLITS                       # noqa
from spec.metrics import kge, log_nse                          # noqa
from spec.io import save_preds                                # noqa
from models.conceptual.kernels import gr4j_fast, hbv_fast, HAVE_NUMBA   # noqa


# ------------------------------------------------------------------ GR4J
def _s_curve1(t, x4):
    return np.where(t <= 0, 0, np.where(t < x4, (t / x4) ** 2.5, 1.0))


def _s_curve2(t, x4):
    out = np.zeros_like(t, float)
    m = (t > 0) & (t <= x4)
    out[m] = 0.5 * (t[m] / x4) ** 2.5
    m = (t > x4) & (t < 2 * x4)
    out[m] = 1 - 0.5 * (2 - t[m] / x4) ** 2.5
    out[t >= 2 * x4] = 1.0
    return out


def gr4j_ref(P, E, params):
    """Perrin et al. 2003. params = [x1 (mm), x2 (mm), x3 (mm), x4 (d)]."""
    x1, x2, x3, x4 = params
    n = len(P)
    nUH1 = int(np.ceil(x4)); nUH2 = int(np.ceil(2 * x4))
    t1 = np.arange(1, nUH1 + 1, dtype=float); t2 = np.arange(1, nUH2 + 1, dtype=float)
    uh1 = np.diff(np.concatenate([[0], _s_curve1(t1, x4)]))
    uh2 = np.diff(np.concatenate([[0], _s_curve2(t2, x4)]))
    S, R = 0.6 * x1, 0.7 * x3
    q1 = np.zeros(nUH1); q2 = np.zeros(nUH2); Q = np.zeros(n)
    for t in range(n):
        p, e = P[t], E[t]
        if p >= e:
            pn, en = p - e, 0.0
            ws = np.tanh(pn / x1); ps = x1 * (1 - (S / x1) ** 2) * ws / (1 + S / x1 * ws); es = 0.0
        else:
            pn, en = 0.0, e - p
            we = np.tanh(en / x1); es = S * (2 - S / x1) * we / (1 + (1 - S / x1) * we); ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        q1 = np.roll(q1, -1); q1[-1] = 0; q1 += uh1 * (0.9 * pr)
        q2 = np.roll(q2, -1); q2[-1] = 0; q2 += uh2 * (0.1 * pr)
        F = x2 * (R / x3) ** 3.5
        R = max(0.0, R + q1[0] + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        qd = max(0.0, q2[0] + F)
        Q[t] = qr + qd
    return Q


def gr4j(P, E, params):
    """快核封装 (numba 可用时 JIT); gr4j_ref 为原始实现, 供数值对拍."""
    return gr4j_fast(P, E, params)


GR4J_BOUNDS = np.array([[10, 2000], [-8, 8], [10, 800], [0.5, 8]], float)


# ------------------------------------------------------------------ HBV
def hbv_ref(P, T, E, params):
    """HBV-light 结构 (Seibert & Vis 2012). params 见 HBV_BOUNDS 顺序."""
    (TT, CFMAX, SFCF, CWH, CFR, FC, LP, BETA, K0, K1, K2, UZL, PERC, MAXBAS) = params
    n = len(P); SP = WC = SM = 0.0; SUZ = SLZ = 0.0
    Q = np.zeros(n)
    for t in range(n):
        p, temp, ep = P[t], T[t], E[t]
        if temp < TT:
            SP += p * SFCF; insoil = 0.0
        else:
            melt = min(CFMAX * (temp - TT), SP)
            SP -= melt; WC += melt + p
            rel = max(0.0, WC - CWH * SP); WC -= rel; insoil = rel
        if temp < TT:
            refr = min(CFR * CFMAX * (TT - temp), WC); WC -= refr; SP += refr
        r = insoil * (max(SM, 0) / FC) ** BETA if FC > 0 else 0.0
        r = min(r, insoil)
        SM += insoil - r
        ea = ep * min(SM / (LP * FC), 1.0) if FC > 0 else 0.0
        SM = max(0.0, SM - ea)
        SUZ += r
        perc = min(PERC, SUZ); SUZ -= perc; SLZ += perc
        q0 = K0 * max(0.0, SUZ - UZL); SUZ -= q0
        q1 = K1 * SUZ; SUZ -= q1
        q2 = K2 * SLZ; SLZ -= q2
        Q[t] = q0 + q1 + q2
    mb = max(int(round(MAXBAS)), 1)
    w = np.array([min(i + 1, mb - i) for i in range(mb)], float); w /= w.sum()
    return np.convolve(Q, w, mode="same")


def hbv(P, T, E, params):
    """快核封装 (numba 可用时 JIT); hbv_ref 为原始实现, 供数值对拍."""
    return hbv_fast(P, T, E, params)


HBV_BOUNDS = np.array([[-2, 2], [1, 8], [0.4, 1.2], [0, .2], [0, .1], [50, 700], [.3, 1],
                       [1, 6], [.05, .8], [.01, .4], [.001, .15], [0, 70], [0, 6], [1, 6]], float)


# ------------------------------------------------------------------ SCE-UA (Duan 1992)
def sceua(objfun, bounds, ngs=6, maxn=8000, seed=0, kstop=10, pcento=0.01):
    """Shuffled Complex Evolution (Duan 1992). **最小化** objfun."""
    rng = np.random.default_rng(seed)
    d = len(bounds); npg = 2 * d + 1; nps = d + 1; npt = ngs * npg
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = lo + rng.random((npt, d)) * (hi - lo)
    F = np.array([objfun(x) for x in X]); nev = npt
    order = np.argsort(F); X, F = X[order], F[order]
    hist = [F[0]]
    while nev < maxn:
        for g in range(ngs):
            idx = np.arange(g, npt, ngs)
            for _ in range(npg):
                w = (2 * (npg + 1 - np.arange(1, npg + 1))) / (npg * (npg + 1))
                sub = rng.choice(idx, nps, replace=False, p=w / w.sum())
                s = np.argsort(F[sub]); sub = sub[s]
                worst = sub[-1]; centroid = X[sub[:-1]].mean(0)
                new = np.clip(centroid + (centroid - X[worst]), lo, hi)         # reflection
                fn = objfun(new); nev += 1
                if fn >= F[worst]:
                    new = (centroid + X[worst]) / 2; fn = objfun(new); nev += 1  # contraction
                if fn >= F[worst]:
                    new = lo + rng.random(d) * (hi - lo); fn = objfun(new); nev += 1  # mutation
                X[worst], F[worst] = new, fn
        order = np.argsort(F); X, F = X[order], F[order]
        hist.append(F[0])
        if len(hist) > kstop and abs(hist[-1] - hist[-kstop]) < pcento * abs(hist[-kstop] + 1e-9):
            break
    return X[0], F[0], nev


# ------------------------------------------------------------------ 逐站率定
def calibrate_basin(task):
    dataset, basin, model, obj, seed, maxn = task
    try:
        df = load_basin(dataset, basin)
        a0 = SPLITS[dataset]["train"][0]; a1 = SPLITS[dataset]["val"][1]
        cal = df.loc[a0:a1]; test = df.loc[SPLITS[dataset]["test"][0]: SPLITS[dataset]["test"][1]]
        warm = 365

        def series(d):
            P = np.nan_to_num(d.get("prcp", pd.Series(0, index=d.index)).values)
            E = np.nan_to_num(d.get("pet", pd.Series(2.0, index=d.index)).values)
            T = np.nan_to_num(d.get("tmax", pd.Series(10.0, index=d.index)).values)
            return P, E, T

        Pc, Ec, Tc = series(cal); qc = cal["q"].values
        objf = {"kge": kge, "lnnse": log_nse}[obj]

        def f(par):
            sim = gr4j(Pc, Ec, par) if model == "gr4j" else hbv(Pc, Tc, Ec, par)
            ok = np.isfinite(qc); ok[:warm] = False
            if ok.sum() < 100 or not np.all(np.isfinite(sim[ok])):
                return 9e9
            v = objf(qc[ok], sim[ok])
            return -v if np.isfinite(v) else 9e9        # SCE-UA 最小化 -> 取负

        bounds = GR4J_BOUNDS if model == "gr4j" else HBV_BOUNDS
        par, fval, nev = sceua(f, bounds, seed=seed, maxn=maxn)
        fval = -fval                                     # 还原为目标函数值 (越大越好)
        Pt, Et, Tt = series(test)
        sim = gr4j(Pt, Et, par) if model == "gr4j" else hbv(Pt, Tt, Et, par)
        out = pd.DataFrame(dict(basin=basin, date=test.index, obs=test["q"].values,
                                sim=np.clip(sim, 0, None)))
        return out, dict(basin=basin, obj_value=float(fval), n_eval=int(nev),
                         params=[float(p) for p in par])
    except Exception as e:
        return None, dict(basin=basin, error=str(e))


# ------------------------------------------------------------------ 基准模型
def benchmarks(dataset, basins, out):
    from numpy.linalg import lstsq
    recs = {"climatology": [], "persistence": [], "arx": []}
    for b in basins:
        df = load_basin(dataset, b)
        tr = df.loc[SPLITS[dataset]["train"][0]: SPLITS[dataset]["val"][1]]
        te = df.loc[SPLITS[dataset]["test"][0]: SPLITS[dataset]["test"][1]]
        doy_mean = tr.groupby(tr.index.dayofyear)["q"].mean()
        clim = te.index.dayofyear.map(doy_mean).values.astype(float)
        recs["climatology"].append(pd.DataFrame(dict(basin=b, date=te.index, obs=te["q"].values, sim=clim)))
        pers = np.concatenate([[np.nan], te["q"].values[:-1]])
        recs["persistence"].append(pd.DataFrame(dict(basin=b, date=te.index, obs=te["q"].values, sim=pers)))
        # 线性 ARX: q_t ~ q_{t-1..3} + P_{t..t-4}
        def design(d):
            q = d["q"].values; P = np.nan_to_num(d.get("prcp", pd.Series(0., index=d.index)).values)
            cols = [np.roll(q, k) for k in (1, 2, 3)] + [np.roll(P, k) for k in range(5)]
            X = np.column_stack(cols + [np.ones(len(q))]); X[:5] = np.nan
            return X, q
        Xtr, ytr = design(tr); ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
        beta = lstsq(Xtr[ok], ytr[ok], rcond=None)[0]
        Xte, yte = design(te)
        sim = np.where(np.isfinite(Xte).all(1), np.nan_to_num(Xte) @ beta, np.nan)
        recs["arx"].append(pd.DataFrame(dict(basin=b, date=te.index, obs=yte, sim=np.clip(sim, 0, None))))
    for k, v in recs.items():
        d = Path(out) / k; d.mkdir(parents=True, exist_ok=True)
        save_preds(pd.concat(v), d)
        json.dump(dict(model=k, dataset=dataset), open(d / "config.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")
    ap.add_argument("--model", default="gr4j", choices=["gr4j", "hbv", "benchmarks"])
    ap.add_argument("--obj", default="kge", choices=["kge", "lnnse"])
    ap.add_argument("--basins_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--maxn", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    basins = [l.strip() for l in open(a.basins_file) if l.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.model == "benchmarks":
        benchmarks(a.dataset, basins, out); return
    tasks = [(a.dataset, b, a.model, a.obj, a.seed, a.maxn) for b in basins]
    with Pool(a.workers) as pool:
        res = pool.map(calibrate_basin, tasks)
    preds = [r[0] for r in res if r[0] is not None]
    save_preds(pd.concat(preds), out)
    pd.DataFrame([r[1] for r in res]).to_csv(out / "calibration.csv", index=False)
    json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    print("done:", out, len(preds), "basins")


if __name__ == "__main__":
    main()
