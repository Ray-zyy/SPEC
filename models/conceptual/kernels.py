"""
models/conceptual/kernels.py —— GR4J / HBV 前向积分的快核

原实现每个时间步调用 np.roll, 在 SCE-UA 里 (8000 次评价 × 9131 天 × 222 站 × 4 组合)
会成为整个流程的瓶颈. 这里改成环形缓冲 + 纯标量循环, 并在 numba 可用时 JIT (约 50-100×).

    pip install numba          # 强烈建议; 没装也能跑, 只是慢

数值结果与原实现逐位一致 (tests/test_kernels_equivalence 校验).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # numba 未安装 -> 原样返回, 走纯 Python
    HAVE_NUMBA = False

    def njit(*a, **kw):
        """numba 缺失时的空装饰器: 支持 @njit 与 @njit(...) 两种写法."""
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]

        def deco(f):
            return f
        return deco


@njit(cache=True, fastmath=True)
def _gr4j_core(P, E, x1, x2, x3, uh1, uh2):
    n = P.shape[0]
    n1, n2 = uh1.shape[0], uh2.shape[0]
    q1 = np.zeros(n1)
    q2 = np.zeros(n2)
    S = 0.6 * x1
    R = 0.7 * x3
    Q = np.zeros(n)
    p1 = 0
    p2 = 0
    for t in range(n):
        p = P[t]
        e = E[t]
        if p >= e:
            pn = p - e
            ws = np.tanh(pn / x1)
            ps = x1 * (1.0 - (S / x1) ** 2) * ws / (1.0 + S / x1 * ws)
            es = 0.0
        else:
            en = e - p
            pn = 0.0
            we = np.tanh(en / x1)
            es = S * (2.0 - S / x1) * we / (1.0 + (1.0 - S / x1) * we)
            ps = 0.0
        S = S - es + ps
        perc = S * (1.0 - (1.0 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        for j in range(n1):
            q1[(p1 + j) % n1] += uh1[j] * 0.9 * pr
        for j in range(n2):
            q2[(p2 + j) % n2] += uh2[j] * 0.1 * pr
        out1 = q1[p1]
        q1[p1] = 0.0
        p1 = (p1 + 1) % n1
        out2 = q2[p2]
        q2[p2] = 0.0
        p2 = (p2 + 1) % n2
        F = x2 * (R / x3) ** 3.5
        R = R + out1 + F
        if R < 0.0:
            R = 0.0
        qr = R * (1.0 - (1.0 + (R / x3) ** 4) ** -0.25)
        R -= qr
        qd = out2 + F
        if qd < 0.0:
            qd = 0.0
        Q[t] = qr + qd
    return Q


def _s1(t, x4):
    return np.where(t <= 0, 0.0, np.where(t < x4, (t / x4) ** 2.5, 1.0))


def _s2(t, x4):
    out = np.zeros_like(t, float)
    m = (t > 0) & (t <= x4)
    out[m] = 0.5 * (t[m] / x4) ** 2.5
    m = (t > x4) & (t < 2 * x4)
    out[m] = 1 - 0.5 * (2 - t[m] / x4) ** 2.5
    out[t >= 2 * x4] = 1.0
    return out


def gr4j_fast(P, E, params):
    x1, x2, x3, x4 = [float(v) for v in params]
    n1, n2 = int(np.ceil(x4)), int(np.ceil(2 * x4))
    uh1 = np.diff(np.concatenate([[0.0], _s1(np.arange(1, n1 + 1, dtype=float), x4)]))
    uh2 = np.diff(np.concatenate([[0.0], _s2(np.arange(1, n2 + 1, dtype=float), x4)]))
    return _gr4j_core(np.ascontiguousarray(P, float), np.ascontiguousarray(E, float),
                      x1, x2, x3, uh1, uh2)


@njit(cache=True, fastmath=True)
def _hbv_core(P, T, E, TT, CFMAX, SFCF, CWH, CFR, FC, LP, BETA, K0, K1, K2, UZL, PERC):
    n = P.shape[0]
    SP = 0.0
    WC = 0.0
    SM = 0.0
    SUZ = 0.0
    SLZ = 0.0
    Q = np.zeros(n)
    for t in range(n):
        p = P[t]
        temp = T[t]
        ep = E[t]
        if temp < TT:
            SP += p * SFCF
            insoil = 0.0
        else:
            melt = CFMAX * (temp - TT)
            if melt > SP:
                melt = SP
            SP -= melt
            WC += melt + p
            rel = WC - CWH * SP
            if rel < 0.0:
                rel = 0.0
            WC -= rel
            insoil = rel
        if temp < TT:
            refr = CFR * CFMAX * (TT - temp)
            if refr > WC:
                refr = WC
            WC -= refr
            SP += refr
        smpos = SM if SM > 0.0 else 0.0
        r = insoil * (smpos / FC) ** BETA if FC > 0.0 else 0.0
        if r > insoil:
            r = insoil
        SM += insoil - r
        ratio = SM / (LP * FC) if FC > 0.0 else 0.0
        if ratio > 1.0:
            ratio = 1.0
        ea = ep * ratio
        SM -= ea
        if SM < 0.0:
            SM = 0.0
        SUZ += r
        perc = PERC if PERC < SUZ else SUZ
        SUZ -= perc
        SLZ += perc
        excess = SUZ - UZL
        if excess < 0.0:
            excess = 0.0
        q0 = K0 * excess
        SUZ -= q0
        q1 = K1 * SUZ
        SUZ -= q1
        q2 = K2 * SLZ
        SLZ -= q2
        Q[t] = q0 + q1 + q2
    return Q


def hbv_fast(P, T, E, params):
    (TT, CFMAX, SFCF, CWH, CFR, FC, LP, BETA, K0, K1, K2, UZL, PERC, MAXBAS) = [float(v) for v in params]
    Q = _hbv_core(np.ascontiguousarray(P, float), np.ascontiguousarray(T, float),
                  np.ascontiguousarray(E, float), TT, CFMAX, SFCF, CWH, CFR, FC, LP,
                  BETA, K0, K1, K2, UZL, PERC)
    mb = max(int(round(MAXBAS)), 1)
    w = np.array([min(i + 1, mb - i) for i in range(mb)], float)
    w /= w.sum()
    return np.convolve(Q, w, mode="same")
