"""F1 双向扫描示意图 + F2 delta-rho 关系与 tanh 恒等式 (纯理论图, 不需要真实数据)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spec.core import ALPHA_GRID, curves, delta_from_ratio, zero_crossing
import spec.plotting as sp

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------ F1
rng = np.random.default_rng(3)
n = 2000
y = np.abs(rng.lognormal(0, 1.0, n)) * np.exp(.5 * np.sin(np.arange(n) / 58))
q90, q70 = np.quantile(y, .9), np.quantile(y, .7)
yh = np.where(y >= q90, q90 + .7 * (y - q90), y)
yh = np.where(y < q70, yh + .12 * np.median(y), yh)
c = curves(y, yh)

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
s = np.sort(y)[::-1]
x = np.linspace(0, 100, len(s))
axs[0].semilogy(x, s + 1e-3, "k", lw=1)
axs[0].fill_between(x[:100], 1e-3, s[:100] + 1e-3, color="tab:red", alpha=.3)
axs[0].fill_between(x[-100:], 1e-3, s[-100:] + 1e-3, color="tab:blue", alpha=.3)
axs[0].annotate(r"$\mathcal{S}^{\uparrow}(5\%)$", (2, s[0]), color="tab:red", fontsize=8)
axs[0].annotate(r"$\mathcal{S}^{\downarrow}(5\%)$", (55, s[-1] * 3 + 1e-2), color="tab:blue", fontsize=8)
axs[0].set_xlabel("exceedance (%)"); axs[0].set_ylabel("Q")
axs[0].set_title("(a) ranking by observations, two sweeps", fontsize=9)

sp.plot_dual(axs[1], ALPHA_GRID, c.E_up, c.E_lo, sp.PALETTE[0])
sp.style_error_axis(axs[1], "(b) error curves and closure")
axs[1].scatter([100], [c.E_up[-1]], zorder=5, color="k", s=18)
axs[1].annotate(r"$E^{\uparrow}(1)=E^{\downarrow}(1)=\bar{\delta}$", (100, c.E_up[-1]),
                xytext=(-120, 30), textcoords="offset points", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=.6))
sp.add_branch_legend(axs[1])

sp.plot_dual(axs[2], ALPHA_GRID, c.B_up, c.B_lo, sp.PALETTE[3])
sp.style_bias_axis(axs[2], r"(c) bias curves and $\alpha^{*}$")
ast = zero_crossing(ALPHA_GRID, c.B_up)
if np.isfinite(ast):
    axs[2].axvline(ast * 100, ls=":", c="k")
    axs[2].annotate(rf"$\alpha^*$={ast:.0%}", (ast * 100, .6), fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "F1_schematic.png"); plt.close(fig)

# ------------------------------------------------ F2
rho = np.logspace(-1.3, 1.3, 500)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.1))
axs[0].semilogx(rho, delta_from_ratio(rho), "k", lw=1.6, label=r"$\delta=|\tanh(\frac{1}{2}\ln\rho)|$")
axs[0].semilogx(rho, np.abs(rho - 1), color="tab:orange", lw=1, label=r"relative error $|\rho-1|$")
axs[0].semilogx(rho, 2 * delta_from_ratio(rho), color="tab:green", ls="--", lw=1, label="SMAPE (fraction)")
axs[0].set_ylim(0, 3); axs[0].axvline(1, lw=.5, c="k"); axs[0].set_xlabel(r"$\rho=\hat y/y$")
axs[0].legend(fontsize=7, frameon=False)
axs[0].set_title("(a) boundedness and multiplicative symmetry", fontsize=9)
for r in (2.0, 0.5):
    axs[0].scatter([r], [delta_from_ratio(r)], s=18, color="tab:red", zorder=5)
axs[0].annotate(r"$\rho=2$ and $\rho=1/2$ both give $1/3$", (0.12, .45), fontsize=7)

u = np.linspace(-3, 3, 300)
axs[1].plot(u, np.tanh(u), "k", lw=1.5, label=r"$\tanh(u)$")
axs[1].plot(u, (np.exp(2 * u) - 1) / (np.exp(2 * u) + 1), "r--", lw=1, label=r"$(\rho-1)/(\rho+1)$")
axs[1].set_xlabel(r"$u=\frac{1}{2}\ln\rho$"); axs[1].set_ylabel(r"$\beta$")
axs[1].legend(fontsize=7, frameon=False)
axs[1].set_title("(b) numerical verification of the identity", fontsize=9)
axs[1].axhline(0, lw=.5, c="k"); axs[1].axvline(0, lw=.5, c="k")
fig.tight_layout(); fig.savefig(FIG / "F2_delta_rho.png"); plt.close(fig)
print("F1, F2 ->", FIG)
