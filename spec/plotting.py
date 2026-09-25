"""
spec.plotting —— 论文图的统一绘图函数 (双支曲线是全文最常见的图元)
约定: 峰值支 E^up 实线, 基流支 E^lo 虚线; 同一模型同一颜色; x 轴 alpha (%) 可对数.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
_FONT_PATH = Path(__file__).resolve().parents[1] / "DejaVuMathTeXGyre.ttf"
if _FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_FONT_PATH))
    _FONT_NAME = font_manager.FontProperties(fname=str(_FONT_PATH)).get_name()
else:
    _FONT_NAME = "DejaVu Sans"
from .core import ALPHA_GRID

plt.rcParams.update({"font.size": 9, "font.family": _FONT_NAME, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})
PALETTE = plt.get_cmap("tab10").colors


def _x(alphas, logx):
    return alphas * 100


def plot_dual(ax, alphas, up, lo, color, label=None, logx=True, band_up=None, band_lo=None,
              lw=1.6, alpha_band=0.18):
    """一对曲线: up 实线, lo 虚线; band_* = (lower, upper)."""
    x = _x(alphas, logx)
    ax.plot(x, up, "-", color=color, lw=lw, label=label)
    ax.plot(x, lo, "--", color=color, lw=lw)
    if band_up is not None:
        ax.fill_between(x, band_up[0], band_up[1], color=color, alpha=alpha_band, lw=0)
    if band_lo is not None:
        ax.fill_between(x, band_lo[0], band_lo[1], color=color, alpha=alpha_band, lw=0, hatch="//")
    if logx:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel(r"$\alpha$ (%)")


def style_error_axis(ax, title=None):
    ax.set_ylim(0, 1); ax.set_ylabel(r"$E(\alpha)$")
    if title:
        ax.set_title(title, fontsize=9)


def style_bias_axis(ax, title=None):
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-1, 1); ax.set_ylabel(r"$B(\alpha)$")
    if title:
        ax.set_title(title, fontsize=9)


def add_branch_legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", ls="-", label=r"peak branch $E^{\uparrow}$"),
         Line2D([], [], color="k", ls="--", label=r"baseflow branch $E^{\downarrow}$")]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0], fontsize=7, frameon=False)


def plot_spec_pair(res, boot=None, title=None, color=PALETTE[0], fname=None, alpha_star=True):
    """单个 (站, 模型): 左误差曲线, 右偏差曲线, 可带 bootstrap 带."""
    c = res["curves"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    bu = bl = bbu = bbl = None
    if boot is not None:
        from .bootstrap import pointwise_band
        bu, bl = pointwise_band(boot["E_up"]), pointwise_band(boot["E_lo"])
        bbu, bbl = pointwise_band(boot["B_up"]), pointwise_band(boot["B_lo"])
    plot_dual(axs[0], c.alphas, c.E_up, c.E_lo, color, band_up=bu, band_lo=bl)
    style_error_axis(axs[0], "error"); add_branch_legend(axs[0])
    plot_dual(axs[1], c.alphas, c.B_up, c.B_lo, color, band_up=bbu, band_lo=bbl)
    style_bias_axis(axs[1], "bias")
    if alpha_star and np.isfinite(res["summary"]["alpha_star_up"]):
        a = res["summary"]["alpha_star_up"] * 100
        axs[1].axvline(a, color=color, lw=0.8, ls=":"); axs[1].annotate(rf"$\alpha^*$={a:.0f}%", (a, 0.85), fontsize=7)
    axs[1].text(0.98, 0.04, f"n={c.n}, m(1%)={c.m[0]}, zero={c.zero_frac:.0%}", transform=axs[1].transAxes,
                ha="right", fontsize=6.5)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    if fname:
        fig.savefig(fname); plt.close(fig)
    return fig


def plot_models_overlay(curve_dict, band_dict=None, title=None, fname=None, kind="E"):
    """多模型叠加 (E3 头图的单个 panel): curve_dict[model] = (up, lo); band_dict[model] = ((lo_up,hi_up),(lo_lo,hi_lo))."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (mname, (up, lo)) in enumerate(curve_dict.items()):
        bu = bl = None
        if band_dict and mname in band_dict:
            bu, bl = band_dict[mname]
        plot_dual(ax, ALPHA_GRID, up, lo, PALETTE[i % 10], label=mname, band_up=bu, band_lo=bl)
    (style_error_axis if kind == "E" else style_bias_axis)(ax, title)
    from matplotlib.lines import Line2D
    model_handles = [Line2D([], [], color=PALETTE[i % 10], lw=2, label=m) for i, m in enumerate(curve_dict)]
    branch_handles = [Line2D([], [], color="k", ls="-", label=r"peak $E^{\uparrow}$"),
                      Line2D([], [], color="k", ls="--", label=r"baseflow $E^{\downarrow}$")]
    ax.legend(handles=branch_handles + model_handles, fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=min(3, len(model_handles) + 2), handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    if fname:
        fig.savefig(fname); plt.close(fig)
    return fig


def plot_five_panel(dates, y, yhat, res, boot, model_name, basin, fname=None):
    """E4 五联图: (i) 过程线+FDC (ii) 误差曲线带 (iii) 偏差曲线带+alpha* (iv) 边际密度+突变点 (v) 局部熵."""
    from .bootstrap import pointwise_band, simultaneous_band
    c = res["curves"]; a = c.alphas * 100
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    ax1 = fig.add_subplot(gs[0, :4]); ax1b = fig.add_subplot(gs[0, 4:])
    ax2 = fig.add_subplot(gs[1, 0:2]); ax3 = fig.add_subplot(gs[1, 2:4]); ax4 = fig.add_subplot(gs[1, 4:6])
    # (i)
    sl = slice(0, min(len(y), 3 * 365))
    ax1.plot(dates[sl], y[sl], "k", lw=0.8, label="obs"); ax1.plot(dates[sl], yhat[sl], color=PALETTE[0], lw=0.8, label=model_name)
    ax1.set_ylabel("Q"); ax1.legend(fontsize=7, frameon=False); ax1.set_title(f"{basin}: hydrograph (3 yr excerpt)", fontsize=9)
    ex = np.linspace(0, 100, len(y))
    ax1b.semilogy(ex, np.sort(y)[::-1] + 1e-3, "k", lw=0.8); ax1b.semilogy(ex, np.sort(yhat)[::-1] + 1e-3, color=PALETTE[0], lw=0.8)
    ax1b.set_xlabel("exceedance (%)"); ax1b.set_title("FDC", fontsize=9)
    # (ii)
    plot_dual(ax2, c.alphas, c.E_up, c.E_lo, PALETTE[0], band_up=pointwise_band(boot["E_up"]), band_lo=pointwise_band(boot["E_lo"]))
    slo, shi, _ = simultaneous_band(boot["E_up"]); ax2.plot(a, slo, ":", color=PALETTE[0], lw=0.7); ax2.plot(a, shi, ":", color=PALETTE[0], lw=0.7)
    style_error_axis(ax2, "(ii) error curves, 95% bands (dotted = simultaneous)"); add_branch_legend(ax2)
    # (iii)
    plot_dual(ax3, c.alphas, c.B_up, c.B_lo, PALETTE[3], band_up=pointwise_band(boot["B_up"]), band_lo=pointwise_band(boot["B_lo"]))
    style_bias_axis(ax3, "(iii) bias curves")
    ast = res["summary"]["alpha_star_up"]
    if np.isfinite(ast):
        ax3.axvline(ast * 100, ls=":", color="k", lw=0.8); ax3.text(ast * 100, 0.8, rf"$\alpha^*$={ast*100:.0f}%", fontsize=7)
    # (iv)
    d = res["density"]
    ax4.plot(a, d["e_up"], "-", color=PALETTE[2], label=r"$e^{\uparrow}$"); ax4.plot(a, d["e_lo"], "--", color=PALETTE[2], label=r"$e^{\downarrow}$")
    for cp in res["changepoints"]["up"]:
        ax4.axvline(cp * 100, color=PALETTE[2], lw=0.7, alpha=0.7)
    for cp in res["changepoints"]["lo"]:
        ax4.axvline(cp * 100, color=PALETTE[2], lw=0.7, alpha=0.7, ls="--")
    ax4.set_xscale("log"); ax4.set_xticks([1, 5, 20, 100]); ax4.set_xticklabels(["1", "5", "20", "100"])
    ax4.set_ylim(0, 1); ax4.set_ylabel(r"$e(\alpha)$"); ax4.set_xlabel(r"$\alpha$ (%)"); ax4.set_title("(iv) marginal density + PELT changepoints", fontsize=9)
    ax4b = ax4.twinx(); ax4b.plot(a, res["entropy"]["H_up"], "-", color="grey", lw=0.9); ax4b.plot(a, res["entropy"]["H_lo"], "--", color="grey", lw=0.9)
    ax4b.set_ylim(0, 1.05); ax4b.set_ylabel(r"(v) $H(\alpha)$", color="grey"); ax4b.spines["right"].set_visible(True)
    ax4.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    if fname:
        fig.savefig(fname); plt.close(fig)
    return fig


def plot_perturbation_grid(sig, fname=None):
    """E1 F3: sig[pert_name] = dict(E_up, E_lo, B_up, B_lo) 各为 (n_basins, K) 数组 -> 中位数+IQR."""
    names = list(sig.keys()); ncol = 4; nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow * 2, ncol, figsize=(2.6 * ncol, 2.0 * nrow * 2), sharex=True)
    for j, nm in enumerate(names):
        r, cc = divmod(j, ncol)
        for k, key in enumerate(("E", "B")):
            ax = axs[2 * r + k, cc]
            up, lo = sig[nm][f"{key}_up"], sig[nm][f"{key}_lo"]
            med_u, med_l = np.nanmedian(up, 0), np.nanmedian(lo, 0)
            plot_dual(ax, ALPHA_GRID, med_u, med_l, PALETTE[0],
                      band_up=(np.nanpercentile(up, 25, 0), np.nanpercentile(up, 75, 0)),
                      band_lo=(np.nanpercentile(lo, 25, 0), np.nanpercentile(lo, 75, 0)))
            if key == "E":
                ax.set_ylim(0, 0.6); ax.set_title(nm, fontsize=8)
            else:
                ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.6, 0.6)
            if cc == 0:
                ax.set_ylabel(rf"${key}(\alpha)$")
    for ax in axs.flat[len(names) * 0:]:
        pass
    fig.tight_layout()
    if fname:
        fig.savefig(fname); plt.close(fig)
    return fig

