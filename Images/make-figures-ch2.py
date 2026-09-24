"""Figures for Chapter 2.  Run from anywhere; writes straight into Images/.

Figure: the two shapes of Subsection 2.2.4 --- the equity skew and the FX
smile --- drawn against the flat line that Black--Scholes predicts.  The
curves are illustrative shapes, not a calibration: the point of the picture
is the contrast with the dotted horizontal line, and nothing is claimed
about any particular market or date.
"""

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
OUT = r"c:/Users/hecto/source/repos/HcZalama/MasterThesis_Volatility/Images/"
K, G = "#1a1a1a", "#8a8a8a"

def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

k = np.linspace(-0.25, 0.25, 400)

# Equity: a falling, mildly convex wing.  Forex: near-symmetric about the money.
eq = lambda x: 0.20 - 0.30 * x + 0.45 * x**2
fx = lambda x: 0.10 - 0.03 * x + 0.90 * x**2

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))

for ax, f, atm, title in ((a1, eq, 0.20, "Equity index: the skew (smirk)"),
                          (a2, fx, 0.10, "Foreign exchange: the smile")):
    ax.plot(k, 100 * f(k), lw=1.2, color=K, zorder=3)
    ax.axhline(100 * atm, lw=0.8, ls=":", color=K, zorder=2,
               label=r"Black--Scholes: $\sigma_{\mathrm{imp}}$ constant")
    ax.axvline(0.0, lw=0.5, color=G, zorder=1)
    ax.set_xlabel(r"log-moneyness $k=\log(K/F)$")
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlim(k[0], k[-1])
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    tidy(ax)

arrow = dict(arrowstyle="->", lw=0.6, color=G, connectionstyle="arc3,rad=-0.25")

# Left: the whole of the empty region is the lower-left quadrant.
a1.set_ylabel(r"$\sigma_{\mathrm{imp}}$  (%)")
a1.set_ylim(12, 31)
a1.annotate("low strikes bid up\n(crash phobia)",
            xy=(-0.205, 100 * eq(-0.205)), xytext=(-0.235, 14.0),
            fontsize=7.5, color=K, ha="left", va="bottom", arrowprops=arrow)
a1.legend(frameon=False, loc="upper right", handlelength=2.0)

# Right: the U leaves the centre free; label the dotted line inline instead
# of with a second legend, which would collide with the arms.
a2.set_ylim(9, 17)
a2.annotate("both tails bid up\n(fat tails)",
            xy=(0.180, 100 * fx(0.180)), xytext=(-0.010, 15.2),
            fontsize=7.5, color=K, ha="center", va="top",
            arrowprops=dict(arrowstyle="->", lw=0.6, color=G,
                            connectionstyle="arc3,rad=0.25"))
a2.text(-0.238, 10.25, "Black--Scholes", fontsize=7.5, color=K,
        ha="left", va="bottom")

fig.tight_layout(pad=0.4)
fig.savefig(OUT + "smile-skew.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT + "smile-skew.pdf")
