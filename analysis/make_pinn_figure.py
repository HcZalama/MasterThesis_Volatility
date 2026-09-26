"""
Comparison figure on the shared book: market bid/ask, the rough Heston fit of
Table 4.8, and the PINN surface of section 4.1, at the two maturities where
Heston comes closest to the spread.

The PINN curve is drawn from Predicted.txt when that file is present in this
directory; without it the figure is still produced, market against Heston only.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rough_heston as rh

OUT = "../Images/pinn-heston-compare.pdf"
P = (0.14411592355356212, 0.13591516550551117, 6.618677361945307,
     1.257764783117263, -0.12341524739208617, 0.5)
MATS = (0.522929, 0.695414)

K, G, L, R = "#1a1a1a", "#8a8a8a", "#c8c8c8", "#b03030"
B = "#2b5d8a"

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 9, "axes.unicode_minus": False,
                     "axes.formatter.use_mathtext": True,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0})


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


if __name__ == "__main__":
    q = pd.read_csv("pinn_heston_sample.csv")
    if os.path.exists("Predicted.txt"):
        pinn = pd.read_csv("Predicted.txt")
        pinn.columns = ["T", "k", "iv"]
        print("PINN curve from the full Predicted.txt export", flush=True)
    elif os.path.exists("pinn_predicted_slice.csv"):
        pinn = pd.read_csv("pinn_predicted_slice.csv")
        print("PINN curve from pinn_predicted_slice.csv (the two plotted "
              "maturities only)", flush=True)
    else:
        pinn = None
        print("no PINN export found -- market vs Heston only", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.12,
                                              wspace=0.24))

    for j, T in enumerate(MATS):
        s = q[np.isclose(q["T"], T)].sort_values("k").reset_index(drop=True)
        kk = s["k"].to_numpy()
        iv = rh.surface_iv(s, P, N=400)
        h = s["half_spread"].to_numpy()

        a = axes[0, j]
        a.fill_between(kk, s["bid"], s["ask"], color=L, lw=0, zorder=1,
                       label="bid/ask")
        a.plot(kk, s["mid"], ".", ms=3.2, color=G, zorder=3, label="mid")
        a.plot(kk, iv, "-", color=R, zorder=4, label="rough Heston")
        if pinn is not None:
            p = pinn[np.isclose(pinn["T"], T)].sort_values("k")
            p = p[(p["k"] >= kk.min()) & (p["k"] <= kk.max())]
            a.plot(p["k"], p["iv"], "-", color=B, lw=1.1, zorder=5, label="PINN")
        a.set_title(rf"$T={T:.4f}$", pad=4)
        a.set_ylabel(r"implied volatility") if j == 0 else None
        tidy(a)
        if j == 0:
            a.legend(frameon=False, fontsize=7.5, loc="upper center",
                     handlelength=1.4, borderaxespad=0.2)

        b = axes[1, j]
        b.axhspan(-1, 1, color=L, lw=0, zorder=1)
        b.axhline(0, color=K, lw=0.5, zorder=2)
        b.plot(kk, (iv - s["mid"]) / h, "o-", ms=2.6, color=R, zorder=4,
               label="rough Heston")
        if pinn is not None:
            p = pinn[np.isclose(pinn["T"], T)].sort_values("k")
            ip = np.interp(kk, p["k"], p["iv"])
            b.plot(kk, (ip - s["mid"]) / h, "o-", ms=2.6, color=B, lw=1.0,
                   zorder=5, label="PINN")
        b.set_xlabel(r"log-moneyness $k$")
        b.set_ylabel(r"error / half-spread") if j == 0 else None
        tidy(b)

    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT, flush=True)
