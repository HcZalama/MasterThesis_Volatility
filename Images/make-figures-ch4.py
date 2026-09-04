"""
Figures for Chapter 4: the rough Heston calibration and the Levy area.

Produces
    rh-smiles.pdf          fit by maturity, market bid/ask against two models
    rh-identification.pdf  the H profile, the ATM term structure, the skew law
    levy-lag.pdf           one window as a planar path, and the lag scan

Run from anywhere; paths are absolute.  Requires the analysis package on the
path, since the model curves are recomputed rather than cached.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:/Users/hecto/source/repos/HcZalama/MasterThesis_Volatility/"
OUT = ROOT + "Images/"
sys.path.insert(0, ROOT + "analysis")

import rough_heston as rh                                   # noqa: E402
from load_options import load, quotes                       # noqa: E402
from levy_area import load as levy_load, drop_stale, window_stats  # noqa: E402

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
K, G, L = "#1a1a1a", "#8a8a8a", "#c8c8c8"


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =====================================================================
# data and the two parameter sets
# =====================================================================
d = load(ROOT + "data")
q = quotes(d, cone=0.5)
prof = pd.read_csv(ROOT + "analysis/H_profile.csv").sort_values("H")

P_SMOOTH = np.loadtxt(ROOT + "analysis/fit_params.txt")          # H = 0.49
r = prof.loc[(prof.H - 0.107).abs().idxmin()]
P_ROUGH = np.array([r.v0, r.theta, r.lam, r.xi, r.rho, r.H])     # H = 0.107

Ts = np.array(sorted(q["T"].unique()))


def model_iv(k, T, par):
    c = rh.call_prices(k, float(T), tuple(par), N=400)
    return rh.implied_vol_vec(c, k, np.full(k.size, float(T)),
                              np.full(k.size, 0.18))


# =====================================================================
# Figure 1: the fit, and the residual in half-spread units
# =====================================================================
SHOW = Ts[[0, 5, 15]]                 # short, belly, long end

fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.6), sharex="col")
for j, T in enumerate(SHOW):
    g = q[(q["T"] == T) & q.in_cone].sort_values("k")
    kk = g.k.to_numpy()
    hs = g.half_spread.to_numpy()
    mid = g["mid"].to_numpy()
    m1 = model_iv(kk, T, P_SMOOTH)
    m2 = model_iv(kk, T, P_ROUGH)

    a = axes[0, j]
    a.fill_between(kk, (mid - hs) * 100, (mid + hs) * 100, color=L, lw=0)
    a.plot(kk, m1 * 100, lw=1.0, color=K)
    a.plot(kk, m2 * 100, lw=1.0, ls="--", color=G)
    a.set_title(rf"$T={T:.3f}$", fontsize=8.5, pad=4)
    a.set_xlim(kk.min(), kk.max())
    tidy(a)

    b = axes[1, j]
    b.axhspan(-1, 1, color=L, lw=0)
    b.axhline(0, lw=0.6, color=G)
    b.plot(kk, (m1 - mid) / hs, lw=1.0, color=K)
    b.plot(kk, (m2 - mid) / hs, lw=1.0, ls="--", color=G)
    b.set_xlabel(r"$k=\log(K/F)$")
    b.set_xlim(kk.min(), kk.max())
    tidy(b)

axes[0, 0].set_ylabel(r"$\sigma_{\mathrm{imp}}$  (%)")
axes[1, 0].set_ylabel("residual\n(half-spreads)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend([Patch(color=L), Line2D([], [], color=K, lw=1.0),
            Line2D([], [], color=G, lw=1.0, ls="--")],
           ["bid/ask", "$H=0.49$", "$H=0.107$"],
           frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.045), handlelength=2.2)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "rh-smiles.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote rh-smiles.pdf")


# =====================================================================
# Figure 2: what the surface does and does not identify
# =====================================================================
# market ATM level and skew, from a local quadratic fit around the money
mA, mS = [], []
for T in Ts:
    g = q[q["T"] == T].sort_values("k")
    kk, vv = g.k.to_numpy(), g["mid"].to_numpy()
    m = np.abs(kk) <= 0.6 * np.sqrt(T)
    c = np.polyfit(kk[m], vv[m], 2)
    mA.append(np.polyval(c, 0.0))
    mS.append(c[1])
mA, mS = np.array(mA), np.array(mS)


def model_atm_skew(par):
    A, S = [], []
    for T in Ts:
        k = np.linspace(-0.15, 0.15, 9) * np.sqrt(T / 0.5)
        iv = model_iv(k, T, par)
        c = np.polyfit(k, iv, 2)
        A.append(np.polyval(c, 0.0))
        S.append(c[1])
    return np.array(A), np.array(S)


A1, S1 = model_atm_skew(P_SMOOTH)
A2, S2 = model_atm_skew(P_ROUGH)
slope = lambda s: np.polyfit(np.log(Ts), np.log(np.abs(s)), 1)[0]

fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(6.4, 2.35))

# -- the profile ------------------------------------------------------
p1.plot(prof.H, prof.Phi_n, "o-", ms=2.6, lw=0.9, color=K)
p1.plot([prof.H.iloc[-1]], [prof.Phi_n.iloc[-1]], "o", ms=5,
        mfc="none", mec=K, mew=1.0)
p1.axvline(0.107, lw=0.6, ls=":", color=G)
p1.text(0.115, 105, "literature\n$H\\approx0.1$", fontsize=7.5, color=G,
        va="top")
p1.set_xlabel("$H$")
p1.set_ylabel(r"$\Phi/n$")
p1.set_title("Objective profiled in $H$", fontsize=8.5, pad=5)
p1.set_xlim(0, 0.52)
tidy(p1)

# -- ATM level --------------------------------------------------------
p2.semilogx(Ts, mA * 100, "o", ms=3, color=K, label="market")
p2.semilogx(Ts, A1 * 100, "-", lw=1.0, color=K, label="$H=0.49$")
p2.semilogx(Ts, A2 * 100, "--", lw=1.0, color=G, label="$H=0.107$")
p2.set_xlabel("$T$  (years)")
p2.set_ylabel(r"ATM $\sigma_{\mathrm{imp}}$  (%)")
p2.set_title("At-the-money level", fontsize=8.5, pad=5)
p2.legend(frameon=False, loc="upper left", handlelength=1.9)
tidy(p2)

# -- the skew power law ----------------------------------------------
p3.loglog(Ts, np.abs(mS), "o", ms=3, color=K)
p3.loglog(Ts, np.abs(S1), "-", lw=1.0, color=K)
p3.loglog(Ts, np.abs(S2), "--", lw=1.0, color=G)
p3.set_xlabel("$T$  (years)")
p3.set_ylabel(r"$|\partial_k\sigma_{\mathrm{imp}}|$ at the money")
p3.set_title("Skew power law", fontsize=8.5, pad=5)
p3.text(0.97, 0.97,
        f"fitted slope\nmarket  ${slope(mS):+.3f}$\n"
        f"$H=0.49$  ${slope(S1):+.3f}$\n"
        f"$H=0.107$  ${slope(S2):+.3f}$",
        transform=p3.transAxes, fontsize=7.2, ha="right", va="top")
tidy(p3)

fig.tight_layout(pad=0.4)
fig.savefig(OUT + "rh-identification.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote rh-identification.pdf")


# =====================================================================
# Figure 3: the Levy area
# =====================================================================
df = levy_load(ROOT + "data/sp500_stoxx50_total_return.csv")
clean, _ = drop_stale(df)
x1 = clean["x1"].to_numpy()
x2 = clean["x2"].to_numpy()
n = 21

# A representative window, not the largest: the 2020 crash has by far the
# biggest area but its path self-intersects repeatedly and reads as noise.
# Take the window at the 85th percentile of |A|.
As = np.array([window_stats(x1[w * n:w * n + n + 1],
                            x2[w * n:w * n + n + 1])["area_str"]
               for w in range((len(x1) - 1) // n)])
bi = int(np.argsort(np.abs(As))[int(0.85 * As.size)])
best = As[bi]

a, b = bi * n, bi * n + n
X1 = (x1[a:b + 1] - x1[a]) * 100
X2 = (x2[a:b + 1] - x2[a]) * 100

fig, (q1, q2) = plt.subplots(1, 2, figsize=(6.4, 2.6))

q1.fill(np.concatenate([X1, [X1[0]]]), np.concatenate([X2, [X2[0]]]),
        color=L, lw=0, zorder=0)
q1.plot(X1, X2, lw=0.9, color=K)
q1.plot([X1[0], X1[-1]], [X2[0], X2[-1]], lw=0.8, ls="--", color=G)
q1.plot([X1[0]], [X2[0]], "o", ms=3.5, color=K)
q1.plot([X1[-1]], [X2[-1]], "s", ms=3.5, color=K)
q1.annotate("start", (X1[0], X2[0]), textcoords="offset points",
            xytext=(6, 7), fontsize=7.5)
q1.annotate("end", (X1[-1], X2[-1]), textcoords="offset points",
            xytext=(7, -2), fontsize=7.5)
q1.set_aspect("equal", adjustable="datalim")
q1.axhline(0, lw=0.5, color=G, zorder=0)
q1.axvline(0, lw=0.5, color=G, zorder=0)
q1.set_xlabel(r"$X^{1}$: S&P 500  (%)")
q1.set_ylabel(r"$X^{2}$: EURO STOXX 50  (%)")
_e = int(np.floor(np.log10(abs(best))))
q1.set_title(f"One window from {clean.Date.iloc[a].date()},   "
             rf"$A={best/10**_e:.1f}\times10^{{{_e}}}$", fontsize=8.5, pad=5)
tidy(q1)

# -- the lag scan -----------------------------------------------------
lags, means, ses = [], [], []
for l in range(-3, 4):
    if l >= 0:
        u, v = x1[:len(x1) - l], x2[l:]
    else:
        u, v = x1[-l:], x2[:len(x2) + l]
    A = np.array([window_stats(u[w * n:w * n + n + 1],
                               v[w * n:w * n + n + 1])["area_str"]
                  for w in range((len(u) - 1) // n)])
    lags.append(l)
    means.append(A.mean())
    ses.append(A.std(ddof=1) / np.sqrt(A.size))

lags = np.array(lags)
means = np.array(means) * 1e4
ses = np.array(ses) * 1e4
q2.errorbar(lags, means, yerr=2 * ses, fmt="o", ms=3.5, lw=0.9,
            capsize=2.5, color=K)
q2.axhline(0, lw=0.7, color=G)
q2.axvline(0, lw=0.6, ls=":", color=G)
q2.set_xlabel(r"imposed lag $\ell$  (trading days)")
q2.set_ylabel(r"$\overline{A}\times10^{4}$")
q2.set_title("Mean area against imposed lag", fontsize=8.5, pad=5)
q2.set_xticks(lags)
tidy(q2)

fig.tight_layout(pad=0.4)
fig.savefig(OUT + "levy-lag.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote levy-lag.pdf")
