"""
Figures for the T <= 2 rerun on quotes.csv.

The free-H calibration lands on H = 1/2, so the classical control and the
"rough" fit are the same point and plotting both would show one curve. The
informative comparison is therefore the fitted model against a genuinely
rough alternative, taken from the profile at fixed H and re-optimised in the
other five parameters -- which is what the calibration rejected.

Produces
    rh-rerun-smiles.pdf    six shortest maturities, market band vs both models
    rh-rerun-compare.pdf   per-maturity error, ATM term structure, H profile
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:/Users/hecto/source/repos/HcZalama/MasterThesis_Volatility/"
OUT = ROOT + "Images/"
AN = ROOT + "analysis/"
sys.path.insert(0, AN)

import rough_heston as rh                                   # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
K, G, L, R = "#1a1a1a", "#8a8a8a", "#c8c8c8", "#b03030"


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


q = pd.read_csv(AN + "rerun_t2_quotes.csv")
pm = pd.read_csv(AN + "rerun_t2_permat.csv")
par = pd.read_csv(AN + "rerun_t2_params.csv")
prof = pd.read_csv(AN + "rerun_t2_profile.csv").sort_values("H")

fit = par.loc[par.model.str.startswith("rough")].iloc[0]
P_FIT = np.array([fit.v0, fit.theta, fit.lam, fit.xi, fit.rho, fit.H])

# the rough alternative: profile row closest to H = 0.10
r = prof.loc[(prof.H - 0.10).abs().idxmin()]
P_ROUGH = np.array([r.v0, r.theta, r.lam, r.xi, r.rho, r.H])
print(f"fitted  H={P_FIT[5]:.4f}   Phi/n={fit.Phi_per_n:.2f}")
print(f"rough   H={P_ROUGH[5]:.4f}  Phi/n={r.Phi_per_n:.2f}")

q["iv_fit"] = rh.surface_iv(q, tuple(P_FIT), N=300)
q["iv_rgh"] = rh.surface_iv(q, tuple(P_ROUGH), N=300)

# =====================================================================
# 1. the six shortest maturities
# =====================================================================
Ts = np.sort(q["T"].unique())[:6]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4))
for j, (ax, T) in enumerate(zip(axes.ravel(), Ts)):
    s = q[q["T"] == T].sort_values("k")
    first = (j == 0)
    ax.fill_between(s.k, s.bid * 100, s.ask * 100, color=L, lw=0,
                    label="market bid/ask" if first else None)
    ax.plot(s.k, s.iv_fit * 100, color=K, lw=1.3,
            label=fr"fitted, $H={P_FIT[5]:.2f}$" if first else None)
    ax.plot(s.k, s.iv_rgh * 100, color=R, lw=1.1, ls="--",
            label=fr"rough, $H={P_ROUGH[5]:.3f}$" if first else None)
    ax.set_title(fr"$T={T:.4f}$", fontsize=8.5)
    ax.set_xlabel(r"$k=\log(K/F)$")
    ax.set_ylabel(r"$\sigma_{\mathrm{imp}}$ (%)")
    tidy(ax)
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT + "rh-rerun-smiles.pdf", bbox_inches="tight")
print("wrote rh-rerun-smiles.pdf")

# =====================================================================
# 2. per-maturity error, ATM term structure, H profile
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))

ax = axes[0]
ax.plot(pm["T"], pm.rmse_bp_rough, "o-", color=K, ms=3, lw=1.1, label="RMSE")
ax.plot(pm["T"], pm.bias_bp_rough, "s--", color=R, ms=3, lw=1.0, label="bias")
ax.axhline(0, color=G, lw=0.7, ls=":")
ax.set_xscale("log"); ax.set_xlabel(r"$T$ (years)")
ax.set_ylabel("error (bp of vol)")
ax.legend(frameon=False); tidy(ax)

ax = axes[1]
atm = (q.assign(a=q.k.abs()).sort_values("a").groupby("T", as_index=False).first())
ax.plot(atm["T"], atm["mid"] * 100, "o", color=G, ms=3.5, label="market")
ax.plot(atm["T"], atm.iv_fit * 100, "-", color=K, lw=1.2,
        label=fr"fitted, $H={P_FIT[5]:.2f}$")
ax.plot(atm["T"], atm.iv_rgh * 100, "--", color=R, lw=1.0,
        label=fr"rough, $H={P_ROUGH[5]:.3f}$")
ax.set_xscale("log"); ax.set_xlabel(r"$T$ (years)")
ax.set_ylabel(r"ATM $\sigma_{\mathrm{imp}}$ (%)")
ax.legend(frameon=False); tidy(ax)

ax = axes[2]
ax.plot(prof.H, prof.Phi_per_n, "o-", color=K, ms=3, lw=1.1)
ax.axhline(fit.Phi_per_n, color=R, lw=0.9, ls="--")
ax.annotate(fr"fit at $H={P_FIT[5]:.2f}$: {fit.Phi_per_n:.1f}",
            xy=(prof.H.min(), fit.Phi_per_n), xytext=(2, 5),
            textcoords="offset points", fontsize=8, color=R)
ax.set_xlabel(r"$H$"); ax.set_ylabel(r"$\Phi/n$  (half-spreads$^2$)")
tidy(ax)

fig.tight_layout()
fig.savefig(OUT + "rh-rerun-compare.pdf", bbox_inches="tight")
print("wrote rh-rerun-compare.pdf")
