"""
Calibration of the rough Heston model to the SPX implied volatility surface.

Objective.  Quotes are weighted by their own half bid/ask spread, so the fit
is measured in ticks rather than in volatility points:

    Phi(p) = sum_i ( (sigma_i(p) - sigma_i^mid) / h_i )^2 ,   h_i = (ask-bid)/2

A value of Phi/n below 1 means the model sits inside the quoted spread on
average.  Only quotes inside the cone |k| <= 0.5 sqrt(T) enter.

Parameters, all six free, H among them:

    v0     spot variance
    theta  long variance
    lam    mean reversion
    xi     vol of vol
    rho    spot/vol correlation
    H      Hurst index          <- optimised, not fixed

Usage
    python calibrate.py            fit, then profile the objective in H
    python calibrate.py --quick    fit only
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

import rough_heston as rh
from load_options import load, quotes

NAMES = ["v0", "theta", "lam", "xi", "rho", "H"]
BOUNDS = [(0.0025, 0.16),    # v0     (vol 5% .. 40%)
          (0.0025, 0.16),    # theta
          (0.05, 10.0),      # lam
          (0.05, 1.5),       # xi
          (-0.999, -0.05),   # rho
          (0.02, 0.50)]      # H


# ----------------------------------------------------------------------

def objective(p, q, N=300, ridge=0.0):
    """Sum of squared errors in half-spread units."""
    iv = rh.surface_iv(q, tuple(p), N=N)
    if not np.all(np.isfinite(iv)):
        return 1e12 * (1 + np.sum(~np.isfinite(iv)))
    r = (iv - q["mid"].to_numpy()) / q["half_spread"].to_numpy()
    return float(np.sum(r**2)) + ridge


def report(p, q, N=400, label=""):
    iv = rh.surface_iv(q, tuple(p), N=N)
    mid = q["mid"].to_numpy()
    h = q["half_spread"].to_numpy()
    err = iv - mid
    inside = np.mean(np.abs(err) <= h) * 100
    print(f"\n--- {label} ---")
    for n, v in zip(NAMES, p):
        print(f"   {n:6s} = {v: .5f}")
    print(f"   alpha = H + 1/2 = {p[5] + 0.5:.4f}")
    print(f"   Phi          = {np.sum((err/h)**2):.1f}   "
          f"Phi/n = {np.mean((err/h)**2):.3f}")
    print(f"   RMSE (vol)   = {np.sqrt(np.mean(err**2))*100:.4f} vol pts")
    print(f"   max |err|    = {np.max(np.abs(err))*100:.4f} vol pts")
    print(f"   inside quote = {inside:.1f}% of {len(q)}")
    print("\n   per maturity:")
    print("     T      n    rmse(bp)  bias(bp)  inside%")
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        print(f"   {T:6.4f} {m.sum():4d}   {np.sqrt(np.mean(err[m]**2))*1e4:7.1f}"
              f"  {np.mean(err[m])*1e4:+8.1f}  {np.mean(np.abs(err[m])<=h[m])*100:6.1f}")
    return iv


# ----------------------------------------------------------------------

def _obj_fixedH(p5, q, N, H):
    """Module level (not a closure) so it survives pickling to DE workers."""
    return objective(np.concatenate([p5, [H]]), q, N=N)


def fit(q, seed=0, N=80, Npolish=300, maxiter=40, popsize=12):
    """Coarse global search, then a polish at the accurate resolution.

    Process pools cannot be spawned here, and the Riccati loop is GIL bound,
    so this runs single threaded: the global stage therefore uses a coarse
    time grid (price error ~1e-5, well inside a half spread) and only the
    local polish pays for the fine one.
    """
    t0 = time.time()
    print(f"global stage (differential evolution, N={N})...", flush=True)
    de = differential_evolution(
        objective, BOUNDS, args=(q, N), seed=seed, maxiter=maxiter,
        popsize=popsize, tol=1e-8, mutation=(0.4, 1.0), recombination=0.8,
        polish=False, init="sobol", disp=True, workers=1)
    print(f"   best Phi = {de.fun:.2f}   ({time.time()-t0:.0f} s)", flush=True)

    print(f"local polish (Nelder-Mead, N={Npolish})...", flush=True)
    nm = minimize(objective, de.x, args=(q, Npolish), method="Nelder-Mead",
                  bounds=BOUNDS,
                  options=dict(maxiter=2000, xatol=1e-7, fatol=1e-7))
    xd = objective(de.x, q, N=Npolish)
    best_x, best_f = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)
    print(f"   best Phi = {best_f:.2f}   total {time.time()-t0:.0f} s", flush=True)
    return np.asarray(best_x), float(best_f)


def profile_H(q, Hgrid, N=80, Npolish=300, seed=0):
    """Minimise over the other five parameters with H held fixed.

    H is often weakly identified by a single surface; this shows how sharply
    the data actually pin it down.
    """
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    rows = []
    for H in Hgrid:
        de = differential_evolution(_obj_fixedH, sub, args=(q, N, H), seed=seed,
                                    maxiter=22, popsize=10, tol=1e-8,
                                    polish=False, init="sobol", workers=1)
        nm = minimize(_obj_fixedH, de.x, args=(q, Npolish, H),
                      method="Nelder-Mead", bounds=sub,
                      options=dict(maxiter=1200, xatol=1e-6, fatol=1e-6))
        xd = _obj_fixedH(de.x, q, Npolish, H)
        val, x = (nm.fun, nm.x) if nm.fun < xd else (xd, de.x)
        rows.append(dict(H=H, Phi=val, **dict(zip(NAMES[:5], x))))
        print(f"   H = {H:5.3f}   Phi = {val:10.2f}   Phi/n = {val/len(q):7.3f}",
              flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    d = load("../data")
    q = quotes(d, cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
    print(f"calibrating on {len(q)} quotes, "
          f"{q['T'].nunique()} maturities, "
          f"T in [{q['T'].min():.4f}, {q['T'].max():.4f}]")
    print(f"half-spread: median {q.half_spread.median()*1e4:.1f} bp, "
          f"min {q.half_spread.min()*1e4:.1f} bp")

    p, phi = fit(q)
    report(p, q, N=400, label="rough Heston, six free parameters")
    np.savetxt("fit_params.txt", p)

    if "--quick" not in sys.argv:
        print("\nprofiling the objective in H (five parameters re-optimised)...")
        prof = profile_H(q, np.round(np.arange(0.03, 0.501, 0.03), 3))
        prof.to_csv("H_profile.csv", index=False)
        print("\nwrote H_profile.csv")
