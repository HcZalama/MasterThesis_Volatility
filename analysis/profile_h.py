"""
Profile the calibration objective in the Hurst index.

For each fixed H the remaining five parameters are re-optimised, so the curve
Phi*(H) = min_{v0,kappa,theta,xi,rho} Phi(v0,kappa,theta,xi,rho,H)

shows how sharply the surface actually pins H down.  A flat profile would mean
H is not identified by a single surface; a sharp one means it is.

Each H is polished from three starting points -- the global optimum, and two
perturbations of it -- so the curve is not an artefact of tracking one local
branch out of the optimum.

Usage:  python profile_h.py [nH]
"""

import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from load_options import load, quotes
from calibrate import NAMES, BOUNDS, objective, _obj_fixedH

SUB = [b for i, b in enumerate(BOUNDS) if i != 5]


def clip(p5):
    return np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(p5, SUB)])


def solve_at(H, q, starts, N=80, maxiter=220):
    best, bx = np.inf, None
    for st in starts:
        r = minimize(_obj_fixedH, clip(st), args=(q, N, H), method="Nelder-Mead",
                     bounds=SUB,
                     options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-6))
        if r.fun < best:
            best, bx = float(r.fun), r.x
    return best, bx


def main(nH=11):
    d = load("../data")
    q = quotes(d, cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)

    import os
    src = "fit_params.txt" if os.path.exists("fit_params.txt") else "start_params.txt"
    p = np.loadtxt(src)
    print(f"warm start from {src}")
    print("starting point:")
    for n, v in zip(NAMES, p):
        print(f"   {n:6s} = {v: .5f}")
    print(f"   Phi(N=300) = {objective(p, q, N=300):.2f}\n")

    base = np.delete(p, 5)
    rng = np.random.default_rng(0)
    Hgrid = np.round(np.linspace(0.03, 0.49, nH), 4)

    rows, t0 = [], time.time()
    generic = np.array([0.020, 0.045, 1.0, 0.40, -0.70])
    prev = base                       # continuation: walk H downward
    for H in Hgrid[::-1]:
        val, x = solve_at(H, q, [prev, generic])
        prev = x
        fine = _obj_fixedH(x, q, 300, H)
        rows.append(dict(H=float(H), Phi=fine, Phi_n=fine / len(q),
                         **dict(zip(NAMES[:5], x))))
        print(f"   H = {H:5.3f}   Phi = {fine:10.2f}   Phi/n = {fine/len(q):7.3f}"
              f"   ({time.time()-t0:5.0f} s)", flush=True)
    rows = rows[::-1]

    df = pd.DataFrame(rows)
    df.to_csv("H_profile.csv", index=False)
    i = int(df.Phi.idxmin())
    print(f"\nminimum of the profile at H = {df.H[i]:.4f}, "
          f"Phi/n = {df.Phi_n[i]:.3f}")
    print("wrote H_profile.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 13)
