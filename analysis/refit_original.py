"""
Is the smooth endpoint a local optimum of the SIX-parameter problem?

The classical control (H pinned at 1/2) beat the six-parameter fit reported in
Table 4.4, which cannot happen at a converged optimum. Rather than repeat the
whole global search, this starts a local search from the control's point with
all six parameters free and sees whether H moves off 1/2 -- and, separately,
profiles the objective in H near the endpoint.

Writes refit_original.csv and refit_original_endprofile.csv.
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import rough_heston as rh
from calibrate import NAMES, BOUNDS, objective, _obj_fixedH
from load_options import load, quotes

if __name__ == "__main__":
    d = load("../data")
    q = quotes(d, cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
    n = len(q)
    print(f"ORIGINAL sample: {n} quotes, {q['T'].nunique()} maturities\n", flush=True)

    ctrl = pd.read_csv("classic_control.csv").iloc[0]
    x0 = np.array([ctrl.v0, ctrl.theta, ctrl.lam, ctrl.xi, ctrl.rho, 0.5])
    thesis = np.array([0.02370, 0.06354, 2.0033, 0.8968, -0.7346, 0.4900])

    print("reference points, both at N=300:", flush=True)
    print(f"   thesis Table 4.4 (H=0.4900)   Phi/n = {objective(thesis,q,N=300)/n:.4f}",
          flush=True)
    print(f"   classical control (H=0.5)     Phi/n = {objective(x0,q,N=300)/n:.4f}\n",
          flush=True)

    t0 = time.time()
    nm = minimize(objective, x0, args=(q, 300), method="Nelder-Mead",
                  bounds=BOUNDS,
                  options=dict(maxiter=4000, xatol=1e-9, fatol=1e-9))
    print(f"six-parameter local search from the control point:", flush=True)
    print(f"   Phi/n = {nm.fun/n:.4f}   H = {nm.x[5]:.6f}   "
          f"({time.time()-t0:.0f}s)", flush=True)
    for name, v in zip(NAMES, nm.x):
        print(f"      {name:6s} {v: .6f}", flush=True)

    iv = rh.surface_iv(q, tuple(nm.x), N=400)
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy(); err = iv - mid
    row = dict(model="rough Heston (H free), local refit from control",
               **{k: float(v) for k, v in zip(NAMES, nm.x)},
               Phi=float(np.sum((err / h) ** 2)),
               Phi_per_n=float(np.mean((err / h) ** 2)),
               rmse_bp=float(np.sqrt(np.mean(err ** 2)) * 1e4),
               inside_pct=float(np.mean(np.abs(err) <= h) * 100), n=n)
    pd.DataFrame([row]).to_csv("refit_original.csv", index=False)

    print("\nobjective near the endpoint, other five re-optimised:", flush=True)
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    rows = []
    for H in (0.44, 0.46, 0.48, 0.49, 0.50):
        r = minimize(_obj_fixedH, np.array(x0[:5]), args=(q, 300, H),
                     method="Nelder-Mead", bounds=sub,
                     options=dict(maxiter=1500, xatol=1e-7, fatol=1e-7))
        rows.append(dict(H=H, Phi=float(r.fun), Phi_per_n=float(r.fun) / n))
        print(f"   H = {H:.3f}   Phi/n = {r.fun/n:.4f}", flush=True)
        pd.DataFrame(rows).to_csv("refit_original_endprofile.csv", index=False)
    print("\ndone", flush=True)
