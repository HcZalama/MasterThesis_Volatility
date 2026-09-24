"""
Objective profiled in H on the T <= 2 sample, run alongside the main fit.

Cheaper budget than rerun_t2.py: the profile only has to show the shape of
Phi(H) and where its minimum sits, not to locate a six-parameter optimum, so
the global stage is smaller and the polish shorter. Writes incrementally.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from calibrate import NAMES, BOUNDS, _obj_fixedH

N_GLOBAL = 80
N_FINE = 150
OUTFILE = "rerun_t2_profile.csv"


def sample():
    q = pd.read_csv("quotes.csv")
    q = q[q.in_cone & (q.half_spread > 0) & (q["T"] <= 2.0)]
    return q.reset_index(drop=True)


if __name__ == "__main__":
    q = sample()
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    grid = np.round(np.linspace(0.03, 0.49, 11), 3)
    print(f"profiling on {len(q)} quotes, {len(grid)} values of H", flush=True)

    rows = []
    for H in grid:
        de = differential_evolution(_obj_fixedH, sub, args=(q, N_GLOBAL, H),
                                    seed=0, maxiter=10, popsize=6, tol=1e-8,
                                    polish=False, init="sobol", workers=1)
        nm = minimize(_obj_fixedH, de.x, args=(q, N_FINE, H),
                      method="Nelder-Mead", bounds=sub,
                      options=dict(maxiter=150, xatol=1e-5, fatol=1e-5))
        xd = _obj_fixedH(de.x, q, N_FINE, H)
        val, x = (nm.fun, nm.x) if nm.fun < xd else (xd, de.x)
        rows.append(dict(H=float(H), Phi=float(val),
                         Phi_per_n=float(val) / len(q),
                         **{n: float(v) for n, v in zip(NAMES[:5], x)}))
        pd.DataFrame(rows).to_csv(OUTFILE, index=False)
        print(f"[profile] H={H:5.3f}  Phi/n={val/len(q):9.3f}  "
              f"theta={x[1]:.4f}  kappa={x[2]:.3f}  xi={x[3]:.3f}  rho={x[4]:+.3f}",
              flush=True)
    print("done", flush=True)
