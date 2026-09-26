"""
Multi-start check on the T <= 2 sample.

The headline fit used a single differential-evolution seed. Review item 3 asks
whether the optimum is a property of the data or of that one search, so this
repeats the whole fit from independent seeds and reports where each lands.

Usage:  python multistart_t2.py <seed>
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from calibrate import NAMES, BOUNDS, objective


def sample():
    q = pd.read_csv("quotes.csv")
    q = q[q.in_cone & (q.half_spread > 0) & (q["T"] <= 2.0)]
    return q.reset_index(drop=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    q = sample()
    t0 = time.time()

    de = differential_evolution(
        objective, BOUNDS, args=(q, 80), seed=seed, maxiter=40, popsize=12,
        tol=1e-8, mutation=(0.4, 1.0), recombination=0.8, polish=False,
        init="sobol", disp=False, workers=1)
    print(f"[seed {seed}] DE  Phi/n = {de.fun/len(q):9.4f}  ({time.time()-t0:.0f}s)",
          flush=True)

    nm = minimize(objective, de.x, args=(q, 300), method="Nelder-Mead",
                  bounds=BOUNDS, options=dict(maxiter=2000, xatol=1e-7, fatol=1e-7))
    xd = objective(de.x, q, N=300)
    x, f = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)

    row = dict(seed=seed, Phi=f, Phi_per_n=f / len(q),
               **{n: float(v) for n, v in zip(NAMES, x)})
    pd.DataFrame([row]).to_csv(f"multistart_seed{seed}.csv", index=False)
    print(f"[seed {seed}] fit Phi/n = {f/len(q):9.4f}   "
          + "  ".join(f"{n}={v:.5f}" for n, v in zip(NAMES, x))
          + f"   ({time.time()-t0:.0f}s)", flush=True)
