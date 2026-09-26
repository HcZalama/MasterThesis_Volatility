"""Does a GLOBAL search recover H on a book built to contain it?

The Nelder-Mead continuation of profile_h.py does not (it reports 19.6 at
H=0.124 where 1.003 is attainable).  This asks the same question of the
search the headline fit actually uses: differential evolution, then polish.
"""
import sys, time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from calibrate import BOUNDS, _obj_fixedH, NAMES
from load_options import load, quotes
from synthetic_recovery import make_book, P_NUISANCE

SUB = [b for i, b in enumerate(BOUNDS) if i != 5]
H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10

d = load("../data"); q = quotes(d, cone=0.5)
q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
n = len(q)
rng = np.random.default_rng(0)
s = make_book(q, 0.10, rng)          # the same book the profile ran on
print(f"synthetic book at H_true = 0.10, {n} quotes", flush=True)
print(f"Phi/n at the true parameters      = "
      f"{_obj_fixedH(P_NUISANCE, s, 300, 0.10)/n:8.3f}", flush=True)

t0 = time.time()
de = differential_evolution(_obj_fixedH, SUB, args=(s, 80, H), seed=0,
                            maxiter=22, popsize=10, tol=1e-8,
                            polish=False, init="sobol", workers=1)
print(f"DE  at H={H:.3f}  Phi/n = {de.fun/n:8.3f}   ({time.time()-t0:5.0f}s)",
      flush=True)
nm = minimize(_obj_fixedH, de.x, args=(s, 300, H), method="Nelder-Mead",
              bounds=SUB, options=dict(maxiter=1200, xatol=1e-6, fatol=1e-6))
print(f"NM  at H={H:.3f}  Phi/n = {nm.fun/n:8.3f}   ({time.time()-t0:5.0f}s)")
for nm_, v in zip(NAMES[:5], nm.x):
    print(f"   {nm_:6s} = {v: .5f}")
