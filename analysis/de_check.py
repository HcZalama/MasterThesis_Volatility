"""Is Table 4.7's profile optimiser-limited?  Global search at a fixed H."""
import sys, time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from calibrate import BOUNDS, _obj_fixedH, NAMES
from load_options import load, quotes

SUB = [b for i, b in enumerate(BOUNDS) if i != 5]

H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.107
d = load("../data"); q = quotes(d, cone=0.5)
q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
n = len(q)
t0 = time.time()
de = differential_evolution(_obj_fixedH, SUB, args=(q, 80, H), seed=0,
                            maxiter=22, popsize=10, tol=1e-8,
                            polish=False, init="sobol", workers=1)
print(f"DE   Phi/n = {de.fun/n:8.3f}   ({time.time()-t0:5.0f}s)", flush=True)
nm = minimize(_obj_fixedH, de.x, args=(q, 300, H), method="Nelder-Mead",
              bounds=SUB, options=dict(maxiter=1200, xatol=1e-6, fatol=1e-6))
print(f"NM   Phi/n = {nm.fun/n:8.3f}   ({time.time()-t0:5.0f}s)")
for nm_, v in zip(NAMES[:5], nm.x):
    print(f"   {nm_:6s} = {v: .5f}")
