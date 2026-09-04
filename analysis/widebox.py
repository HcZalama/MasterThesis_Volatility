import numpy as np
from scipy.optimize import minimize, differential_evolution
from load_options import load, quotes
from calibrate import _obj_fixedH

d = load("../data")
q = quotes(d, cone=0.5)
q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)

# deliberately generous: theta to 80% vol, kappa over three orders, xi to 3
WIDE = [(0.0025, 0.30), (0.0025, 0.80), (0.005, 20.0), (0.02, 3.0), (-0.999, -0.05)]

for H in (0.107, 0.183, 0.490):
    de = differential_evolution(_obj_fixedH, WIDE, args=(q, 80, H), seed=1,
                                maxiter=30, popsize=12, tol=1e-8,
                                polish=False, init="sobol", workers=1)
    fd = _obj_fixedH(de.x, q, 300, H)
    nm = minimize(_obj_fixedH, de.x, args=(q, 300, H), method="Nelder-Mead",
                  bounds=WIDE, options=dict(maxiter=900, xatol=1e-6, fatol=1e-6))
    v, x = (nm.fun, nm.x) if nm.fun < fd else (fd, de.x)
    print("H=%.3f  WIDE  Phi=%9.1f  Phi/n=%7.2f   v0=%.4f th=%.4f kap=%.3f "
          "xi=%.3f rho=%.3f" % (H, v, v / len(q), *x), flush=True)
