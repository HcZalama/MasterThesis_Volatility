import numpy as np, pandas as pd
from scipy.optimize import minimize
from load_options import load, quotes
from calibrate import _obj_fixedH

d = load("../data"); q = quotes(d, cone=0.5)
q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
pr = pd.read_csv("H_profile.csv")

# open theta and kappa well beyond the original box; keep xi moderate so the
# solver stays in its fast path
WIDE = [(0.0025, 0.30), (0.0025, 0.80), (0.005, 20.0), (0.05, 1.5), (-0.999, -0.05)]

for Htgt in (0.490, 0.375, 0.260):
    r = pr.loc[(pr.H - Htgt).abs().idxmin()]
    H = float(r.H)
    x0 = np.array([r.v0, r.theta, r.lam, r.xi, r.rho])
    base = _obj_fixedH(x0, q, 300, H)
    nm = minimize(_obj_fixedH, x0, args=(q, 150, H), method="Nelder-Mead",
                  bounds=WIDE, options=dict(maxiter=600, xatol=1e-6, fatol=1e-6))
    fine = _obj_fixedH(nm.x, q, 300, H)
    print("H=%.3f  narrow Phi/n=%7.3f -> wide Phi/n=%7.3f   "
          "theta %.4f -> %.4f   kappa %.3f -> %.3f"
          % (H, base/len(q), fine/len(q), r.theta, nm.x[1], r.lam, nm.x[2]),
          flush=True)
