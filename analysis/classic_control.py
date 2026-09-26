"""
Classical Heston control on the ORIGINAL calibration sample.

Same book, same cone, same objective, same pricer and same optimiser as the
rough fit reported in Table 4.4 -- the only difference is that H is pinned at
1/2 instead of being optimised. That makes the two rows a controlled
comparison rather than two separate studies.

Writes classic_control.csv and classic_control_permat.csv.
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

import rough_heston as rh
from calibrate import NAMES, BOUNDS, _obj_fixedH
from load_options import load, quotes

H_FIXED = 0.5


def stats(p, q, label):
    iv = rh.surface_iv(q, tuple(p), N=400)
    mid = q["mid"].to_numpy()
    h = q["half_spread"].to_numpy()
    err = iv - mid
    out = dict(model=label,
               **{n: float(v) for n, v in zip(NAMES, p)},
               Phi=float(np.sum((err / h) ** 2)),
               Phi_per_n=float(np.mean((err / h) ** 2)),
               rmse_bp=float(np.sqrt(np.mean(err ** 2)) * 1e4),
               max_abs_bp=float(np.max(np.abs(err)) * 1e4),
               inside_pct=float(np.mean(np.abs(err) <= h) * 100),
               n=int(len(q)))
    for k, v in out.items():
        print(f"    {k:12s} {v}", flush=True)
    return out, iv


if __name__ == "__main__":
    d = load("../data")
    q = quotes(d, cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
    print(f"ORIGINAL sample: {len(q)} quotes, {q['T'].nunique()} maturities, "
          f"T in [{q['T'].min():.4f}, {q['T'].max():.4f}]", flush=True)

    t0 = time.time()
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    de = differential_evolution(
        _obj_fixedH, sub, args=(q, 80, H_FIXED), seed=0, maxiter=40, popsize=12,
        tol=1e-8, mutation=(0.4, 1.0), recombination=0.8, polish=False,
        init="sobol", disp=False, workers=1)
    print(f"DE  Phi/n = {de.fun/len(q):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    nm = minimize(_obj_fixedH, de.x, args=(q, 300, H_FIXED),
                  method="Nelder-Mead", bounds=sub,
                  options=dict(maxiter=2000, xatol=1e-7, fatol=1e-7))
    xd = _obj_fixedH(de.x, q, 300, H_FIXED)
    x5, f = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)
    print(f"polished Phi/n = {f/len(q):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    p = np.concatenate([x5, [H_FIXED]])
    print("\n--- classical Heston, H = 1/2, original sample ---", flush=True)
    s, iv = stats(p, q, "classical Heston (H = 1/2)")
    pd.DataFrame([s]).to_csv("classic_control.csv", index=False)

    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    rows = []
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        e = iv[m] - mid[m]
        rows.append(dict(T=float(T), n=int(m.sum()),
                         rmse_bp=float(np.sqrt(np.mean(e ** 2)) * 1e4),
                         bias_bp=float(np.mean(e) * 1e4),
                         inside=float(np.mean(np.abs(e) <= h[m]) * 100),
                         phin=float(np.mean((e / h[m]) ** 2))))
    pd.DataFrame(rows).to_csv("classic_control_permat.csv", index=False)
    print("\nwrote classic_control.csv and classic_control_permat.csv", flush=True)
