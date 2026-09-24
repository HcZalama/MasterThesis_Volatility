"""
Rerun of the rough Heston calibration on the quote table in quotes.csv,
restricted to T <= 2 years, together with a classical Heston control fitted
on exactly the same sample by fixing H = 1/2.

The point of the control is that it shares the pricer, the objective, the
sample and the optimiser with the rough fit, so the only thing that differs
between the two columns of the output is the kernel exponent.

Writes:
    rerun_t2_params.csv      fitted parameters, both models
    rerun_t2_permat.csv      per-maturity error decomposition, both models
    rerun_t2_profile.csv     objective profiled in H
    rerun_t2_quotes.csv      the sample with both model vols attached
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

import rough_heston as rh
from calibrate import NAMES, BOUNDS, objective, _obj_fixedH

TMAX = 2.0
N_GLOBAL = 80
N_FINE = 300
N_REPORT = 400


def sample():
    q = pd.read_csv("quotes.csv")
    q = q[q.in_cone & (q.half_spread > 0) & (q["T"] <= TMAX)]
    return q.reset_index(drop=True)


def fit_rough(q, seed=0):
    t0 = time.time()
    print(f"[rough]   differential evolution, N={N_GLOBAL} ...", flush=True)
    de = differential_evolution(
        objective, BOUNDS, args=(q, N_GLOBAL), seed=seed, maxiter=40,
        popsize=12, tol=1e-8, mutation=(0.4, 1.0), recombination=0.8,
        polish=False, init="sobol", disp=False, workers=1)
    print(f"[rough]   DE Phi = {de.fun:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    nm = minimize(objective, de.x, args=(q, N_FINE), method="Nelder-Mead",
                  bounds=BOUNDS, options=dict(maxiter=2000, xatol=1e-7, fatol=1e-7))
    xd = objective(de.x, q, N=N_FINE)
    x, f = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)
    print(f"[rough]   polished Phi = {f:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(x), float(f)


def fit_classical(q, seed=0):
    """Same everything, H pinned to 1/2."""
    t0 = time.time()
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    print(f"[classic] differential evolution, N={N_GLOBAL} ...", flush=True)
    de = differential_evolution(
        _obj_fixedH, sub, args=(q, N_GLOBAL, 0.5), seed=seed, maxiter=40,
        popsize=12, tol=1e-8, mutation=(0.4, 1.0), recombination=0.8,
        polish=False, init="sobol", disp=False, workers=1)
    print(f"[classic] DE Phi = {de.fun:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    nm = minimize(_obj_fixedH, de.x, args=(q, N_FINE, 0.5),
                  method="Nelder-Mead", bounds=sub,
                  options=dict(maxiter=2000, xatol=1e-7, fatol=1e-7))
    xd = _obj_fixedH(de.x, q, N_FINE, 0.5)
    x5, f = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)
    print(f"[classic] polished Phi = {f:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate([x5, [0.5]]), float(f)


def stats(p, q, label):
    iv = rh.surface_iv(q, tuple(p), N=N_REPORT)
    mid = q["mid"].to_numpy()
    h = q["half_spread"].to_numpy()
    err = iv - mid
    out = dict(
        model=label,
        **{n: float(v) for n, v in zip(NAMES, p)},
        Phi=float(np.sum((err / h) ** 2)),
        Phi_per_n=float(np.mean((err / h) ** 2)),
        rmse_bp=float(np.sqrt(np.mean(err ** 2)) * 1e4),
        max_abs_bp=float(np.max(np.abs(err)) * 1e4),
        inside_pct=float(np.mean(np.abs(err) <= h) * 100),
        n=int(len(q)),
    )
    print(f"\n--- {label} ---", flush=True)
    for k, v in out.items():
        if k != "model":
            print(f"    {k:12s} {v}", flush=True)
    return out, iv


def per_maturity(q, iv_r, iv_c):
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    rows = []
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        row = dict(T=float(T), n=int(m.sum()))
        for tag, iv in (("rough", iv_r), ("classic", iv_c)):
            e = iv[m] - mid[m]
            row[f"rmse_bp_{tag}"] = float(np.sqrt(np.mean(e ** 2)) * 1e4)
            row[f"bias_bp_{tag}"] = float(np.mean(e) * 1e4)
            row[f"inside_{tag}"] = float(np.mean(np.abs(e) <= h[m]) * 100)
            row[f"phin_{tag}"] = float(np.mean((e / h[m]) ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def profile(q, grid, seed=0):
    sub = [b for i, b in enumerate(BOUNDS) if i != 5]
    rows = []
    for H in grid:
        de = differential_evolution(_obj_fixedH, sub, args=(q, N_GLOBAL, H),
                                    seed=seed, maxiter=22, popsize=10, tol=1e-8,
                                    polish=False, init="sobol", workers=1)
        nm = minimize(_obj_fixedH, de.x, args=(q, N_FINE, H),
                      method="Nelder-Mead", bounds=sub,
                      options=dict(maxiter=1200, xatol=1e-6, fatol=1e-6))
        xd = _obj_fixedH(de.x, q, N_FINE, H)
        val, x = (nm.fun, nm.x) if nm.fun < xd else (xd, de.x)
        rows.append(dict(H=float(H), Phi=float(val), Phi_per_n=float(val) / len(q),
                         **{n: float(v) for n, v in zip(NAMES[:5], x)}))
        print(f"[profile] H={H:5.3f}  Phi/n={val/len(q):8.3f}", flush=True)
        pd.DataFrame(rows).to_csv("rerun_t2_profile.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    q = sample()
    print(f"sample: {len(q)} quotes, {q['T'].nunique()} maturities, "
          f"T in [{q['T'].min():.4f}, {q['T'].max():.4f}]", flush=True)

    p_r, f_r = fit_rough(q)
    p_c, f_c = fit_classical(q)

    s_r, iv_r = stats(p_r, q, "rough Heston (H free)")
    s_c, iv_c = stats(p_c, q, "classical Heston (H = 1/2)")

    pd.DataFrame([s_r, s_c]).to_csv("rerun_t2_params.csv", index=False)
    per_maturity(q, iv_r, iv_c).to_csv("rerun_t2_permat.csv", index=False)

    out = q.copy()
    out["iv_rough"] = iv_r
    out["iv_classic"] = iv_c
    out.to_csv("rerun_t2_quotes.csv", index=False)
    print("\nwrote params / permat / quotes", flush=True)

    print("\nprofiling in H ...", flush=True)
    profile(q, np.round(np.linspace(0.03, 0.49, 13), 3))
    print("done", flush=True)
