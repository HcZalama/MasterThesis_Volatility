"""
Rough Heston on the PINN book, restricted to T <= 1 and the cone
|k| <= 0.5 sqrt(T + 0.1).

Same pricer, same half-spread-weighted objective and same optimiser as the
fit reported in Table 4.4; the only thing that changes is the sample. The
classical control (H pinned at 1/2) is fitted alongside so the two rows are a
controlled comparison, and the objective is then profiled in H.

Writes pinn_fit.csv, pinn_permat.csv and pinn_profile.csv.
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

import rough_heston as rh
from calibrate import NAMES, BOUNDS, objective, _obj_fixedH

N_DE, N_POLISH, N_REPORT = 80, 300, 400
SUB = [b for i, b in enumerate(BOUNDS) if i != 5]


def stats(p, q, label):
    iv = rh.surface_iv(q, tuple(p), N=N_REPORT)
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    err = iv - mid
    out = dict(model=label, **{n: float(v) for n, v in zip(NAMES, p)},
               Phi=float(np.sum((err / h) ** 2)),
               Phi_per_n=float(np.mean((err / h) ** 2)),
               rmse_bp=float(np.sqrt(np.mean(err ** 2)) * 1e4),
               max_abs_bp=float(np.max(np.abs(err)) * 1e4),
               inside_pct=float(np.mean(np.abs(err) <= h) * 100),
               n=int(len(q)))
    print(f"\n--- {label} ---", flush=True)
    for k, v in out.items():
        if k != "model":
            print(f"    {k:11s} {v}", flush=True)
    return out, iv


def permat(iv, q, label):
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    rows = []
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        e = iv[m] - mid[m]
        rows.append(dict(model=label, T=float(T), n=int(m.sum()),
                         rmse_bp=float(np.sqrt(np.mean(e ** 2)) * 1e4),
                         bias_bp=float(np.mean(e) * 1e4),
                         inside=float(np.mean(np.abs(e) <= h[m]) * 100),
                         phin=float(np.mean((e / h[m]) ** 2))))
    return rows


if __name__ == "__main__":
    q = pd.read_csv("pinn_heston_sample.csv")
    n = len(q)
    print(f"sample: {n} quotes, {q['T'].nunique()} smiles, "
          f"T in [{q['T'].min():.4f}, {q['T'].max():.4f}]", flush=True)

    # ---- rough Heston, all six free ------------------------------------
    t0 = time.time()
    de = differential_evolution(
        objective, BOUNDS, args=(q, N_DE), seed=0, maxiter=60, popsize=16,
        tol=1e-10, mutation=(0.4, 1.0), recombination=0.8, polish=False,
        init="sobol", disp=False, workers=1)
    print(f"\nrough  DE       Phi/n = {de.fun/n:9.4f}  H = {de.x[5]:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    nm = minimize(objective, de.x, args=(q, N_POLISH), method="Nelder-Mead",
                  bounds=BOUNDS, options=dict(maxiter=4000, xatol=1e-9, fatol=1e-9))
    xd = objective(de.x, q, N=N_POLISH)
    xr, fr = (nm.x, nm.fun) if nm.fun < xd else (de.x, xd)
    print(f"rough  polished Phi/n = {fr/n:9.4f}  H = {xr[5]:.6f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    sr, ivr = stats(xr, q, "rough Heston (H free)")

    # ---- classical control, H = 1/2 ------------------------------------
    t1 = time.time()
    dec = differential_evolution(
        _obj_fixedH, SUB, args=(q, N_DE, 0.5), seed=0, maxiter=60, popsize=16,
        tol=1e-10, mutation=(0.4, 1.0), recombination=0.8, polish=False,
        init="sobol", disp=False, workers=1)
    nmc = minimize(_obj_fixedH, dec.x, args=(q, N_POLISH, 0.5),
                   method="Nelder-Mead", bounds=SUB,
                   options=dict(maxiter=3000, xatol=1e-9, fatol=1e-9))
    xdc = _obj_fixedH(dec.x, q, N_POLISH, 0.5)
    x5, fc = (nmc.x, nmc.fun) if nmc.fun < xdc else (dec.x, xdc)
    print(f"\nclassical polished Phi/n = {fc/n:9.4f}  ({time.time()-t1:.0f}s)",
          flush=True)
    xc = np.concatenate([x5, [0.5]])
    sc, ivc = stats(xc, q, "classical Heston (H = 1/2)")

    pd.DataFrame([sr, sc]).to_csv("pinn_fit.csv", index=False)
    pd.DataFrame(permat(ivr, q, "rough") + permat(ivc, q, "classical")
                 ).to_csv("pinn_permat.csv", index=False)

    print("\nper maturity (rough / classical), bias in bp:", flush=True)
    print("     T        n   rmse_r  rmse_c   bias_r  bias_c   in_r  in_c",
          flush=True)
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        er, ec = ivr[m] - mid[m], ivc[m] - mid[m]
        print(f"   {T:.5f} {m.sum():4d}  {np.sqrt(np.mean(er**2))*1e4:7.1f} "
              f"{np.sqrt(np.mean(ec**2))*1e4:7.1f}  "
              f"{np.mean(er)*1e4:7.1f} {np.mean(ec)*1e4:7.1f}  "
              f"{np.mean(np.abs(er)<=h[m])*100:5.1f} "
              f"{np.mean(np.abs(ec)<=h[m])*100:5.1f}", flush=True)

    # ---- profile in H --------------------------------------------------
    print("\nobjective profiled in H (other five re-optimised):", flush=True)
    rows = []
    for H in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        r = minimize(_obj_fixedH, np.array(xc[:5]), args=(q, N_POLISH, H),
                     method="Nelder-Mead", bounds=SUB,
                     options=dict(maxiter=2000, xatol=1e-8, fatol=1e-8))
        rows.append(dict(H=H, Phi=float(r.fun), Phi_per_n=float(r.fun) / n))
        print(f"   H = {H:.2f}   Phi/n = {r.fun/n:9.4f}", flush=True)
        pd.DataFrame(rows).to_csv("pinn_profile.csv", index=False)
    print("\ndone", flush=True)
