"""
Can the calibration recover a Hurst index it is known to contain?

Section 4.2.3 shows that the log-log skew regression cannot separate two
fitted models whose H differs five-fold.  That is a statement about THAT
ESTIMATOR.  Whether the objective Phi can separate them is a different
question, and the step from one to the other is an inference rather than a
measurement.

This script measures it.  Quotes are MANUFACTURED from the model itself at a
known H, on exactly the real (k, T) grid and with Gaussian noise of one
half-spread per quote, and the calibration is then run on them with H held at
the planted value.  The truth is known because we built it.

    Phi/n at the planted parameters  ->  the noise floor, which is 1 by
                                         construction: E[(noise/h)^2] = 1
    Phi/n recovered by the search    ->  1 if the search finds the truth

Only the vols are synthetic.  The maturities, the strikes, the cone and the
half-spreads are the real ones, so the experiment inherits the geometry of the
book whose identifying power is in question -- which is the whole point.

TWO SEARCHES ARE COMPARED, and the difference matters.

  * DE + Nelder-Mead polish, the procedure behind Table 4.5, recovers the
    planted parameters to three or four significant figures.
  * The Nelder-Mead CONTINUATION of profile_h.py, the procedure behind
    Table 4.7, does not: walking H down from the smooth end it reports
    Phi/n = 19.6 at H = 0.124, where 1.003 is attainable, because the
    nuisance parameters that go with a rough kernel lie far from the warm
    start it carries down.

So Table 4.7 is an UPPER BOUND on Phi*(H).  On the real book that bound is
tight -- an independent global search at H = 0.107 returns 82.71 against the
table's 82.83, and the same parameters -- because there is no distant rough
basin to miss.  On a book that is genuinely rough there is one, and the
continuation misses it.  Both facts are reported below.

Outputs
    synthetic_recovery.csv        one row per search
    synthetic_profile.csv         the continuation profile, for contrast

Usage:  python synthetic_recovery.py [H_true] [seed]
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import rough_heston as rh
from calibrate import NAMES, BOUNDS, _obj_fixedH
from load_options import load, quotes

SUB = [b for i, b in enumerate(BOUNDS) if i != 5]

#  The five nuisance parameters are held at the market fit of Table 4.5, so
#  the synthetic book differs from the real one in H and in nothing else.
P_NUISANCE = np.array([0.023678887, 0.062474094, 2.118423187, 0.922004336,
                       -0.734739817])


def clip(p5):
    return np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(p5, SUB)])


def make_book(q, H_true, rng, N=300):
    """Synthetic mids at H_true on the real grid, perturbed by one half-spread.

    The noise is Gaussian with standard deviation equal to the quote's own
    half-spread, which is the scale at which the market itself is indifferent.
    Anything tighter would make the test easier than the data ever is.
    """
    p = np.concatenate([P_NUISANCE, [H_true]])
    iv = rh.surface_iv(q, tuple(p), N=N)
    if not np.all(np.isfinite(iv)):
        raise RuntimeError(f"pricer returned non-finite vols at H={H_true}")
    h = q["half_spread"].to_numpy()
    s = q.copy()
    s["mid"] = iv + rng.normal(0.0, h)
    s["clean"] = iv
    return s


def global_fit(q, H, seed=0, N=80, Npolish=300):
    """DE then polish at fixed H -- the procedure behind Table 4.5."""
    t0 = time.time()
    de = differential_evolution(_obj_fixedH, SUB, args=(q, N, H), seed=seed,
                                maxiter=22, popsize=10, tol=1e-8,
                                polish=False, init="sobol", workers=1)
    nm = minimize(_obj_fixedH, de.x, args=(q, Npolish, H),
                  method="Nelder-Mead", bounds=SUB,
                  options=dict(maxiter=1200, xatol=1e-6, fatol=1e-6))
    return float(nm.fun), nm.x, time.time() - t0


def continuation_profile(q, Hgrid, N=80, maxiter=220):
    """NM continuation walking H downward -- the procedure behind Table 4.7."""
    generic = np.array([0.020, 0.045, 1.0, 0.40, -0.70])
    prev, rows = P_NUISANCE.copy(), []
    for H in Hgrid[::-1]:
        best, bx = np.inf, None
        for st in (prev, generic):
            r = minimize(_obj_fixedH, clip(st), args=(q, N, H),
                         method="Nelder-Mead", bounds=SUB,
                         options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-6))
            if r.fun < best:
                best, bx = float(r.fun), r.x
        prev = bx
        fine = _obj_fixedH(bx, q, 300, H)
        rows.append(dict(H=float(H), Phi=fine, Phi_n=fine / len(q),
                         **dict(zip(NAMES[:5], bx))))
    return rows[::-1]


def main(H_true=0.10, seed=0):
    d = load("../data")
    q = quotes(d, cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
    n = len(q)
    print(f"grid: {n} quotes, {q['T'].nunique()} maturities, "
          f"T in [{q['T'].min():.4f}, {q['T'].max():.4f}]")
    print(f"median half-spread {np.median(q['half_spread'])*1e4:.1f} bp\n")

    rng = np.random.default_rng(seed)
    s = make_book(q, H_true, rng)
    dev = np.abs(s["mid"] - s["clean"]) / s["half_spread"]
    floor = _obj_fixedH(P_NUISANCE, s, 300, H_true) / n
    print(f"synthetic book at H_true = {H_true:.2f}")
    print(f"   noise          mean |dev| = {dev.mean():.2f} half-spreads")
    print(f"   Phi/n at the planted parameters = {floor:.3f}"
          f"   (the noise floor)\n", flush=True)

    rows = []
    for H, tag in ((H_true, "planted H"), (0.50, "smooth endpoint")):
        val, x, dt = global_fit(s, H)
        print(f"global fit at H = {H:.3f} ({tag}):  Phi/n = {val/n:8.3f}"
              f"   ({dt:.0f}s)", flush=True)
        rows.append(dict(book="synthetic", H=H, Phi_n=val / n,
                         **dict(zip(NAMES[:5], x))))

    print("\nrecovered against planted:")
    r0 = rows[0]
    for nm_, tv in zip(NAMES[:5], P_NUISANCE):
        print(f"   {nm_:6s}  planted {tv: .6f}   recovered {r0[nm_]: .6f}")

    pd.DataFrame(rows).to_csv("synthetic_recovery.csv", index=False)

    print("\n--- the continuation profile, for contrast ---", flush=True)
    Hgrid = np.round(np.linspace(0.03, 0.50, 11), 4)
    prof = continuation_profile(s, Hgrid)
    P = pd.DataFrame(prof)
    P.to_csv("synthetic_profile.csv", index=False)
    i = int(P.Phi_n.idxmin())
    for _, r in P.iterrows():
        print(f"   H = {r.H:5.3f}   Phi/n = {r.Phi_n:9.3f}")
    print(f"   -> continuation minimum at H = {P.H[i]:.3f} "
          f"(planted {H_true:.2f}), Phi/n = {P.Phi_n[i]:.3f}")
    print("wrote synthetic_recovery.csv and synthetic_profile.csv")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.10,
         int(sys.argv[2]) if len(sys.argv) > 2 else 0)
