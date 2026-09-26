"""
Lag scan for the Levy area of the SPX / STOXX pair, with the classical
cross-correlogram computed alongside it.

The synchronous area came out statistically zero while a one-day artificial
misalignment produced a large, strongly significant one.  This maps the area
as a function of an imposed lag: pair x1_t with x2_{t+l} for l = -3..3.

If the estimator is measuring misalignment (and the data are genuinely
aligned at daily frequency), A(l) should pass through zero near l = 0 and
grow in magnitude with |l|, with a sign that flips with the sign of l.

WHAT THE CROSS-CORRELOGRAM ADDS.  The area is not an independent statistic:
for a linearly interpolated path with increments d1, d2 over a window of n
steps, the left-point level-two entries are

    XX12 = sum_{i<j} d1_i d2_j ,      XX21 = sum_{i<j} d2_i d1_j ,

so with g(h) = E[d1_t d2_{t+h}] and stationarity,

    E[A] = 1/2 sum_{h=1}^{n-1} (n-h) [ g(h) - g(-h) ] .                 (*)

Only the ANTISYMMETRIC part of the cross-covariance survives: g(0) cancels,
which is Remark "Correlation is invisible to the area" read on data.  (*) is
evaluated below and compared with the measured mean, so the two routes to the
same question can be checked against each other rather than merely contrasted.

Also reports a trimmed mean and a block bootstrap, to check that nothing is
being driven by a handful of crisis windows.

Usage:  python levy_lag.py ../data/sp500_stoxx50_total_return.csv
"""

import sys
import numpy as np
import pandas as pd

from levy_area import load, drop_stale, window_stats


def areas(x1, x2, n):
    """Level-two areas over consecutive non-overlapping windows of n steps."""
    out = []
    for w in range((len(x1) - 1) // n):
        a, b = w * n, w * n + n
        s = window_stats(x1[a : b + 1], x2[a : b + 1])
        out.append((s["area_str"], s["X1T"], s["X2T"]))
    return np.array(out)


def summarise(A):
    m = A.mean()
    se = A.std(ddof=1) / np.sqrt(A.size)
    return m, se, (m / se if se > 0 else np.nan)


def block_bootstrap(A, B=20000, seed=0):
    """Circular block bootstrap of the mean, block length ~ sqrt(M)."""
    rng = np.random.default_rng(seed)
    M = A.size
    L = max(2, int(np.sqrt(M)))
    nb = int(np.ceil(M / L))
    idx = (rng.integers(0, M, size=(B, nb))[:, :, None]
           + np.arange(L)[None, None, :]) % M
    return A[idx.reshape(B, -1)[:, :M]].mean(axis=1)


# ----------------------------------------------------------------------
#  The classical route: the cross-covariance of the daily increments.
#
#  g(h) = E[ d1_t d2_{t+h} ].  A POSITIVE h means asset 1 leads asset 2,
#  the same orientation as a positive imposed lag in the scan above.
# ----------------------------------------------------------------------

def xcov(d1, d2, h):
    """Cross-covariance of the increments at lag h, about their own means."""
    a = d1 - d1.mean()
    b = d2 - d2.mean()
    if h >= 0:
        u, v = a[: len(a) - h], b[h:]
    else:
        u, v = a[-h:], b[: len(b) + h]
    return float((u * v).mean())


def cross_correlogram(d1, d2, L=5):
    """corr(d1_t, d2_{t+h}) for h = -L..L, with the Bartlett 1/sqrt(N) band."""
    s = np.sqrt(d1.var() * d2.var())
    N = len(d1)
    se = 1.0 / np.sqrt(N)
    rows = []
    for h in range(-L, L + 1):
        c = xcov(d1, d2, h)
        r = c / s if s > 0 else np.nan
        rows.append((h, c, r, r / se if se > 0 else np.nan))
    return rows, se, N


def predicted_area(d1, d2, n):
    """E[A] implied by the cross-covariances through (*), and the lag-1 term.

    Returns (full sum over h = 1..n-1, the h = 1 term alone).
    """
    full = 0.5 * sum((n - h) * (xcov(d1, d2, h) - xcov(d1, d2, -h))
                     for h in range(1, n))
    lag1 = 0.5 * (n - 1) * (xcov(d1, d2, 1) - xcov(d1, d2, -1))
    return full, lag1


def main(path, n=21):
    df = load(path)
    clean, ndrop = drop_stale(df)
    print(f"rows {len(df)}, stale dropped {ndrop}, used {len(clean)}")
    print(f"window = {n} observations\n")

    x1 = clean["x1"].to_numpy()
    x2 = clean["x2"].to_numpy()

    print("  lag    windows     mean A        s.e.       t      A/scale")
    res = {}
    for l in range(-3, 4):
        if l >= 0:
            a, b = x1[: len(x1) - l], x2[l:]
        else:
            a, b = x1[-l:], x2[: len(x2) + l]
        R = areas(a, b, n)
        A = R[:, 0]
        m, se, t = summarise(A)
        scale = np.sqrt((R[:, 1] ** 2).mean() * (R[:, 2] ** 2).mean())
        res[l] = A
        print(f"  {l:+3d}    {A.size:5d}   {m: .4e}  {se:.3e}  {t:+6.2f}   "
              f"{m/scale:+.4f}")

    # ------------------------------------------------------------------
    #  The classical comparison the area is supposed to improve upon.
    # ------------------------------------------------------------------
    d1 = np.diff(x1)
    d2 = np.diff(x2)
    rows, se_b, N = cross_correlogram(d1, d2, L=5)

    print("\n--- cross-correlogram of the daily increments ---")
    print(f"  N = {N} increments;  Bartlett s.e. ~ 1/sqrt(N) = {se_b:.4f}")
    print("  h>0 means SPX leads STOXX, the same orientation as lag>0 above.\n")
    print("    h     cov(d1_t, d2_{t+h})     corr        t")
    for h, c, r, t in rows:
        star = "  <-- contemporaneous" if h == 0 else ""
        print(f"  {h:+3d}       {c: .6e}      {r:+.4f}   {t:+6.2f}{star}")

    c_p1 = xcov(d1, d2, +1)
    c_m1 = xcov(d1, d2, -1)
    print(f"\n  lag-one pair:  c12 = {c_p1: .6e}   c21 = {c_m1: .6e}")
    print(f"  antisymmetric part  (c12 - c21)/2 = {(c_p1 - c_m1) / 2: .6e}")
    print(f"  symmetric   part  (c12 + c21)/2 = {(c_p1 + c_m1) / 2: .6e}")

    # ------------------------------------------------------------------
    #  Do the two routes agree?  (*) predicts the mean area from the
    #  correlogram alone; compare it with what the area actually measured.
    # ------------------------------------------------------------------
    full, lag1 = predicted_area(d1, d2, n)
    A0 = res[0]
    m0, se0, t0 = summarise(A0)
    print("\n--- the two routes, on the same data ---")
    print(f"  measured   mean A                      {m0: .4e}"
          f"   (s.e. {se0:.3e},  t = {t0:+.2f})")
    print(f"  predicted  by (*), all h = 1..{n-1:<2d}          {full: .4e}")
    print(f"  predicted  by the lag-one term alone   {lag1: .4e}")
    print(f"  measured - predicted(full)             {m0 - full: .4e}"
          f"   ({abs(m0 - full) / se0:.2f} s.e.)")

    print("\n--- is the synchronous result robust, or crisis-driven? ---")
    print(f"  full sample          mean {m0: .4e}  t = {t0:+.2f}")
    q = np.quantile(np.abs(A0), 0.95)
    Atr = A0[np.abs(A0) <= q]
    mt, set_, tt = summarise(Atr)
    print(f"  drop |A| top 5%      mean {mt: .4e}  t = {tt:+.2f}  "
          f"({Atr.size} windows)")
    bs = block_bootstrap(A0)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"  block bootstrap 95%  [{lo: .4e}, {hi: .4e}]  "
          f"contains 0: {bool(lo <= 0 <= hi)}")

    print("\n--- same for the +1 control ---")
    A1 = res[1]
    m, se, t = summarise(A1)
    q = np.quantile(np.abs(A1), 0.95)
    Atr = A1[np.abs(A1) <= q]
    mt, _, tt = summarise(Atr)
    bs = block_bootstrap(A1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    print(f"  full sample          mean {m: .4e}  t = {t:+.2f}")
    print(f"  drop |A| top 5%      mean {mt: .4e}  t = {tt:+.2f}")
    print(f"  block bootstrap 95%  [{lo: .4e}, {hi: .4e}]  "
          f"contains 0: {bool(lo <= 0 <= hi)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "../data/sp500_stoxx50_total_return.csv")
