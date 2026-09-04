"""
Lag scan for the Levy area of the SPX / STOXX pair.

The synchronous area came out statistically zero while a one-day artificial
misalignment produced a large, strongly significant one.  This maps the area
as a function of an imposed lag: pair x1_t with x2_{t+l} for l = -3..3.

If the estimator is measuring misalignment (and the data are genuinely
aligned at daily frequency), A(l) should pass through zero near l = 0 and
grow in magnitude with |l|, with a sign that flips with the sign of l.

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

    print("\n--- is the synchronous result robust, or crisis-driven? ---")
    A0 = res[0]
    m, se, t = summarise(A0)
    print(f"  full sample          mean {m: .4e}  t = {t:+.2f}")
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
