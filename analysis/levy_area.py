"""
Levy area of a pair of index total-return series.

Computes, over non-overlapping windows, the level-two signature of the planar
path (X^1, X^2) of cumulative log-returns, and separates it into

    Sym  = 1/2 * X^1_T * X^2_T          (determined by the endpoints; Prop 3.78)
    A    = 1/2 * (XX^{12} - XX^{21})    (the Levy area; the only free datum)

and reports the three exact structural identities that the theory predicts,
plus the estimate of E[A] and the lead-lag coefficient it implies.

Usage:  python levy_area.py ../data/sp500_stoxx50_total_return.csv
"""

import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Level-two signature of a two-dimensional discrete path.
#
# Two quadrature rules are computed deliberately.  Left-point sums are the
# Ito lift; trapezoidal sums are the Stratonovich (geometric) lift.  The
# symmetric parts differ by the realised covariation, the antisymmetric
# parts are identical -- which is Remark 3.80, and is checked below.
# ----------------------------------------------------------------------

def level_two(x1, x2):
    """Return (XX12_ito, XX21_ito, XX12_str, XX21_str) for one window.

    x1, x2 are the levels; increments are taken from the first observation,
    so that X^i_0 = 0 as the signature requires.
    """
    X1 = x1 - x1[0]
    X2 = x2 - x2[0]
    d1 = np.diff(X1)
    d2 = np.diff(X2)

    XX12_ito = float(np.sum(X1[:-1] * d2))
    XX21_ito = float(np.sum(X2[:-1] * d1))

    # trapezoidal = left-point + half the increment product
    half_cov = 0.5 * float(np.sum(d1 * d2))
    return XX12_ito, XX21_ito, XX12_ito + half_cov, XX21_ito + half_cov


def window_stats(x1, x2):
    XX12_i, XX21_i, XX12_s, XX21_s = level_two(x1, x2)
    X1T = float(x1[-1] - x1[0])
    X2T = float(x2[-1] - x2[0])
    return dict(
        area_ito=0.5 * (XX12_i - XX21_i),
        area_str=0.5 * (XX12_s - XX21_s),
        sym_str=0.5 * (XX12_s + XX21_s),
        endpoint_sym=0.5 * X1T * X2T,
        X1T=X1T,
        X2T=X2T,
    )


def chen_defect(x1, x2):
    """A_{s,t} - A_{s,u} - A_{u,t} minus the triangle determinant (Prop 3.81).

    Should be zero to machine precision for any split u.
    """
    n = len(x1)
    u = n // 2
    A_st = window_stats(x1, x2)["area_str"]
    A_su = window_stats(x1[: u + 1], x2[: u + 1])["area_str"]
    A_ut = window_stats(x1[u:], x2[u:])["area_str"]
    c1_su, c2_su = x1[u] - x1[0], x2[u] - x2[0]
    c1_ut, c2_ut = x1[-1] - x1[u], x2[-1] - x2[u]
    tri = 0.5 * (c1_su * c2_ut - c2_su * c1_ut)
    return A_st - A_su - A_ut - tri


# ----------------------------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    df["x1"] = np.log(df["SP500_TR"].astype(float))
    df["x2"] = np.log(df["STOXX50_TR"].astype(float))
    return df


def drop_stale(df):
    """Remove rows where either market did not move at all.

    Holidays are forward-filled in the source, so a stale day contributes an
    increment to one leg and not the other, which biases the area.  The first
    row is always kept.
    """
    d1 = df["x1"].diff()
    d2 = df["x2"].diff()
    stale = (d1 == 0) | (d2 == 0)
    stale.iloc[0] = False
    return df.loc[~stale].reset_index(drop=True), int(stale.sum())


def run(df, n, label):
    x1 = df["x1"].to_numpy()
    x2 = df["x2"].to_numpy()
    nwin = (len(x1) - 1) // n

    rows, defects = [], []
    for w in range(nwin):
        a, b = w * n, w * n + n
        s = window_stats(x1[a : b + 1], x2[a : b + 1])
        rows.append(s)
        defects.append(chen_defect(x1[a : b + 1], x2[a : b + 1]))

    R = pd.DataFrame(rows)
    M = len(R)

    # --- exact identities predicted by the theory -----------------------
    geo_err = np.max(np.abs(R.sym_str - R.endpoint_sym))       # Prop 3.78
    ito_str_err = np.max(np.abs(R.area_str - R.area_ito))      # Remark 3.80
    chen_err = np.max(np.abs(defects))                         # Prop 3.81

    # --- the measurement ------------------------------------------------
    A = R.area_str.to_numpy()
    Abar = A.mean()
    se = A.std(ddof=1) / np.sqrt(M)
    tstat = Abar / se if se > 0 else np.nan

    # dimensionless: area against the typical size of the endpoint product
    scale = np.sqrt((R.X1T**2).mean() * (R.X2T**2).mean())
    print(f"\n=== {label}:  window = {n} obs,  {M} windows ===")
    print(f"  geometricity  max|Sym - X1T*X2T/2| = {geo_err:.3e}")
    print(f"  Ito vs Strat  max|A_str - A_ito|   = {ito_str_err:.3e}")
    print(f"  Chen defect   max|.|               = {chen_err:.3e}")
    print(f"  mean A       = {Abar: .6e}   (x1e4: {Abar*1e4: .4f})")
    print(f"  s.e.         = {se: .6e}    t = {tstat: .2f}")
    print(f"  A / scale    = {Abar/scale: .4f}")
    print(f"  mean Sym     = {R.sym_str.mean(): .6e}")
    return dict(n=n, M=M, Abar=Abar, se=se, t=tstat, norm=Abar / scale,
                geo=geo_err, itostr=ito_str_err, chen=chen_err)


def main(path):
    df = load(path)
    print(f"rows: {len(df)}   {df.Date.iloc[0].date()} .. {df.Date.iloc[-1].date()}")

    clean, ndrop = drop_stale(df)
    print(f"stale (one market shut) rows dropped: {ndrop}  ->  {len(clean)} rows")

    out = []
    for n in (5, 10, 21, 63):
        out.append(run(clean, n, "SPX / STOXX50, synchronous pairing"))

    # --- control: re-pair SPX_t with STOXX_{t+1} ------------------------
    # If the measured area is the 4.5h closing-time offset, aligning the two
    # legs should collapse it.  If it survives, it is not a timing artefact.
    lag = clean.copy()
    lag["x2"] = lag["x2"].shift(-1)
    lag = lag.dropna().reset_index(drop=True)
    run(lag, 21, "CONTROL: STOXX advanced one day")

    # --- placebo: reverse one leg in time -------------------------------
    rev = clean.copy()
    rev["x2"] = np.log(np.asarray(rev["STOXX50_TR"])[::-1].astype(float))
    run(rev, 21, "PLACEBO: STOXX time-reversed")

    pd.DataFrame(out).to_csv("levy_area_results.csv", index=False)
    print("\nwrote levy_area_results.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "../data/sp500_stoxx50_total_return.csv")
