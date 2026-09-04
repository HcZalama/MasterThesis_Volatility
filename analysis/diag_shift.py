"""Which column shift aligns TotalVar with the vol sheet?

Hypothesis A:  tv[:, j]   pairs with mid[:, j+1]   -> T known for mid cols 2..17
Hypothesis B:  tv[:, j+1] pairs with mid[:, j]     -> T known for mid cols 1..16
Hypothesis 0:  tv[:, j]   pairs with mid[:, j]     -> T known for all 17

For each, report the coefficient of variation of w/sigma^2 down every column
(0 = perfect) and the implied maturity sequence.
"""
import numpy as np
from load_options import load, NMAT

d = load("../data")
r = min(d["tv_a"].shape[0], d["mid"].shape[0])
tv_a, tv_b = d["tv_a"][:r], d["tv_b"][:r]
mid, bid, ask = d["mid"][:r], d["bid"][:r], d["ask"][:r]


def col_T(w, s):
    """T = w/s^2 down one column: (median, CV, n)."""
    t = w / s**2
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan, np.nan, t.size
    return float(np.median(t)), float(t.std() / t.mean()), t.size


def test(name, pairs, tv, vol):
    """pairs = list of (tv_col, vol_col)."""
    Ts, cvs = [], []
    for jt, jv in pairs:
        T, cv, n = col_T(tv[:, jt], vol[:, jv])
        Ts.append(T)
        cvs.append(cv)
    cvs = np.array(cvs)
    Ts = np.array(Ts)
    mono = bool(np.all(np.diff(Ts[np.isfinite(Ts)]) > 0))
    print(f"\n{name}")
    print(f"   median CV = {np.nanmedian(cvs):.5f}   max CV = {np.nanmax(cvs):.5f}"
          f"   monotone T: {mono}")
    print("   T = " + ", ".join(f"{t:.5f}" for t in Ts))
    return Ts, cvs


print("=" * 74)
print("MID VOL")
test("H0  tv[j] <-> mid[j]",
     [(j, j) for j in range(NMAT)], tv_a, mid)
test("HA  tv[j] <-> mid[j+1]   (mid col 1 orphaned)",
     [(j, j + 1) for j in range(NMAT - 1)], tv_a, mid)
test("HB  tv[j+1] <-> mid[j]   (mid col 17 orphaned)",
     [(j + 1, j) for j in range(NMAT - 1)], tv_a, mid)

print("\n" + "=" * 74)
print("Does the SECOND TotalVar block (tv_b) match bid or ask?  If tv_b is")
print("bid^2*T or ask^2*T under the same shift, that pins the shift down.")
for lab, vol in (("bid", bid), ("ask", ask), ("mid", mid)):
    for hname, pairs in (
        ("H0", [(j, j) for j in range(NMAT)]),
        ("HA", [(j, j + 1) for j in range(NMAT - 1)]),
        ("HB", [(j + 1, j) for j in range(NMAT - 1)]),
    ):
        cvs = [col_T(tv_b[:, jt], vol[:, jv])[1] for jt, jv in pairs]
        print(f"   tv_b vs {lab}  {hname}:  median CV = {np.nanmedian(cvs):.5f}")

print("\n" + "=" * 74)
print("Is tv_b just a copy of tv_a?")
n = min(tv_a.shape[0], tv_b.shape[0])
diff = tv_a[:n] - tv_b[:n]
print(f"   max|tv_a - tv_b| = {np.nanmax(np.abs(diff)):.3e}"
      if np.isfinite(diff).any() else "   (no overlap)")
print(f"   finite cells: tv_a {np.isfinite(tv_a).sum()}, tv_b {np.isfinite(tv_b).sum()}")
