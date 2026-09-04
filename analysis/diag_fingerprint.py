"""Per-column fingerprints.  Two columns describing the SAME set of options
must contain the same number of finite cells, in the same rows.
"""
import numpy as np
from load_options import load, NMAT

d = load("../data")
r = min(d["tv_a"].shape[0], d["mid"].shape[0], d["moneyness"].shape[0])
S = d["moneyness"][:r]
M, B, A = d["mid"][:r], d["bid"][:r], d["ask"][:r]
TA, TB = d["tv_a"][:r], d["tv_b"][:r]

print("finite-cell count per column")
print(" col   strikes   vol   tv_a   tv_b")
for j in range(NMAT):
    print(f" {j+1:3d}   {np.isfinite(S[:,j]).sum():7d} {np.isfinite(M[:,j]).sum():5d}"
          f" {np.isfinite(TA[:,j]).sum():6d} {np.isfinite(TB[:,j]).sum():6d}")

print("\nrow-support agreement (fraction of rows where 'finite' flags agree)")


def agree(X, Y, jx, jy):
    return float((np.isfinite(X[:, jx]) == np.isfinite(Y[:, jy])).mean())


for lab, off in (("aligned  tv[j] vs vol[j]", 0), ("shifted  tv[j] vs vol[j+1]", 1)):
    v = [agree(TA, M, j, j + off) for j in range(NMAT - 1)]
    s = [agree(TA, S, j, j + off) for j in range(NMAT - 1)]
    print(f"  {lab}:   vs vol {np.mean(v):.4f}   vs strikes {np.mean(s):.4f}")

print("\nEXACT identities under the shift  tv[j] <-> vol[j+1]")
for name, TV, V in (("tv_a vs bid", TA, B), ("tv_a vs mid", TA, M),
                    ("tv_a vs ask", TA, A), ("tv_b vs bid", TB, B),
                    ("tv_b vs mid", TB, M), ("tv_b vs ask", TB, A)):
    worst = 0.0
    for j in range(NMAT - 1):
        t = TV[:, j] / V[:, j + 1] ** 2
        t = t[np.isfinite(t)]
        if t.size < 3:
            continue
        worst = max(worst, float(t.std() / t.mean()))
    print(f"  {name}:  max CV over columns = {worst:.3e}")

print("\nATM diagnostics per VOL column (moneyness nearest 1), using the")
print("strike grid in the SAME sheet position as the vol.")
print(" col   atm_mny   atm_vol   skew(90-110)   nquotes   mny range")
for j in range(NMAT):
    ok = np.isfinite(M[:, j]) & np.isfinite(S[:, j])
    if not ok.any():
        continue
    idx = np.where(ok)[0]
    i = idx[np.argmin(np.abs(S[idx, j] - 1.0))]
    # skew: vol at moneyness ~0.9 minus vol at ~1.1
    i90 = idx[np.argmin(np.abs(S[idx, j] - 0.90))]
    i110 = idx[np.argmin(np.abs(S[idx, j] - 1.10))]
    skew = M[i90, j] - M[i110, j]
    print(f" {j+1:3d}   {S[i,j]:7.4f}   {M[i,j]:7.4f}   {skew:12.4f}"
          f"   {ok.sum():7d}   [{np.nanmin(S[:,j]):.3f},{np.nanmax(S[:,j]):.3f}]")
