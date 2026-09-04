"""Recover T for every vol column, including the wrap pair (tv col 17 -> vol col 1).

Established: TotalVar block A = bid^2 * T, block B = ask^2 * T, with the
TotalVar sheet cyclically rotated one column left relative to Strikes/Times.
So  T(vol column m) = tv[:, (m-2) mod 17] / bid[:, m-1]^2.
"""
import numpy as np
from load_options import load, NMAT

d = load("../data")
r = min(d["tv_a"].shape[0], d["mid"].shape[0], d["moneyness"].shape[0])
S, Mid, Bid, Ask = (d[k][:r] for k in ("moneyness", "mid", "bid", "ask"))
TA, TB = d["tv_a"][:r], d["tv_b"][:r]


def implied_T(w, vol):
    t = w / vol**2
    t = t[np.isfinite(t)]
    if t.size < 3:
        return np.nan, np.nan, t.size
    return float(np.median(t)), float(t.std() / t.mean()), t.size


print("T per VOL column, from the cyclic pairing  tv[(m-2) mod 17] <-> vol[m-1]")
print(" volcol  tvcol      T(bid)        CV        n      T(ask)        CV")
Ts = []
for m in range(1, NMAT + 1):
    jt = (m - 2) % NMAT
    Ta, cva, na = implied_T(TA[:, jt], Bid[:, m - 1])
    Tb, cvb, nb = implied_T(TB[:, jt], Ask[:, m - 1])
    Ts.append(Ta)
    print(f" {m:5d}  {jt+1:5d}  {Ta:10.5f}  {cva:.3e}  {na:5d}  "
          f"{Tb:10.5f}  {cvb:.3e}")

Ts = np.array(Ts)
print(f"\n monotone increasing across columns? {bool(np.all(np.diff(Ts) > 0))}")

print("\n--- calendar check: ATM total variance must be increasing in T ---")
print(" col       T    atm_vol   atm_totvar   skew(90-110)")
prev = -np.inf
ok_cal = True
for m in range(1, NMAT + 1):
    j = m - 1
    ok = np.isfinite(Mid[:, j]) & np.isfinite(S[:, j])
    idx = np.where(ok)[0]
    i = idx[np.argmin(np.abs(S[idx, j] - 1.0))]
    i90 = idx[np.argmin(np.abs(S[idx, j] - 0.90))]
    i110 = idx[np.argmin(np.abs(S[idx, j] - 1.10))]
    w = Mid[i, j] ** 2 * Ts[j]
    flag = "" if w > prev else "   <-- CALENDAR VIOLATION"
    if w <= prev:
        ok_cal = False
    prev = w
    print(f" {m:3d}  {Ts[j]:7.4f}   {Mid[i,j]:7.4f}   {w:10.6f}   "
          f"{Mid[i90,j]-Mid[i110,j]:8.4f}{flag}")
print(f"\n calendar-arbitrage free? {ok_cal}")

np.savetxt("maturities.txt", Ts, fmt="%.6f")
print("\nwrote maturities.txt")
