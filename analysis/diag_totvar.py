"""Why does w / sigma^2 fail to be constant?  Test alignment and pairing."""
import numpy as np
from load_options import load, NMAT

d = load("../data")

print("finite entries per column (rows where the cell is a number)")
print(" col  strike   mid   tv_a   tv_b")
for j in range(NMAT):
    print(f" {j+1:3d}  {np.isfinite(d['moneyness'][:, j]).sum():6d}"
          f" {np.isfinite(d['mid'][:, j]).sum():5d}"
          f" {np.isfinite(d['tv_a'][:, j]).sum():6d}"
          f" {np.isfinite(d['tv_b'][:, j]).sum():6d}")

r = min(d["tv_a"].shape[0], d["mid"].shape[0])


def const_score(num, den):
    """coefficient of variation of num/den, per column; small = constant."""
    out = []
    for j in range(NMAT):
        t = num[:r, j] / den[:r, j] ** 2
        t = t[np.isfinite(t)]
        out.append(np.nan if t.size < 3 else t.std() / t.mean())
    return np.array(out)


print("\ncoefficient of variation of w/sigma^2  (0 = perfect)")
combos = {
    "tv_a / mid^2": (d["tv_a"], d["mid"]),
    "tv_a / bid^2": (d["tv_a"], d["bid"]),
    "tv_a / ask^2": (d["tv_a"], d["ask"]),
    "tv_b / mid^2": (d["tv_b"], d["mid"]),
    "tv_b / bid^2": (d["tv_b"], d["bid"]),
    "tv_b / ask^2": (d["tv_b"], d["ask"]),
}
for name, (n, dd) in combos.items():
    cv = const_score(n, dd)
    print(f"  {name}:  median CV = {np.nanmedian(cv):.4f}")

# does a row shift fix it?
print("\nbest row offset for tv_a / mid^2 (searching -5..+5)")
for j in [0, 5, 10, 16]:
    best = None
    for off in range(-5, 6):
        a = d["tv_a"][:, j]
        b = d["mid"][:, j]
        if off >= 0:
            x, y = a[off:], b[: len(b) - off] if off else b
        else:
            x, y = a[: len(a) + off], b[-off:]
        n = min(len(x), len(y))
        t = x[:n] / y[:n] ** 2
        t = t[np.isfinite(t)]
        if t.size < 3:
            continue
        cv = t.std() / t.mean()
        if best is None or cv < best[1]:
            best = (off, cv, t.mean())
    print(f"  col {j+1:2d}: offset {best[0]:+d}, CV {best[1]:.4f}, T~{best[2]:.4f}")

# Is tv actually sigma^2 * T with T read off the ATM quote only?
print("\nT implied by the single quote closest to the money, per column")
Ts = []
for j in range(NMAT):
    k = np.abs(np.log(d["moneyness"][:r, j]))
    ok = np.isfinite(k) & np.isfinite(d["tv_a"][:r, j]) & np.isfinite(d["mid"][:r, j])
    if not ok.any():
        Ts.append(np.nan)
        continue
    i = np.nanargmin(np.where(ok, k, np.inf))
    T = d["tv_a"][i, j] / d["mid"][i, j] ** 2
    Ts.append(T)
    print(f"  col {j+1:2d}: atm moneyness {d['moneyness'][i, j]:.4f}  T = {T:.4f}")
Ts = np.array(Ts)
print("\n  monotone increasing?", bool(np.all(np.diff(Ts) > 0)))
