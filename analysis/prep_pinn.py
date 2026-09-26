"""
Verify the PINN bid/ask export against Table 4.2, then build the rough Heston
sample from it.

Table 4.2 records 576 quotes on 22 smiles, T = 0.1971 .. 9.28, the shortest
smile carrying 30 quotes and every other one 26. This checks all of that
against the file before anything is fitted.

The Heston sample is then the sub-book the user asked for: T <= 1 year and an
at-the-money cone |k| <= 0.5 sqrt(T + 0.1). Both filters apply to the Heston
fit alone; the PINN used the whole book.
"""

import numpy as np
import pandas as pd

TMAX = 1.0


def cone_halfwidth(T):
    return 0.5 * np.sqrt(T + 0.1)


if __name__ == "__main__":
    q = pd.read_csv("pinn_quotes.csv")
    q["mid"] = 0.5 * (q.bid + q.ask)
    q["half_spread"] = 0.5 * (q.ask - q.bid)

    Ts = np.unique(q["T"])
    per = q.groupby("T").size()

    print("=== against Table 4.2 ===")
    print(f"  total quotes        {len(q):5d}   table: 576   "
          f"{'OK' if len(q)==576 else 'MISMATCH'}")
    print(f"  smiles              {len(Ts):5d}   table:  22   "
          f"{'OK' if len(Ts)==22 else 'MISMATCH'}")
    print(f"  shortest T          {Ts[0]:.4f}   table: 0.1971")
    print(f"  longest  T          {Ts[-1]:.4f}   table: 9.28")
    print(f"  quotes at T={Ts[0]:.4f}  {per.iloc[0]:5d}   table:  30   "
          f"{'OK' if per.iloc[0]==30 else 'MISMATCH'}")
    rest = per.iloc[1:]
    print(f"  quotes elsewhere    {sorted(set(rest)) }   table:  26 ea.   "
          f"{'OK' if set(rest)=={26} else 'MISMATCH'}")
    print(f"  ask > bid everywhere      {'OK' if (q.ask > q.bid).all() else 'NO'}")
    print(f"  duplicated (T,k) rows     {q.duplicated(['T','k']).sum()}"
          "   (one per smile: the PINN book counts both)")

    print("\n  maturities present:")
    for i in range(0, len(Ts), 6):
        print("    " + "  ".join(f"{t:8.4f}" for t in Ts[i:i + 6]))

    print("\n  the two smiles review 1 flagged:")
    for T in (1.27036, 1.36619):
        s = q[q["T"] == T]
        print(f"    T = {T:.5f}   {len(s)} quotes   "
              f"median half-spread {s.half_spread.median()*1e4:5.1f} bp")

    # ---- the Heston sub-book -------------------------------------------
    s = q[q["T"] <= TMAX].copy()
    s["cone"] = cone_halfwidth(s["T"])
    s["in_cone"] = s["k"].abs() <= s["cone"]
    kept = s[s.in_cone].reset_index(drop=True)

    print(f"\n=== rough Heston sample: T <= {TMAX}, |k| <= 0.5 sqrt(T+0.1) ===")
    print(f"  T <= {TMAX}                 {len(s):5d} quotes on "
          f"{s['T'].nunique()} smiles")
    print(f"  inside the cone           {len(kept):5d} quotes on "
          f"{kept['T'].nunique()} smiles")
    print(f"  T range                   [{kept['T'].min():.4f}, "
          f"{kept['T'].max():.4f}]")
    print(f"  |k| range                 [{kept['k'].abs().min():.4f}, "
          f"{kept['k'].abs().max():.4f}]")
    print(f"  median half-spread        {kept.half_spread.median()*1e4:.1f} bp")
    print(f"  mid vol range             [{kept['mid'].min():.4f}, "
          f"{kept['mid'].max():.4f}]")
    print("\n  per smile:")
    print("     T        all  cone  halfwidth   median h (bp)")
    for T in np.unique(s["T"]):
        a = s[s["T"] == T]
        c = a[a.in_cone]
        print(f"   {T:.5f}  {len(a):4d}  {len(c):4d}   {cone_halfwidth(T):.4f}"
              f"      {c.half_spread.median()*1e4:6.1f}")

    kept[["T", "k", "bid", "ask", "mid", "half_spread"]].to_csv(
        "pinn_heston_sample.csv", index=False)
    print(f"\nwrote pinn_heston_sample.csv  ({len(kept)} quotes)")
