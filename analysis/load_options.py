"""
Loader and diagnostics for the SPX option surface (Strikes / Times / TotalVar).

Geometry, inferred from the files and verified on row 1:
    17 columns  = 17 maturities, ordered short -> long
    rows        = the strike ladder for that maturity
    #N/A, blank = padding where a maturity has fewer strikes

    Strikes_CSV.csv   cols  1..17 == cols 18..34  (one moneyness grid, duplicated)
    Times.csv         block A header "VOL;...;SPREAD;..."  -> mid vol | (ask-bid) in %
                      block B header "BID;...;ASK;..."     -> bid vol | ask vol
    TotalVar_CSV.csv  cols  1..17 and 18..34

Checked exactly on (row 1, col 1): (bid+ask)/2 == VOL and ask-bid == SPREAD.

Usage:  python load_options.py [data_dir] [T1,T2,...,T17]
"""

import sys
import os
import numpy as np
import pandas as pd

NMAT = 17


# ----------------------------------------------------------------------
# raw reading
# ----------------------------------------------------------------------

def _to_float(tok):
    """Parse one cell: '' and '#N/A' -> nan; a trailing % -> /100."""
    tok = tok.strip()
    if tok == "" or tok.upper() == "#N/A":
        return np.nan
    if tok.endswith("%"):
        return float(tok[:-1]) / 100.0
    return float(tok)


def _read_matrix(path):
    """Read a ';'-separated sheet into a (rows x cols) float array, no header."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            rows.append([_to_float(t) for t in line.split(";")])
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def _read_labelled_blocks(path, labels):
    """Times.csv holds two stacked blocks, each introduced by a header line
    whose first field is a label (VOL / BID).  Return {label: array}."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = [l.rstrip("\n").rstrip("\r") for l in fh]

    starts = {}
    for i, line in enumerate(lines):
        head = line.split(";")[0].strip().upper()
        if head in labels:
            starts[head] = i

    blocks = {}
    for lab, i0 in starts.items():
        body = []
        for line in lines[i0 + 1:]:
            first = line.split(";")[0].strip().upper()
            if first in labels:            # next block begins
                break
            if line.strip(";").strip() == "":   # blank separator row
                continue
            body.append([_to_float(t) for t in line.split(";")])
        width = max(len(r) for r in body)
        arr = np.full((len(body), width), np.nan)
        for i, r in enumerate(body):
            arr[i, : len(r)] = r
        blocks[lab] = arr
    return blocks


# ----------------------------------------------------------------------

def load(data_dir):
    strikes = _read_matrix(os.path.join(data_dir, "Strikes_CSV.csv"))
    totvar = _read_matrix(os.path.join(data_dir, "TotalVar_CSV.csv"))
    blocks = _read_labelled_blocks(os.path.join(data_dir, "Times.csv"),
                                   {"VOL", "BID"})

    volspread = blocks["VOL"]        # cols 0..16 mid vol, 17..33 spread
    bidask = blocks["BID"]           # cols 0..16 bid vol, 17..33 ask vol

    d = dict(
        moneyness=strikes[:, :NMAT],
        moneyness_dup=strikes[:, NMAT:2 * NMAT],
        mid=volspread[:, :NMAT],
        spread=volspread[:, NMAT:2 * NMAT],
        bid=bidask[:, :NMAT],
        ask=bidask[:, NMAT:2 * NMAT],
        tv_a=totvar[:, :NMAT],
        tv_b=totvar[:, NMAT:2 * NMAT],
    )
    return d


def check(d):
    """Report the identities that must hold if the parse is right."""
    def mx(a):
        a = np.asarray(a)
        return np.nanmax(np.abs(a)) if np.isfinite(a).any() else np.nan

    n = min(d["moneyness"].shape[0], d["moneyness_dup"].shape[0])
    print("--- structural identities ---")
    print(f"  strikes block duplicated   max|diff| = "
          f"{mx(d['moneyness'][:n] - d['moneyness_dup'][:n]):.3e}")

    m = min(d["mid"].shape[0], d["bid"].shape[0])
    print(f"  mid == (bid+ask)/2         max|diff| = "
          f"{mx(d['mid'][:m] - 0.5 * (d['bid'][:m] + d['ask'][:m])):.3e}")
    print(f"  spread == ask - bid        max|diff| = "
          f"{mx(d['spread'][:m] - (d['ask'][:m] - d['bid'][:m])):.3e}")

    print("\n--- quotes per maturity ---")
    for j in range(NMAT):
        k = np.isfinite(d["mid"][:, j]).sum()
        mn = np.nanmin(d["moneyness"][:, j])
        mxn = np.nanmax(d["moneyness"][:, j])
        print(f"  col {j+1:2d}: {k:4d} quotes,  moneyness [{mn:.4f}, {mxn:.4f}]")

    # If TotalVar were sigma^2 T on this grid, w/sigma^2 would be constant
    # down each column.  It is not; this prints the evidence.
    print("\n--- implied T = w / mid^2 (should be constant per column) ---")
    print("  col      min      med      max    spread%")
    r = min(d["tv_a"].shape[0], d["mid"].shape[0])
    for j in range(NMAT):
        t = d["tv_a"][:r, j] / d["mid"][:r, j] ** 2
        t = t[np.isfinite(t)]
        if t.size == 0:
            continue
        rng = 100 * (t.max() - t.min()) / np.median(t)
        print(f"  {j+1:3d}  {t.min():7.4f}  {np.median(t):7.4f}  "
              f"{t.max():7.4f}   {rng:6.1f}%")


def tidy(d, maturities, cone=0.5):
    """Flatten to a quote table and apply the ATM cone |k| <= cone*sqrt(T).

    k = log(moneyness) = log(K/F).
    """
    T = np.asarray(maturities, dtype=float)
    assert T.size == NMAT, f"need {NMAT} maturities, got {T.size}"

    recs = []
    for j in range(NMAT):
        ok = np.isfinite(d["mid"][:, j]) & np.isfinite(d["moneyness"][:, j])
        for i in np.where(ok)[0]:
            recs.append(dict(
                mat=j,
                T=T[j],
                k=np.log(d["moneyness"][i, j]),
                bid=d["bid"][i, j],
                ask=d["ask"][i, j],
                mid=d["mid"][i, j],
            ))
    q = pd.DataFrame(recs)
    q["half_spread"] = 0.5 * (q["ask"] - q["bid"])
    q["in_cone"] = q["k"].abs() <= cone * np.sqrt(q["T"])

    print(f"\n--- cone |k| <= {cone}*sqrt(T) ---")
    print(f"  total quotes {len(q)},  kept {int(q.in_cone.sum())}")
    for j in range(NMAT):
        s = q[q.mat == j]
        print(f"  col {j+1:2d}: T={T[j]:7.4f}  kept {int(s.in_cone.sum()):4d} "
              f"/ {len(s):4d}")
    return q



# ----------------------------------------------------------------------
# Maturities.
#
# The TotalVar sheet is displaced one column left relative to Strikes/Times:
# block A of TotalVar is bid^2 * T and block B is ask^2 * T, both for the
# vol column ONE TO THE RIGHT.  The identity holds to CV ~1e-8, so T is not
# fitted but read off exactly.  Vol column 1 has quotes but no total
# variance anywhere in the file and is therefore dropped.
# ----------------------------------------------------------------------

FIRST_COL = 1          # 0-based index of the first usable vol column


def maturities(d):
    """Exact T for vol columns 2..17.  Returns (cols, T, worst_cv)."""
    r = min(d["tv_a"].shape[0], d["bid"].shape[0])
    cols, Ts, cvs = [], [], []
    for m in range(FIRST_COL, NMAT):          # vol column index m
        t = d["tv_a"][:r, m - 1] / d["bid"][:r, m] ** 2
        t = t[np.isfinite(t)]
        if t.size < 3:
            continue
        cols.append(m)
        Ts.append(float(np.median(t)))
        cvs.append(float(t.std() / t.mean()))
    return np.array(cols), np.array(Ts), max(cvs)


def quotes(d, cone=0.5):
    """Flat quote table on the usable columns, with the ATM cone flag.

    k = log(moneyness) = log(K/F);  cone keeps |k| <= cone*sqrt(T).
    """
    cols, T, worst = maturities(d)
    assert worst < 1e-5, f"maturity identity broke: worst CV {worst:.2e}"

    # the vol sheet is blank-padded well past the strike ladder
    r = min(d["moneyness"].shape[0], d["mid"].shape[0], d["bid"].shape[0])
    mny, mid = d["moneyness"][:r], d["mid"][:r]
    bid, ask = d["bid"][:r], d["ask"][:r]

    recs = []
    for c, Tc in zip(cols, T):
        ok = np.isfinite(mid[:, c]) & np.isfinite(mny[:, c]) & (mny[:, c] > 0)
        for i in np.where(ok)[0]:
            recs.append(dict(mat=int(c), T=Tc,
                             k=float(np.log(mny[i, c])),
                             bid=bid[i, c], ask=ask[i, c], mid=mid[i, c]))
    q = pd.DataFrame(recs)
    q["w"] = q["mid"] ** 2 * q["T"]
    q["half_spread"] = 0.5 * (q["ask"] - q["bid"])
    q["in_cone"] = q["k"].abs() <= cone * np.sqrt(q["T"])
    return q


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    d = load(data_dir)
    cols, T, worst = maturities(d)
    print(f"maturities recovered for {len(T)} columns, worst CV {worst:.2e}")
    for c, t in zip(cols, T):
        print(f"  vol col {c+1:2d}:  T = {t:.5f}")

    q = quotes(d)
    print(f"\ntotal quotes {len(q)},  in cone {int(q.in_cone.sum())}")
    for c, t in zip(cols, T):
        s = q[q.mat == c]
        print(f"  col {c+1:2d}  T={t:7.4f}  kept {int(s.in_cone.sum()):4d} / {len(s):4d}"
              f"   k in [{s.k.min():+.3f}, {s.k.max():+.3f}]")
    q.to_csv("quotes.csv", index=False)
    print("\nwrote quotes.csv")
