"""
Regenerate Table 4.5 at the corrected parameters of Table 4.4.

Also recomputes the per-maturity decomposition of the preference for the smooth
kernel, which the old caption expressed as a gap to H = 0.49. The fit now sits
at H = 1/2, so the gap is taken against H = 0.107 -- the value typical of the
rough volatility literature -- at that row's own profiled parameters.
"""
import numpy as np, pandas as pd
import rough_heston as rh
from load_options import load, quotes

FIT  = (0.023678882534143478, 0.06247408905600814, 2.11842446557372,
        0.9220047555857658, -0.7347398003277532, 0.5)
ROUGH = (0.0231, 0.1600, 0.267, 0.392, -0.721, 0.107)   # Table 4.6, H = 0.107

if __name__ == "__main__":
    q = quotes(load("../data"), cone=0.5)
    q = q[q.in_cone & (q.half_spread > 0)].reset_index(drop=True)
    mid = q["mid"].to_numpy(); h = q["half_spread"].to_numpy()
    e_f = rh.surface_iv(q, FIT,   N=400) - mid
    e_r = rh.surface_iv(q, ROUGH, N=400) - mid
    P_f = (e_f / h) ** 2
    P_r = (e_r / h) ** 2
    gap = P_r.sum() - P_f.sum()
    print(f"Phi(H=0.5) = {P_f.sum():.0f}   Phi(H=0.107) = {P_r.sum():.0f}   "
          f"gap = {gap:.0f}\n")
    print("$T$ & quotes & med.\ $h$ (bp) & RMSE (bp) & bias (bp) & "
          "\% of $\Phi$ & \% of gap\\\\")
    rows = []
    for T in np.unique(q["T"]):
        m = q["T"].to_numpy() == T
        rows.append(dict(T=float(T), n=int(m.sum()),
                         medh=float(np.median(h[m])) * 1e4,
                         rmse=float(np.sqrt(np.mean(e_f[m] ** 2))) * 1e4,
                         bias=float(np.mean(e_f[m])) * 1e4,
                         shPhi=100 * P_f[m].sum() / P_f.sum(),
                         shGap=100 * (P_r[m].sum() - P_f[m].sum()) / gap))
        r = rows[-1]
        pad = "\;\;" if r["n"] < 100 else ""
        print(f"${r['T']:.4f}$ & ${pad}{r['n']}$ & ${r['medh']:.1f}$ & "
              f"${r['rmse']:.1f}$ & ${r['bias']:+.1f}$ & ${r['shPhi']:.1f}$ & "
              f"${r['shGap']:+.1f}$\\\\")
    pd.DataFrame(rows).to_csv("permat_table45.csv", index=False)
