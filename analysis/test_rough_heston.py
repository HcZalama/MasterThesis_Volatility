"""Validation of the rough Heston pricer.

Three tests, in increasing strength:

 1. Degenerate case xi = 0 with v0 = theta.  Then V_t == v0 deterministically
    for EVERY H, so the model is Black-Scholes and the implied volatility must
    equal sqrt(v0) at every strike and maturity.  This exercises the Riccati
    solve, the Lewis integral and the inversion end to end.

 2. H = 1/2 (alpha = 1).  The fractional Adams scheme must then agree with a
    high-accuracy ODE integration of the same Riccati.

 3. Convergence of the price in the number of time steps N.
"""

import numpy as np
from scipy.integrate import solve_ivp
import rough_heston as rh

nodes = rh.lewis_nodes()
k = np.array([-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30])

print("=" * 70)
print("TEST 1  xi = 0, v0 = theta  ->  implied vol must be exactly sqrt(v0)")
v0 = 0.04
for H in (0.05, 0.12, 0.3, 0.5):
    worst = 0.0
    for T in (0.10404, 0.60233, 2.84463):
        p = (v0, v0, 1.5, 0.0, -0.7, H)
        c = rh.call_prices(k, T, p, N=200, nodes=nodes)
        iv = np.array([rh.implied_vol(ci, ki, T) for ci, ki in zip(c, k)])
        worst = max(worst, np.nanmax(np.abs(iv - np.sqrt(v0))))
    print(f"   H = {H:4.2f}   max |iv - {np.sqrt(v0):.4f}| = {worst:.3e}")

print("\n" + "=" * 70)
print("TEST 2  H = 1/2: fractional Adams vs solve_ivp on the same Riccati")
lam, xi, rho, theta = 1.5, 0.4, -0.7, 0.04
T = 1.0
u = nodes[0][::24]                      # a subset of the Fourier grid
const = -0.5 * (u**2 + 0.25)
lin = (1j * u + 0.5) * rho * xi - lam
quad = 0.5 * xi**2


def rhs(t, y):
    x = y[: u.size] + 1j * y[u.size:]
    d = const + lin * x + quad * x * x
    return np.concatenate([d.real, d.imag])


sol = solve_ivp(rhs, (0, T), np.zeros(2 * u.size), rtol=1e-11, atol=1e-13,
                dense_output=True)
ref = sol.y[: u.size, -1] + 1j * sol.y[u.size:, -1]

for N in (100, 200, 400, 800):
    dt = T / N
    ih, iF = rh.riccati(u, T, N, 0.5, lam, xi, rho)
    # recover h(T) from the scheme for comparison
    alpha = 1.0
    hN = rh.riccati.__wrapped__ if False else None
    # re-solve keeping the terminal value
    b, w, c = rh._adams_coeffs(alpha, N, dt)
    from scipy.special import gamma as G
    Ga = G(alpha)
    h = np.zeros((N + 1, u.size), complex)
    Fv = np.zeros((N + 1, u.size), complex)
    Fv[0] = const
    F = lambda x: const + lin * x + quad * x * x
    for n in range(N):
        hp = (b[n::-1] @ Fv[: n + 1]) / Ga
        a0 = c * (n ** 2 - (n - 1) * (n + 1))
        s = a0 * Fv[0]
        if n >= 1:
            s = s + (w[n - 1 :: -1] @ Fv[1 : n + 1])
        h[n + 1] = (s + c * F(hp)) / Ga
        Fv[n + 1] = F(h[n + 1])
    print(f"   N = {N:4d}   max |h_N(T) - h_ref(T)| = "
          f"{np.max(np.abs(h[-1] - ref)):.3e}")

print("\n" + "=" * 70)
print("TEST 3  price convergence in N  (rough case H = 0.10)")
p = (0.025, 0.045, 1.2, 0.35, -0.75, 0.10)
for T in (0.10404, 1.02396, 2.84463):
    ref = rh.call_prices(k, T, p, N=1600, nodes=nodes)
    line = f"   T = {T:7.4f}: "
    for N in (100, 200, 400, 800):
        c = rh.call_prices(k, T, p, N=N, nodes=nodes)
        line += f" N={N}: {np.max(np.abs(c - ref)):.2e}"
    print(line)

print("\n" + "=" * 70)
print("TEST 4  Lewis truncation: umax / node count")
for T in (0.10404, 2.84463):
    ref = rh.call_prices(k, T, p, N=400, nodes=rh.lewis_nodes(512, 600.0))
    for nq, um in ((128, 150.0), (192, 200.0), (256, 300.0)):
        c = rh.call_prices(k, T, p, N=400, nodes=rh.lewis_nodes(nq, um))
        print(f"   T={T:7.4f}  n={nq:4d} umax={um:5.0f}  "
              f"max|dC| = {np.max(np.abs(c - ref)):.2e}")
