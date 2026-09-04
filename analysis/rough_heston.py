"""
Rough Heston: characteristic function, Lewis pricing, implied volatility.

Model (El Euch--Rosenbaum), with alpha = H + 1/2:

    dS_t / S_t = sqrt(V_t) dW_t
    V_t = V_0 + 1/Gamma(a) int_0^t (t-s)^{a-1} lam (th - V_s) ds
              + xi/Gamma(a) int_0^t (t-s)^{a-1} sqrt(V_s) dB_s ,   d<W,B> = rho dt

The characteristic function is  phi(z,T) = exp( lam*th*I^1 h + V_0*I^{1-a} h ),
where h solves the fractional Riccati  D^a h = F(z,h),  h(0)=0, with

    F(z,x) = -(z^2 + i z)/2 + (i z rho xi - lam) x + (xi^2 / 2) x^2 .

Two implementation points that matter:

 *  h = I^a F(h) since h(0)=0, hence  I^{1-a} h = I^1 F(h).  The singular
    kernel therefore never has to be quadratured -- both terms of log phi are
    ordinary integrals of quantities the solver already produces.

 *  Pricing is done on the Lewis contour Im z = -1/2, where z^2 + i z reduces
    to the real quantity u^2 + 1/4.

The fractional Adams predictor-corrector is O(N^2); it is vectorised over the
whole Fourier grid, so one solve prices every strike of a maturity at once.
"""

import numpy as np
from scipy.special import gamma as Gamma
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# quadrature grid for the Lewis integral
# ----------------------------------------------------------------------

def lewis_nodes(n=192, umax=200.0):
    """Gauss-Legendre nodes/weights on [0, umax]."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * umax * (x + 1.0)
    return u, 0.5 * umax * w


def nodes_for(T, N=300, xi=0.4, H=0.1, n=192, c=120.0, cap=200.0, safety=0.45):
    """Contour cut for maturity T, respecting the scheme's stability limit.

    Two constraints bind.

    (a) Empirically the boundary sits near u_max*sqrt(T) ~ 150-200, while the
        characteristic function has already decayed past 1e-11 inside it; the
        cut c/sqrt(T) with c = 120 is safely inside and costs < 1e-6 of a vega.

    (b) The initial layer.  Since h(0) = 0 the first step is
        h_1 ~ const * dt^alpha / Gamma(1+alpha) with const = -(u^2+1/4)/2,
        whereas the true solution settles at the root scale
        sqrt(|const|/quad), quad = xi^2/2.  When the first step overshoots
        that scale the quadratic term takes over and the iteration diverges.
        Requiring no overshoot gives

            u_max  <=  2 Gamma(1+alpha) / (xi * dt^alpha),   dt = T/N,

        which is what actually bites for large vol-of-vol.
    """
    alpha = H + 0.5
    dt = T / N
    u_stab = safety * 2.0 * Gamma(1.0 + alpha) / (xi * dt**alpha)
    return lewis_nodes(n, min(cap, c / np.sqrt(T), u_stab))


# ----------------------------------------------------------------------
# fractional Adams predictor-corrector
# ----------------------------------------------------------------------

def _adams_coeffs(alpha, N, dt):
    """Predictor weights b[m] and corrector weights w[m], m = 0..N."""
    m = np.arange(N + 1, dtype=float)
    b = (dt**alpha / alpha) * ((m + 1) ** alpha - m**alpha)
    c = dt**alpha / (alpha * (alpha + 1))
    w = c * ((m + 2) ** (alpha + 1) + m ** (alpha + 1)
             - 2 * (m + 1) ** (alpha + 1))
    return b, w, c


def riccati(u, T, N, H, lam, xi, rho):
    """Solve the fractional Riccati on [0,T] for the Lewis contour z = u - i/2.

    Returns (int_h, int_F): the two ordinary integrals entering log phi,
    each an array over the Fourier grid u.
    """
    alpha = H + 0.5
    dt = T / N
    b, w, c = _adams_coeffs(alpha, N, dt)
    Ga = Gamma(alpha)

    # On Im z = -1/2:  z^2 + i z = u^2 + 1/4,  i z = i u + 1/2.
    const = -0.5 * (u**2 + 0.25)                 # real, negative
    lin = (1j * u + 0.5) * rho * xi - lam        # complex
    quad = 0.5 * xi**2

    def F(x):
        return const + lin * x + quad * x * x

    nu = u.size
    h = np.zeros((N + 1, nu), dtype=complex)
    Fv = np.zeros((N + 1, nu), dtype=complex)
    Fv[0] = const                                 # F(0)

    for n in range(N):
        # predictor
        hp = (b[n::-1] @ Fv[: n + 1]) / Ga
        # corrector
        a0 = c * (n ** (alpha + 1) - (n - alpha) * (n + 1) ** alpha)
        s = a0 * Fv[0]
        if n >= 1:
            s = s + (w[n - 1 :: -1] @ Fv[1 : n + 1])
        h[n + 1] = (s + c * F(hp)) / Ga
        Fv[n + 1] = F(h[n + 1])

    return np.trapezoid(h, dx=dt, axis=0), np.trapezoid(Fv, dx=dt, axis=0)


def char_fn(u, T, N, v0, theta, lam, xi, rho, H):
    """log phi(u - i/2, T) on the Fourier grid."""
    ih, iF = riccati(u, T, N, H, lam, xi, rho)
    return lam * theta * ih + v0 * iF


# ----------------------------------------------------------------------
# prices and implied volatilities
# ----------------------------------------------------------------------

def call_prices(k, T, params, N=200, nodes=None, _tries=3):
    """Undiscounted call price / forward, for log-moneyness array k.

    If the Riccati diverges (large xi can push the boundary in), the contour
    is pulled back and the solve retried; the discarded band contributes far
    less than a tick, so this trades no accuracy for robustness.
    """
    v0, theta, lam, xi, rho, H = params
    u, wq = nodes_for(T, N, xi, H) if nodes is None else nodes

    for _ in range(_tries):
        with np.errstate(over="ignore", invalid="ignore"):
            logphi = char_fn(u, T, N, v0, theta, lam, xi, rho, H)
        if np.all(np.isfinite(logphi)):
            break
        u, wq = lewis_nodes(u.size, 0.5 * u[-1])
    else:
        return np.full(k.shape, np.nan)

    phi = np.exp(logphi)
    # C/F = 1 - e^{k/2}/pi * int Re[e^{-iuk} phi] / (u^2+1/4) du
    integ = np.real(np.exp(-1j * np.outer(k, u)) * phi[None, :]) / (u**2 + 0.25)
    return 1.0 - np.exp(0.5 * k) / np.pi * (integ @ wq)


def bs_call(k, T, sigma):
    """Undiscounted Black call / forward."""
    s = sigma * np.sqrt(T)
    d1 = (-k + 0.5 * s**2) / s
    return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - s)


def implied_vol(price, k, T, lo=1e-4, hi=5.0):
    """Invert one Black price; nan if outside the no-arbitrage bounds."""
    intrinsic = max(1.0 - np.exp(k), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-14 or price >= 1.0:
        return np.nan
    try:
        return brentq(lambda s: bs_call(k, T, s) - price, lo, hi, xtol=1e-10)
    except ValueError:
        return np.nan


def _bs(k, T, sig):
    s = sig * np.sqrt(T)
    d1 = (-k + 0.5 * s * s) / s
    return ndtr(d1) - np.exp(k) * ndtr(d1 - s), d1


def implied_vol_vec(price, k, T, sig0, iters=14):
    """Safeguarded Newton on the whole surface at once.

    Started from the market volatility, so 3-4 Newton steps suffice; the
    running bracket catches the deep wings where vega collapses and the
    step would otherwise overshoot, falling back to bisection there.
    """
    k = np.asarray(k, float)
    T = np.asarray(T, float)
    price = np.asarray(price, float)

    sig = np.clip(np.asarray(sig0, float), 1e-3, 3.0).copy()
    lo = np.full(sig.shape, 1e-5)
    hi = np.full(sig.shape, 5.0)
    intrinsic = np.maximum(1.0 - np.exp(k), 0.0)
    bad = (~np.isfinite(price)) | (price <= intrinsic + 1e-14) | (price >= 1.0)

    for _ in range(iters):
        c, d1 = _bs(k, T, sig)
        f = c - price
        hi = np.where(f > 0, np.minimum(hi, sig), hi)
        lo = np.where(f < 0, np.maximum(lo, sig), lo)
        vega = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt = sig - f / vega
        take = np.isfinite(nxt) & (nxt > lo) & (nxt < hi)
        sig = np.where(take, nxt, 0.5 * (lo + hi))

    sig[bad] = np.nan
    return sig


def surface_iv(quotes, params, N=300):
    """Model implied vols for a quote table with columns T, k.

    One Riccati solve per maturity prices that whole slice, with the contour
    chosen per maturity by nodes_for.
    """
    xi, H = params[3], params[5]
    Tv = quotes["T"].to_numpy()
    kv = quotes["k"].to_numpy()
    sig0 = quotes["mid"].to_numpy()

    px = np.full(len(quotes), np.nan)
    for T in np.unique(Tv):
        m = Tv == T
        px[m] = call_prices(kv[m], float(T), params, N=N,
                            nodes=nodes_for(float(T), N, xi, H))
    return implied_vol_vec(px, kv, Tv, sig0)
