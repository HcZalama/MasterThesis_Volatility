import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
OUT = r"c:/Users/hecto/source/repos/HcZalama/MasterThesis_Volatility/Images/"
K, G = "#1a1a1a", "#8a8a8a"

def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---------------- Figure 1: path + the two variation sums ----------------
rng = np.random.default_rng(20260429)
N, T = 2**16, 1.0
dW = rng.normal(0.0, np.sqrt(T/N), N)
W = np.concatenate([[0.0], np.cumsum(dW)])
t = np.linspace(0, T, N+1)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5))
a1.plot(t, W, lw=0.55, color=K)
a1.axhline(0, lw=0.5, color=G, zorder=0)
a1.set_xlabel("$t$"); a1.set_ylabel("$W_t$"); a1.set_xlim(0, T)
a1.set_title("A sample path on $[0,1]$", fontsize=9, pad=6)
tidy(a1)

ks = np.arange(1, 17)                       # 2^k subintervals
n_sub, tv, qv = 2**ks, [], []
for n in n_sub:
    inc = np.diff(W[::N//n])
    tv.append(np.abs(inc).sum()); qv.append((inc**2).sum())
a2.loglog(n_sub, tv, "o-", ms=3, lw=0.9, color=K,
          label=r"$\sum_k |W_{t_k}-W_{t_{k-1}}|$")
a2.loglog(n_sub, qv, "s--", ms=3, lw=0.9, color=G,
          label=r"$\sum_k (W_{t_k}-W_{t_{k-1}})^2$")
a2.axhline(T, lw=0.7, ls=":", color=K)
a2.text(1.6e4, 1.55, "$t=1$", fontsize=8)
a2.set_xlabel(r"number of subintervals $n$   (mesh $\|\mathcal{P}\|=1/n$)")
a2.set_title("Partition sums as the mesh is refined", fontsize=9, pad=6)
a2.legend(frameon=False, loc="upper left", handlelength=2.4)
a2.set_ylim(0.28, max(tv)*3.2)
a2.set_yticks([1, 10, 100]); a2.set_yticklabels(["$1$", "$10$", "$100$"])
tidy(a2)
fig.tight_layout(pad=0.4)
fig.savefig(OUT+"bm-variation.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: reflection principle ----------------
rng = np.random.default_rng(7)
n, T = 4000, 1.0
while True:
    w = np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(T/n), n))])
    a = 1.0
    hit = np.argmax(w >= a)
    if hit > 0 and 0.15*n < hit < 0.6*n and w[-1] < a - 0.25:
        break
tt = np.linspace(0, T, n+1)
refl = w.copy(); refl[hit:] = 2*a - w[hit:]

fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.plot(tt, w, lw=0.7, color=K, label=r"$W$")
ax.plot(tt[hit:], refl[hit:], lw=0.7, ls="--", color=G,
        label=r"reflected path $\widetilde{W}$")
ax.axhline(a, lw=0.7, ls=":", color=K)
ax.plot([tt[hit]], [a], "o", ms=3.5, color=K)
ax.annotate(r"$\tau_a$", (tt[hit], a), textcoords="offset points",
            xytext=(-14, -13), fontsize=9)
ax.text(0.015, a + 0.03, "$a$", va="bottom", ha="left", fontsize=9)
ax.plot([T], [w[-1]], "o", ms=3, color=K)
ax.plot([T], [refl[-1]], "o", ms=3, color=G)
ax.annotate(r"$W_T$", (T, w[-1]), textcoords="offset points", xytext=(5, -3), fontsize=8.5)
ax.annotate(r"$2a-W_T$", (T, refl[-1]), textcoords="offset points", xytext=(5, -3), fontsize=8.5)
ax.axhline(0, lw=0.5, color=G, zorder=0)
ax.set_xlabel("$t$"); ax.set_ylabel("$W_t$"); ax.set_xlim(0, T*1.16)
ax.legend(frameon=False, loc="lower right", handlelength=2.4)
tidy(ax)
fig.tight_layout(pad=0.4)
fig.savefig(OUT+"reflection.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 3: forward vs backward ----------------
S0, r, sig, Tm, Kst = 100.0, 0.0, 0.25, 1.0, 100.0
x = np.linspace(30, 210, 700)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.4, 2.6))

for tau, ls, sh in [(0.06, ":", G), (0.30, "--", G), (1.00, "-", K)]:
    m = (np.log(x/S0) - (r-0.5*sig**2)*tau)/(sig*np.sqrt(tau))
    dens = np.exp(-0.5*m**2)/(x*sig*np.sqrt(2*np.pi*tau))
    b1.plot(x, dens, lw=0.95, ls=ls, color=sh, label=fr"$t={tau:g}$")
b1.set_xlabel("$y$"); b1.set_ylabel(r"$p_{0,\,S_0}(t,y)$")
b1.set_title(r"Forward: the density spreads out", fontsize=9, pad=6)
b1.legend(frameon=False); b1.set_xlim(30, 210); tidy(b1)
b1.annotate("", xy=(178, b1.get_ylim()[1]*0.72), xytext=(122, b1.get_ylim()[1]*0.72),
            arrowprops=dict(arrowstyle="->", lw=0.7, color=K))

def call(S, tau):
    if tau <= 1e-12: return np.maximum(S-Kst, 0.0)
    d1 = (np.log(S/Kst)+(r+0.5*sig**2)*tau)/(sig*np.sqrt(tau)); d2 = d1-sig*np.sqrt(tau)
    from math import erf
    Nv = np.vectorize(lambda z: 0.5*(1+erf(z/np.sqrt(2))))
    return S*Nv(d1) - Kst*np.exp(-r*tau)*Nv(d2)

for tau, ls, sh, lab in [(0.0, "-", K, r"$s=T$  (payoff)"),
                         (0.30, "--", G, r"$s=T-0.3$"),
                         (1.00, ":", G, r"$s=T-1$")]:
    b2.plot(x, call(x, tau), lw=0.95, ls=ls, color=sh, label=lab)
b2.set_xlabel("$x$"); b2.set_ylabel(r"$u(s,x)$")
b2.set_title(r"Backward: the payoff is smoothed", fontsize=9, pad=6)
b2.legend(frameon=False, loc="upper left", handlelength=2.4); b2.set_xlim(62, 148)
b2.set_ylim(-2.5, 52); tidy(b2)
b2.annotate("", xy=(103, 5.0), xytext=(103, 20.0),
            arrowprops=dict(arrowstyle="->", lw=0.7, color=K))
b2.plot([100, 100], [-2.5, 0], lw=0.6, ls=":", color=G)
b2.text(97.5, -2.1, "$K$", fontsize=8.5, ha="right", va="bottom")
for ax in (b1, b2): ax.yaxis.set_major_locator(MaxNLocator(4))
fig.tight_layout(pad=0.4)
fig.savefig(OUT+"forward-backward.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote 3 figures")
