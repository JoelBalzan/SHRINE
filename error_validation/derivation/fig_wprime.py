import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

N = 512; t = np.arange(N)
s = np.exp(-0.5*((t-256)/12.0)**2)                      # toy burst at d0
k = np.arange(1, N+1); kc, O = 30, 5
f = 1/(1+(k/kc)**(2*O))                                  # S23 f(k)
S = lambda x: idct(f*dct(x, norm='ortho'), norm='ortho') # S = C diag(f) C^T
D1 = lambda x: np.diff(x)                                # (D1 x)_j = x_{j+1}-x_j
D1T = lambda y: np.concatenate(([0], y)) - np.concatenate((y, [0]))  # row a^T D1

a = D1(S(s)); ahat = a/np.linalg.norm(a)
w = S(D1T(ahat))                                         # w = a^T D1 S  (S symmetric)
w /= np.abs(w).max()
wdot = np.gradient(w)

rng = np.random.default_rng(3)
blue, orange, grey = "#2a78d6", "#eb6834", "#9a9a9a"
slides = [(1.0, blue, r"a channel near the middle of the band"), (3.0, orange, r"a channel near the bottom of the band")]
lo, hi = 196, 316; m = (t >= lo) & (t <= hi)

fig, ax = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True,
                       gridspec_kw=dict(height_ratios=[1, 1, 1.05], hspace=0.32))
for i, (sl, c, name) in enumerate(slides):
    x = 0.35*rng.standard_normal(N)
    ws = np.interp(t + sl, t, w)                         # w(t + τ_ch Δd)
    A = ax[i]
    A.plot(t[m], x[m], color=grey, lw=0.8, alpha=0.6, label=r"raw noise $x_{\nu}(t)$ (fixed)")
    A.plot(t[m], w[m], color=c, lw=2, label=r"weights at $d_0$")
    A.plot(t[m], ws[m], color=c, lw=2, ls="--", label=r"weights at $d_0+\Delta d$")
    A.annotate("", xy=(290-sl, 0.62), xytext=(290, 0.62),
               arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    A.text(290+1.5, 0.62, f"slides {sl:g} sample{'s' if sl!=1 else ''}", va="center", fontsize=9, backgroundcolor="white")
    A.set_title(name[0].upper()+name[1:], loc="left", fontsize=11)
    A.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    A.set_yticks([]); A.set_ylim(-1.15, 1.05)
    for sp in ("top", "right", "left"): A.spines[sp].set_visible(False)
    ax[2].plot(t[m], (ws - w)[m], color=c, lw=2, label=f"{name}")
    
A = ax[2]
A.axhline(0, color="#cccccc", lw=0.8, zorder=0)
A.set_title(r"$\Delta d\cdot w^\prime_{\nu}(t)$: change in the weight multiplying each raw noise sample", loc="left", fontsize=11)
A.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
A.set_yticks([])
for sp in ("top", "right", "left"): A.spines[sp].set_visible(False)
A.set_xlabel("time sample $t$")
fig.savefig("fig_wprime.png", dpi=170, bbox_inches="tight")
