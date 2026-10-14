import numpy as np
from scipy.fft import dct, idct, rfft, irfft, rfftfreq
rng = np.random.default_rng(1)
N, Nch = 1024, 32
t = np.arange(N); f = rfftfreq(N)
# same profile in every channel: two Gaussian components
p = np.exp(-0.5*((t-500)/8)**2) + 0.6*np.exp(-0.5*((t-520)/5)**2)
nu = np.linspace(1.1, 1.5, Nch)
tau = 60*(nu**-2 - nu.max()**-2)          # samples per unit d, nu_ref = TOP channel
A = 2.0*(nu/1.3)**-1.5                     # unequal channel amplitudes
taubar = np.sum(A*tau)/np.sum(A)
k = np.arange(1, N+1); kc, O = 40, 5
fk = 1/(1+(k/kc)**(2*O))
S  = lambda x: idct(fk*dct(x, norm='ortho', axis=-1), norm='ortho', axis=-1)
D1 = lambda x: np.diff(x, axis=-1)
D1T = lambda y: np.concatenate(([0], y)) - np.concatenate((y, [0]))
def dedisp(Xf, d):                         # Xf: (..., Nch, Nf) channel spectra; shift ch by d*tau
    ph = np.exp(-2j*np.pi*np.outer(tau, f)*d)
    return irfft((Xf*ph).sum(-2), n=N)
Pf = rfft(A[:, None]*p[None, :])           # noise-free channel data, aligned at d0 = 0
Snorm = lambda x: np.linalg.norm(D1(S(x)), axis=-1)

# noise-free curvature, numerical
h = 1e-3
c_num = (Snorm(dedisp(Pf, h)) - 2*Snorm(dedisp(Pf, 0)) + Snorm(dedisp(Pf, -h)))/h**2
# predicted: [sum A (tau-taubar)^2] * sum_t w(t) pddot(t)
a0 = D1(S(dedisp(Pf, 0))); ahat = a0/np.linalg.norm(a0)
w = S(D1T(ahat))
pddot = irfft(rfft(p)*(2j*np.pi*f)**2, n=N)
wdot  = irfft(rfft(w)*(2j*np.pi*f), n=N)
c_pred = np.sum(A*(tau-taubar)**2) * np.sum(w*pddot)
print(f"||a||''(d0): numerical {c_num:.6g}   predicted {c_pred:.6g}   ratio {c_num/c_pred:.4f}")

# Monte Carlo of d-hat vs 6h formula
sx = 1.0
sig_pred = sx*np.sqrt(np.sum((tau-taubar)**2))*np.sqrt(np.sum(wdot**2))/abs(c_pred)
dg = np.linspace(-4*sig_pred, 4*sig_pred, 81)
ph = np.exp(-2j*np.pi*tau[:, None, None]*f[None, None, :]*dg[None, :, None])  # (Nch, Nd, Nf)
dhat = []
for r in range(500):
    Xf = Pf + rfft(sx*rng.standard_normal((Nch, N)))
    series = irfft(np.einsum('cf,cdf->df', Xf, ph), n=N)                # (Nd, N)
    Sd = Snorm(series)
    j = np.argmax(Sd); j = min(max(j, 2), len(dg)-3)
    cc = np.polyfit(dg[j-2:j+3], Sd[j-2:j+3], 2); dhat.append(-cc[1]/(2*cc[0]))
dhat = np.array(dhat)
print(f"sigma_DM: MC {dhat.std():.4g}   formula {sig_pred:.4g}   ratio {dhat.std()/sig_pred:.3f}   mean dhat {dhat.mean():.2g}")
print(f"S/N check: peak of summed burst / summed noise std = {dedisp(Pf,0).max()/ (sx*np.sqrt(Nch)):.1f}")
