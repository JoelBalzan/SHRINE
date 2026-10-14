"""Check the general sigma_DM recipe (Step 6g) on a drifting burst, where the same-profile
closed form does not apply. Everything follows the walkthrough notation."""
import numpy as np, sys
from scipy.fft import dct, idct, rfft, irfft, rfftfreq
rng = np.random.default_rng(7)
N, Nch = 1024, 32
t = np.arange(N); f = rfftfreq(N)
nu = np.linspace(1.1, 1.5, Nch)
tau = 60*(nu**-2 - nu.max()**-2)              # samples per unit d, nu_ref = top channel
A = (nu/1.3)**-1.5
drift = 12*(1.3 - nu)/0.2                     # lower frequencies later: 'sad trombone' drift
g = lambda x, s: np.exp(-0.5*(x/s)**2)
P = np.array([A[c]*(g(t-495-drift[c], 7) + 0.7*g(t-525-1.5*drift[c], 5)) for c in range(Nch)])
k = np.arange(1, N+1); kc, O = 40, 5
fk = 1/(1+(k/kc)**(2*O))
S   = lambda x: idct(fk*dct(x, norm='ortho', axis=-1), norm='ortho', axis=-1)
D1  = lambda x: np.diff(x, axis=-1)
D1T = lambda y: np.concatenate(([0], y)) - np.concatenate((y, [0]))
def series(Xf, d):          # n(t; d) = sum_ch x_ch(t - d tau_ch)
    return irfft((Xf*np.exp(-2j*np.pi*np.outer(tau, f)*d)).sum(0), n=N)
def Scurve(Xf, ds):
    ph = np.exp(-2j*np.pi*tau[:, None, None]*f[None, None, :]*np.asarray(ds)[None, :, None])
    return np.linalg.norm(D1(S(irfft(np.einsum('cf,cdf->df', Xf, ph), n=N))), axis=-1)
def wrow(Xf, d):            # w(t; d) = ahat^T D1 S, ahat from the series dedispersed at d
    a = D1(S(series(Xf, d))); return S(D1T(a/np.linalg.norm(a)))
def wshift(w, d):           # w(t + d tau_ch; d) for every channel
    return irfft(rfft(w)[None, :]*np.exp(2j*np.pi*np.outer(tau, f)*d), n=N)
def wprime(Xf, d, dl):      # central difference of w(t + d tau_ch; d)
    return (wshift(wrow(Xf, d+dl), d+dl) - wshift(wrow(Xf, d-dl), d-dl))/(2*dl)
def peak(ds, Sd, half):     # parabola fit to S(d) within +-half of the grid maximum
    j = np.argmax(Sd); m = np.abs(ds - ds[j]) <= half
    c = np.polyfit(ds[m], Sd[m], 2); return -c[1]/(2*c[0]), 2*c[0]

Pf = rfft(P)
# noise-free curve: d0 and ||a||''(d0)
dc = np.linspace(-2, 1, 601); jc = np.argmax(Scurve(Pf, dc))
dfine = dc[jc] + np.linspace(-0.02, 0.02, 81); d0, _ = peak(dfine, Scurve(Pf, dfine), 0.02)
assert abs(d0-dc[jc]) < 0.01 and 0 < jc < len(dc)-1
h = 1e-3
curv0 = (Scurve(Pf, [d0+h])[0] - 2*Scurve(Pf, [d0])[0] + Scurve(Pf, [d0-h])[0])/h**2
sx = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
wp0 = wprime(Pf, d0, 1e-3)
sig_true = sx*np.sqrt((wp0**2).sum())/abs(curv0)
# same-profile closed form evaluated anyway (wrong model here) for comparison
taubar = (A*tau).sum()/A.sum(); w0 = wrow(Pf, d0)
wdot = irfft(rfft(w0)*(2j*np.pi*f), n=N)
sig_same = sx*np.sqrt(((tau-taubar)**2).sum()*(wdot**2).sum())/abs(curv0)
print(f"d0 (noise-free structure peak) = {d0:.4f}   ||a||''(d0) = {curv0:.4g}")
print(f"6g formula, true burst: {sig_true:.4g}   same-profile closed form (not valid here): {sig_same:.4g}")

# Monte Carlo + per-realisation estimates from the data
ds = d0 + np.linspace(-6*sig_true, 6*sig_true, 121)
R = int(sys.argv[2]) if len(sys.argv) > 2 else 300; out = []
for r in range(R):
    Xf = Pf + rfft(sx*rng.standard_normal((Nch, N)))
    Sd = Scurve(Xf, ds)
    dh, _ = peak(ds, Sd, 1.0*sig_true)
    row = [dh]
    for half in (1.5, 3, 6):                           # fit windows, in units of sig_true
        row.append(abs(peak(ds, Sd, half*sig_true)[1]))
    for hh in (1, 2, 4):                               # finite-difference steps
        S3 = Scurve(Xf, [dh-hh*sig_true, dh, dh+hh*sig_true])
        row.append(abs((S3[0]-2*S3[1]+S3[2])/(hh*sig_true)**2))
    for dl in (0.1, 0.5):                              # delta for w'_ch, in units of sig_true
        row.append(sx*np.sqrt((wprime(Xf, dh, dl*sig_true)**2).sum()))
    out.append(row)
out = np.array(out); mc = out[:, 0].std()
print(f"MC std(d-hat) = {mc:.4g}  (+-{mc/np.sqrt(2*R):.2g})   6g(true)/MC = {sig_true/mc:.3f}   mean offset {out[:,0].mean()-d0:+.2g}")
names = ['fit +-1.5sig', 'fit +-3sig', 'fit +-6sig', 'FD h=1sig', 'FD h=2sig', 'FD h=4sig']
for i, nm in enumerate(names):
    c = out[:, 1+i]; print(f"  curvature {nm:13s}: median/true {np.median(c)/abs(curv0):.3f}   16-84% {np.percentile(c,16)/abs(curv0):.2f}-{np.percentile(c,84)/abs(curv0):.2f}")
for i, dl in enumerate((0.1, 0.5)):
    num = out[:, 7+i]
    est = num/out[:, 2]   # with the +-3 sig fit curvature
    print(f"  numerator (delta={dl}sig) median/true {np.median(num)/(sig_true*abs(curv0)):.3f};  full data estimate sigma_DM/MC: median {np.median(est)/mc:.3f}, 16-84% {np.percentile(est,16)/mc:.2f}-{np.percentile(est,84)/mc:.2f}")
print(f"S/N: summed peak/noise = {series(Pf,d0).max()/(sx*np.sqrt(Nch)):.1f}")
