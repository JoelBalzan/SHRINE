# Is the MC excess over the 6g formula second order?  Compare d-hat from the full S(d) with
# d-hat from the first-order curve ||a(d)|| + e_par(d), e_par(d) = sum_t w(t;d) n(t;d) (true w).
import numpy as np, sys
exec(open('general_check.py').read().split("sx = float")[0])     # reuse definitions, d0, curv0
sx = float(sys.argv[1]); R = int(sys.argv[2])
sig_true = sx*np.sqrt((wprime(Pf, d0, 1e-3)**2).sum())/abs(curv0)
ds = d0 + np.linspace(-6*sig_true, 6*sig_true, 121)
acurve = Scurve(Pf, ds)
W = np.array([wrow(Pf, d) for d in ds])                          # true weight rows
ph = np.exp(-2j*np.pi*tau[:, None, None]*f[None, None, :]*ds[None, :, None])
full, lin = [], []
for r in range(R):
    Nf = rfft(sx*rng.standard_normal((Nch, N)))
    nser = irfft(np.einsum('cf,cdf->df', Nf, ph), n=N)              # n(t; d) for each d
    full.append(peak(ds, Scurve(Pf+Nf, ds), sig_true)[0])
    lin.append(peak(ds, acurve + (W*nser).sum(1), sig_true)[0])
full, lin = np.array(full), np.array(lin); e = 1/np.sqrt(2*R)
print(f"S/N {series(Pf,d0).max()/(sx*np.sqrt(Nch)):.0f}: 6g/MC(full) = {sig_true/full.std():.3f}+-{e*sig_true/full.std():.3f}   6g/MC(first-order curve) = {sig_true/lin.std():.3f}+-{e*sig_true/lin.std():.3f}")
