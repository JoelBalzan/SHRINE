import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
"""Toy SHRINE validation: known-DM FRB, many noise realisations.

Per-channel intensity (chi^2_4 self-noise, |X|^2+|Y|^2), channel-normalised and
summed after a fractional (Fourier) redispersion shift per trial DM, as in
generate_profiles.py. Then SHRINE (struct_max_core) is run unchanged.
"""
import numpy as np
from scipy.fftpack import dct
import struct_max_core as smc

rng = np.random.default_rng(1)

NCH, NT, DT = 84, 2048, 1.0              # channels, samples, us
f = np.linspace(1440, 1104, NCH, endpoint=False) - 2.0   # MHz
KDM = 4.149e9                            # us MHz^2 / (pc cm^-3)
fref = np.median(f)
tau_unit = KDM * (f**-2 - fref**-2)      # us per unit dDM
dDMs = np.round(np.arange(-0.04, 0.0401, 0.001), 4)
ND = len(dDMs)
t = np.arange(NT) * DT
nu = np.fft.rfftfreq(NT, DT)
PH = np.exp(-2j*np.pi*nu[None, None, :]*(dDMs[:, None, None]*tau_unit[None, :, None]))  # (ND,NCH,NF)

def profile(amp):
    s = (np.exp(-0.5*((t-1000)/12)**2) + 0.7*np.exp(-0.5*((t-1050)/20)**2))
    return amp * s

SELFNOISE = True
def realise(amp, with_signal=True):
    S = profile(amp)[None, :] * np.ones((NCH, 1)) if with_signal else np.zeros((NCH, NT))
    # |X|^2+|Y|^2 with X,Y complex Gaussian of power (1+S)/2 each -> mean 1+S, chi2_4 stats
    X = (rng.standard_normal((NCH, NT)) + 1j*rng.standard_normal((NCH, NT)))*np.sqrt((1+S)/4)
    Y = (rng.standard_normal((NCH, NT)) + 1j*rng.standard_normal((NCH, NT)))*np.sqrt((1+S)/4)
    I = np.abs(X)**2 + np.abs(Y)**2
    if not SELFNOISE:
        I = np.abs(X/np.sqrt(1+S))**2 + np.abs(Y/np.sqrt(1+S))**2 + S   # additive (radiometer) noise only
    I = (I - I.mean(1, keepdims=True)) / I.std(1, keepdims=True)
    FI = np.fft.rfft(I, axis=1)
    return np.fft.irfft((FI[None]*PH).sum(1), n=NT, axis=1)   # (ND, NT) I(DM,t)

if __name__ == "__main__":
    import sys
    amp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    NR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    data = [realise(amp) for _ in range(NR)]
    np.save(f"sims_amp{amp}.npy", np.array(data, dtype=np.float32))
    kc = smc.get_kc(dct(data[0], norm="ortho"))
    print("kc", kc)
