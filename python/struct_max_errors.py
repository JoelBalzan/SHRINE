"""Statistically calibrated uncertainties for the SHRINE structure parameter.

Replaces the triangle-inequality error of Sutinjo et al. (2023), Eqs. (7)-(11),
which uses the full N-dimensional norm of the filtered *residual*
||D S Delta|| as a 1-sigma error. That quantity is (i) the radius of the noise
sphere, not its projection onto the signal direction (overestimate by
~sqrt(N_eff)), and (ii) contaminated by burst power above kc that sits in
Delta = (1 - S) i (deterministic, does not fluctuate between realisations).

To first order in noise, SP = ||g c + g n|| = ||g c|| + s_hat . (g n) + O(n^2),
with g = hp * fL (the combined filter) and s_hat = g c / ||g c||.
Only the projection s_hat . (g n) moves SP, so

    sigma_SP^2 = sum_t w(t)^2 sigma_t^2,   w = IDCT(g^2 c_hat) / ||g c_hat||,

with c_hat = fL * DCT(i) and sigma_t^2 the local (self-noise inclusive) noise
variance, measured from the signal-free band k > N/4.

The DM error is the scatter of the peak position under the same linear
perturbation, evaluated with noise-only I(DM, t) cubes made by passing
off-pulse voltage segments through generate_profiles.py unchanged. This
carries the (strong) noise correlation between neighbouring trial DMs, which
no single-realisation formula captures.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct

from struct_max_core import highpass_eigenvalues, lowpass_response


def _filters(n_time: int, kc: int, order: int):
    fL = lowpass_response(n_time, kc, order)
    g = highpass_eigenvalues(n_time) * fL
    return fL, g


def local_noise_var(i_data: np.ndarray, kmin_frac: float = 0.25,
                    window: int = 31) -> np.ndarray:
    """Local noise variance sigma_t^2 for every DM row, from the band k > kmin_frac*N.

    The burst has no power there, so this sees radiometer + self-noise only.
    """
    i_data = np.atleast_2d(np.asarray(i_data, float))
    n = i_data.shape[1]
    kmin = int(kmin_frac * n)
    C = dct(i_data, norm="ortho")
    C[:, :kmin] = 0.0
    hi = idct(C, norm="ortho")
    ker = np.ones(window) / window
    v = np.array([np.convolve(r**2, ker, mode="same") for r in hi])
    return v / (1.0 - kmin / n)


def projection_weights(i_data: np.ndarray, kc: int, order: int = 3) -> np.ndarray:
    """Time-domain weights w_d(t) such that delta SP_d = sum_t w_d(t) n_d(t)."""
    i_data = np.atleast_2d(np.asarray(i_data, float))
    fL, g = _filters(i_data.shape[1], kc, order)
    gc = g[None] * fL[None] * dct(i_data, norm="ortho")
    return idct(g[None] * gc, norm="ortho") / np.linalg.norm(gc, axis=1)[:, None]


def sp_sigma(i_data: np.ndarray, kc: int, order: int = 3) -> np.ndarray:
    """1-sigma uncertainty of the structure parameter at every trial DM.

    Drop-in replacement for ``uncertainty * sp`` from SHRINE (absolute units).
    """
    W = projection_weights(i_data, kc, order)
    return np.sqrt(np.sum(W**2 * local_noise_var(i_data), axis=1))


def _peak(y: np.ndarray, x: np.ndarray) -> float:
    k = int(np.clip(np.argmax(y), 1, len(y) - 2))
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    return x[k] + (0.5 * (a - c) / den if den != 0 else 0.0) * (x[1] - x[0])


def dm_uncertainty(i_data: np.ndarray, dm: np.ndarray, noise_cubes: np.ndarray,
                   kc: int, order: int = 3, half_window: int = 3,
                   selfnoise: bool = True) -> dict:
    """Structure-maximising DM and its 1-sigma error.

    Parameters
    ----------
    i_data : (n_dm, n_time) on-pulse I(DM, t) from generate_profiles.py
    dm : (n_dm,) trial DMs
    noise_cubes : (J, n_dm, n_time) the same product built from J off-pulse
        voltage crops (same crop length, DM grid, dt). J ~ 100-200.
    selfnoise : rescale the off-pulse noise by the on-pulse/off-pulse local
        noise ratio measured at the peak DM (bright bursts).
    """
    i_data = np.asarray(i_data, float)
    n_time = i_data.shape[1]
    _, g = _filters(n_time, kc, order)
    sp = np.linalg.norm(g[None] * dct(i_data, norm="ortho"), axis=1)
    dpk = int(np.argmax(sp))
    dm_hat = _peak(sp, dm)

    W = projection_weights(i_data, kc, order)
    scale = np.ones(n_time)
    if selfnoise:
        v = local_noise_var(i_data[dpk:dpk + 1])[0]
        off = np.asarray(noise_cubes, float)[:, dpk].var()
        scale = np.sqrt(v / off)
    dSP = np.einsum("jdt,dt->jd", np.asarray(noise_cubes, float) * scale, W)

    win = slice(max(dpk - half_window, 0), min(dpk + half_window + 1, len(dm)))
    model = np.polyval(np.polyfit(dm[win], sp[win], 2), dm)
    samples = np.array([_peak(model + d, dm) for d in dSP])
    return {"dm_hat": dm_hat, "sigma_dm": float(np.std(samples)),
            "samples": samples + (dm_hat - np.mean(samples)),
            "sp": sp, "sigma_sp": sp_sigma(i_data, kc, order)}
