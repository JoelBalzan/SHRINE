"""Parameterisable core math for SHRINE structure maximisation.

Extracts the DCT-based structure parameter, low-pass filter, uncertainty,
and kc detection into callable functions where filter order and kc are
explicit arguments rather than hardcoded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.fftpack import dct, idct


# ---------------------------------------------------------------------------
# kc detection (from dm_processing.py, duplicated to avoid path hacks)
# ---------------------------------------------------------------------------

def get_kc(CI_data: np.ndarray) -> int:
    """Determine spectral cutoff kc for the low-pass filter.

    Parameters
    ----------
    CI_data : ndarray, shape (dm, k)
        FRB data in DCT domain (absolute values for all delta-DM).

    Returns
    -------
    kc : int
        Cutoff index.
    """
    cumulative_window_size = 5
    noise_margin = int(CI_data.shape[1] / 2)
    CI_transposed = np.transpose(np.abs(CI_data))
    CI_maxima = np.max(CI_transposed, axis=1)
    noise_top = np.mean(CI_maxima[noise_margin:])

    for i in range(cumulative_window_size, len(CI_transposed)):
        rolling_avg = np.mean(CI_maxima[i - cumulative_window_size : i + 1])
        if rolling_avg <= noise_top:
            return i

    raise RuntimeError("Could not find a value for kc")


# ---------------------------------------------------------------------------
# low-pass filter response
# ---------------------------------------------------------------------------

def lowpass_response(k_length: int, kc: int, order: int = 3) -> np.ndarray:
    """Return the Butterworth low-pass filter response fL(k).

    Parameters
    ----------
    k_length : int
        Number of DCT coefficients (k axis length).
    kc : int
        Spectral cutoff index.
    order : int
        Butterworth filter order O.  The response is
        ``fL = 1 / (1 + (k/kc)^(2*order))``.

    Returns
    -------
    fL : ndarray, shape (k_length,)
    """
    k = np.linspace(1, k_length, k_length)
    fL = 1.0 / (1.0 + (k / kc) ** (2 * order))
    return fL


def highpass_eigenvalues(k_length: int) -> np.ndarray:
    """Square root of the eigenvalues of D_1^T D_1 (1st-derivative high-pass)."""
    k = np.linspace(1, k_length, k_length)
    return np.sqrt(2.0 - 2.0 * np.cos((k - 1) * np.pi / k_length))


# ---------------------------------------------------------------------------
# structure parameter computation
# ---------------------------------------------------------------------------

def structure_param(i_data: np.ndarray,
                    order: int = 3,
                    kc: Optional[int] = None,
                    ) -> dict:
    """Compute the SHRINE structure parameter vs delta-DM.

    Parameters
    ----------
    i_data : ndarray, shape (n_dm, n_time)
        I(DM, t) profile array (rows = DM trials, cols = time samples).
    order : int
        Butterworth filter order.
    kc : int or None
        Spectral cutoff.  If *None*, auto-detected via :func:`get_kc`.

    Returns
    -------
    result : dict
        ``dm_data``        — copy of input DM array
        ``sp``             — structure parameter vs DM, shape (n_dm,)
        ``kc``             — kc used
        ``order``          — filter order used
        ``i_smooth``       — smoothed I(DM, t)
        ``delta_I``        — detrended noise (I − I_smooth)
        ``uncertainty``    — formal uncertainty vs DM
        ``relative_uncertainty`` — relative uncertainty vs DM
    """
    i_data = np.asarray(i_data, dtype=float)

    CI_data = dct(i_data, norm="ortho")
    n_dm, k_length = CI_data.shape

    if kc is None:
        kc = get_kc(CI_data)

    fL = lowpass_response(k_length, kc, order)
    fL_diag = np.diag(fL)

    # LPF smoothing
    LPF_data = fL_diag @ np.transpose(CI_data)
    I_smooth = idct(np.transpose(LPF_data), norm="ortho")

    # 1st-derivative high-pass
    hp = highpass_eigenvalues(k_length)

    # Combined filter
    combined = hp * fL
    combined_diag = np.diag(combined)

    CI_filtered = combined_diag @ np.transpose(CI_data)
    sp = linalg.norm(CI_filtered, axis=0)

    # Uncertainty
    delta_I = i_data - I_smooth
    uncertainty = _uncertainty_calc(delta_I, LPF_data, fL_diag)

    max_idx = int(np.argmax(sp))
    delta_delta_I = delta_I - delta_I[max_idx]
    relative_uncertainty = _uncertainty_calc(delta_delta_I, LPF_data, fL_diag)

    return {
        "sp": sp,
        "kc": kc,
        "order": order,
        "i_smooth": I_smooth,
        "delta_I": delta_I,
        "uncertainty": uncertainty,
        "relative_uncertainty": relative_uncertainty,
    }


# ---------------------------------------------------------------------------
# structure parameter from pre-computed DCT data (for sweeping without re-DCTing)
# ---------------------------------------------------------------------------

def structure_param_from_dct(CI_data: np.ndarray,
                             kc: int,
                             order: int = 3) -> dict:
    """Structure parameter from already-DCT'd data.

    Same as :func:`structure_param` but skips the DCT step, useful when
    sweeping order/kc on the same input.
    """
    CI_data = np.asarray(CI_data, dtype=float)
    n_dm, k_length = CI_data.shape

    fL = lowpass_response(k_length, kc, order)
    fL_diag = np.diag(fL)

    LPF_data = fL_diag @ np.transpose(CI_data)
    I_smooth = idct(np.transpose(LPF_data), norm="ortho")

    hp = highpass_eigenvalues(k_length)
    combined = hp * fL
    combined_diag = np.diag(combined)

    CI_filtered = combined_diag @ np.transpose(CI_data)
    sp = linalg.norm(CI_filtered, axis=0)

    # Uncertainty (same recipe as structure_param)
    i_data = idct(CI_data, norm="ortho")
    delta_I = i_data - I_smooth
    uncertainty = _uncertainty_calc(delta_I, LPF_data, fL_diag)

    max_idx = int(np.argmax(sp))
    delta_delta_I = delta_I - delta_I[max_idx]
    relative_uncertainty = _uncertainty_calc(delta_delta_I, LPF_data, fL_diag)

    return {
        "sp": sp,
        "kc": kc,
        "order": order,
        "i_smooth": I_smooth,
        "delta_I": delta_I,
        "uncertainty": uncertainty,
        "relative_uncertainty": relative_uncertainty,
    }


# ---------------------------------------------------------------------------
# DM range extraction from relative uncertainty
# ---------------------------------------------------------------------------

def dm_range_from_relative_uncertainty(sp: np.ndarray,
                                       relative_uncertainty: np.ndarray,
                                       dm_data: np.ndarray,
                                       ) -> Tuple[Optional[float], Optional[float]]:
    """Return (low_dm, high_dm) uncertainty bounds using SHRINE recipe.

    Adjusts SP by relative uncertainty and finds the contiguous range above
    the peak SP value.
    """
    sp = np.asarray(sp, dtype=float)
    rel = np.asarray(relative_uncertainty, dtype=float)
    dm = np.asarray(dm_data, dtype=float)

    max_idx = int(np.argmax(sp))
    max_sp = float(sp[max_idx])
    adjusted = sp + sp * rel

    ranges = _get_ranges_above_max(max_sp, adjusted)
    if not ranges:
        return None, None

    low_dm = float(dm[ranges[0][0]])
    high_dm = None
    last = ranges[-1]
    if len(last) == 2:
        high_dm = float(dm[last[1]])
    elif len(last) == 1:
        high_dm = float(dm[-1])

    return low_dm, high_dm


def recovered_dm(dm_data: np.ndarray, sp: np.ndarray) -> float:
    """Return the DM at the structure parameter peak."""
    return float(dm_data[int(np.argmax(sp))])


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _uncertainty_calc(detrended_noise: np.ndarray,
                      LPF_data: np.ndarray,
                      filter_diag: np.ndarray) -> np.ndarray:
    """SHRINE uncertainty calculation (from dm_processing.py)."""
    CI_dbl_filtered = filter_diag @ LPF_data
    norm_CI_dbl_filtered = linalg.norm(CI_dbl_filtered, axis=0)

    C_delta_I = dct(detrended_noise, norm="ortho")
    C_delta_I_filtered = filter_diag @ np.transpose(C_delta_I)
    norm_C_delta_I_filtered = linalg.norm(C_delta_I_filtered, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        uncertainty = norm_C_delta_I_filtered / norm_CI_dbl_filtered
    uncertainty = np.where(np.isfinite(uncertainty), uncertainty, 0.0)
    return uncertainty


def _get_ranges_above_max(max_SP: float, adjusted_SPs: np.ndarray) -> list:
    """Find contiguous ranges above max_SP (from dm_processing.py)."""
    possible_max_ranges = []
    continuous = False
    for idx, SP in enumerate(adjusted_SPs):
        if not continuous and SP >= max_SP:
            continuous = True
            possible_max_ranges.append([idx])
        elif continuous and SP < max_SP:
            continuous = False
            possible_max_ranges[-1].append(idx)
    return possible_max_ranges
