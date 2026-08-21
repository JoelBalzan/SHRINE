#!/usr/bin/env python3
"""Streamlined CLI for SHRINE structure maximisation.

A stripped-down entry point that keeps only the flags needed for
structure-parameter maximisation.  Supports:
  - Individual Stokes I/Q/U/V .npy files
  - Combined Stokes cube (.npy or .npz)
  - Manual / automatic peak selection
  - Configurable filter order and kc
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from struct_max_core import (dm_range_from_relative_uncertainty, get_kc,
                             recovered_dm, structure_param)

# ──────────────────────────────────────────────────────────────────────
# Stokes I/O  (inlined from frbop/utils/stokes_io.py to stay standalone)
# ──────────────────────────────────────────────────────────────────────

_DM_CONSTANT = 4.148808e6  # ms MHz^2 pc^-1 cm^3


def _normalise_pol_order(pol_order) -> str:
    if pol_order is None:
        return ""
    value = pol_order
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.dtype.kind in {"S", "U", "O"}:
            parts = []
            for item in value.ravel():
                if isinstance(item, (bytes, np.bytes_)):
                    parts.append(item.decode())
                else:
                    parts.append(str(item))
            value = "".join(parts)
        else:
            value = "".join(str(item) for item in value.ravel())
    elif isinstance(value, (bytes, np.bytes_)):
        value = value.decode()
    else:
        value = str(value)
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _load_stokes_cube_from_npz(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as archive:
        if "stokes" not in archive.files:
            raise ValueError(f"{npz_path} does not contain a 'stokes' array")
        cube = np.asarray(archive["stokes"])
        metadata = {k: archive[k] for k in archive.files if k != "stokes"}
    return cube, metadata


def _extract_stokes_components(cube, pol_order=None, nchan=None, nsamp=None):
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Stokes cube must be 3D, got shape {cube.shape}")

    order = _normalise_pol_order(pol_order)
    if not order:
        if cube.shape[0] in (3, 4):
            order = "IQUV"[: cube.shape[0]]
        elif cube.shape[-1] in (3, 4):
            order = "IQUV"[: cube.shape[-1]]
        else:
            raise ValueError("Cannot infer polarisation order from cube shape")

    n_stokes = len(order)
    aligned = None
    if nchan is not None and nsamp is not None:
        nchan = int(np.asarray(nchan).reshape(()))
        nsamp = int(np.asarray(nsamp).reshape(()))
        for stokes_axis in range(3):
            if cube.shape[stokes_axis] != n_stokes:
                continue
            remaining = [a for a in range(3) if a != stokes_axis]
            for freq_axis in remaining:
                for time_axis in remaining:
                    if freq_axis == time_axis:
                        continue
                    if cube.shape[freq_axis] == nchan and cube.shape[time_axis] == nsamp:
                        aligned = np.moveaxis(cube, (stokes_axis, freq_axis, time_axis), (0, 1, 2))
                        break
                if aligned is not None:
                    break
            if aligned is not None:
                break

    if aligned is None:
        if cube.shape[0] == n_stokes:
            aligned = cube
        elif cube.shape[-1] == n_stokes:
            aligned = np.moveaxis(cube, -1, 0)
        else:
            raise ValueError(f"Cannot identify Stokes axis in shape {cube.shape}")

    comp = {label: aligned[idx] for idx, label in enumerate(order)}
    return comp.get("I"), comp.get("Q"), comp.get("U"), comp.get("V"), aligned, order


def _derive_axes_from_metadata(metadata, nchan, nsamp):
    freq_mhz = None
    time_ms = None
    if "freq_mhz" in metadata:
        freq_mhz = np.asarray(metadata["freq_mhz"], dtype=float)
    elif "freq" in metadata:
        freq_mhz = np.asarray(metadata["freq"], dtype=float)
    if "time_ms" in metadata:
        time_ms = np.asarray(metadata["time_ms"], dtype=float)
    elif "time" in metadata:
        time_ms = np.asarray(metadata["time"], dtype=float)
    return freq_mhz, time_ms


# ──────────────────────────────────────────────────────────────────────
# Peak helpers  (inlined from frbop/utils/peaks.py)
# ──────────────────────────────────────────────────────────────────────

def _parse_peak_index_pairs(peak_indices, n_time):
    if peak_indices is None:
        return []
    values = list(peak_indices)
    if len(values) == 0:
        raise ValueError("--peak-indices requires at least one pair")
    if len(values) % 2 != 0:
        raise ValueError("--peak-indices requires an even number of values")
    regions = []
    for i in range(0, len(values), 2):
        s = int(np.clip(min(values[i], values[i + 1]), 0, n_time - 1))
        e = int(np.clip(max(values[i], values[i + 1]), 0, n_time))
        if e <= s:
            e = min(n_time, s + 1)
        regions.append((s, e))
    return regions


def _select_peaks_manual(time_axis, profile):
    """Interactive peak selection via matplotlib clicks."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, profile, color="k", linewidth=1)
    ax.set_title("Click start/end bounds for each peak (close window when done)")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("S [arb.]")
    ax.grid(True, alpha=0.3)
    cursor = ax.axvline(time_axis[0], color="tab:blue", alpha=0.4, linewidth=1)
    clicks: List[float] = []

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor.set_xdata([event.xdata, event.xdata])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x = float(event.xdata)
        clicks.append(x)
        ax.axvline(x, color="tab:red", alpha=0.7, linewidth=1)
        if len(clicks) % 2 == 0:
            lo, hi = sorted((clicks[-2], clicks[-1]))
            ax.axvspan(lo, hi, color="tab:orange", alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if not clicks:
        return [(0, time_axis.size)]
    if len(clicks) % 2 != 0:
        clicks = clicks[:-1]
    regions = []
    for i in range(0, len(clicks), 2):
        lo_t, hi_t = sorted((clicks[i], clicks[i + 1]))
        s = int(np.argmin(np.abs(time_axis - lo_t)))
        e = int(np.argmin(np.abs(time_axis - hi_t)))
        s, e = min(s, e), max(s, e) + 1
        e = min(time_axis.size, e)
        if e <= s:
            e = min(time_axis.size, s + 1)
        regions.append((s, e))
    return regions if regions else [(0, time_axis.size)]


# ──────────────────────────────────────────────────────────────────────
# Dedispersion  (faithful port of frbop/dmop/dedispersion.py)
# ──────────────────────────────────────────────────────────────────────

def _rescale_peak_indices(indices, factor):
    """Map original-resolution peak bounds onto a scrunched time axis."""
    if factor <= 1 or not indices:
        return list(indices) if indices is not None else []
    values = list(indices)
    scaled: list[int] = []
    for i in range(0, len(values), 2):
        scaled.append(int(values[i]) // factor)
        scaled.append(-(-int(values[i + 1]) // factor))
    return scaled


def _build_dm_values(dm_range: Tuple[float, float], n_points: int = 100,
                     dm_step: Optional[float] = None) -> np.ndarray:
    """Port of frbop MetricsMixin._build_dm_values."""
    if dm_step is not None:
        if dm_step <= 0:
            raise ValueError("dm_step must be positive")
        span = dm_range[1] - dm_range[0]
        n_points = max(2, int(np.floor(span / dm_step)) + 1)
        dm_values = dm_range[0] + np.arange(n_points) * dm_step
        if dm_values[-1] < dm_range[1] and (dm_range[1] - dm_values[-1]) > 0.5 * dm_step:
            dm_values = np.append(dm_values, dm_range[1])
    else:
        dm_values = np.linspace(dm_range[0], dm_range[1], n_points)
    return dm_values


class _Dedisperser:
    """Mirrors DMOptimiser's dedispersion exactly.

    Noise statistics come from the first 5% of the FULL (uncropped) time
    axis, fills are drawn per channel from a seeded generator, and rows are
    aligned at their own minimum shift ('crop' aligns at the maximum shift).
    """

    def __init__(self, stokes_i_full: np.ndarray, freq_mhz: np.ndarray,
                 time_ms: np.ndarray, input_dm: float = 0.0,
                 reference_freq: Optional[float] = None, seed: int = 1234):
        self.stokes_i = stokes_i_full
        self.freq_mhz = freq_mhz
        self.time_ms = time_ms
        self.input_dm = float(input_dm)
        self.reference_freq = (float(reference_freq) if reference_freq is not None
                               else float(np.max(freq_mhz)))
        self.rng = np.random.default_rng(seed)
        self.dt = float(np.median(np.diff(time_ms)))
        n_edge = max(1, int(0.05 * self.stokes_i.shape[1]))
        self.noise_std = np.std(self.stokes_i[:, :n_edge], axis=1)
        self.noise_mean = np.mean(self.stokes_i[:, :n_edge], axis=1)

    def delay_samples(self, dm: float) -> np.ndarray:
        effective_dm = self.input_dm - dm
        delays_ms = _DM_CONSTANT * effective_dm * (
            1.0 / self.freq_mhz**2 - 1.0 / self.reference_freq**2
        )
        return np.round(delays_ms / self.dt).astype(int)

    def max_output_size_for_dm_range(self, data: np.ndarray,
                                     dm_range: Tuple[float, float]) -> int:
        max_size = data.shape[1]
        for dm in dm_range:
            d = self.delay_samples(dm)
            max_size = max(max_size, data.shape[1] + int(np.max(d)) - int(np.min(d)))
        return max_size

    def _noise_fill(self, n_freq: int, n_time_out: int) -> np.ndarray:
        return self.rng.normal(
            self.noise_mean[:, None], self.noise_std[:, None],
            size=(n_freq, n_time_out),
        )

    def dedisperse(self, data: np.ndarray, dm: float,
                   output_size: Optional[int] = None,
                   mode: str = "expand") -> np.ndarray:
        delay_samples = self.delay_samples(dm)
        if mode == "crop":
            origin = int(np.max(delay_samples))
            if output_size is None:
                n_time_out = data.shape[1] + int(np.min(delay_samples)) - origin
            else:
                n_time_out = output_size
            out = self._noise_fill(data.shape[0], n_time_out)
        elif mode in ("expand", "expand_zero", "expand_nan"):
            origin = int(np.min(delay_samples))
            if output_size is None:
                n_time_out = data.shape[1] + int(np.max(delay_samples)) - origin
            else:
                n_time_out = output_size
            if mode == "expand_zero":
                out = np.zeros((data.shape[0], n_time_out), dtype=float)
            elif mode == "expand_nan":
                out = np.full((data.shape[0], n_time_out), np.nan)
            else:
                out = self._noise_fill(data.shape[0], n_time_out)
        else:
            raise ValueError(f"Unknown dedispersion mode '{mode}'")

        for i in range(data.shape[0]):
            shift = int(delay_samples[i])
            for t in range(data.shape[1]):
                t_out = t + shift - origin
                if 0 <= t_out < out.shape[1]:
                    out[i, t_out] = data[i, t]
        return out


# ──────────────────────────────────────────────────────────────────────
# Plotting  (ports of maximise_structure.py so saved figures match
# frbop shrine_logs output exactly)
# ──────────────────────────────────────────────────────────────────────

def plot_DM_index(dm_data: np.ndarray, prefix: str, dt_us: int):
    plt.plot(dm_data)
    plt.xlabel('DM Index')
    plt.ylabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.savefig(f"{prefix}_{dt_us}us_DM_index.png")
    plt.clf()


def plot_noisy_I_DM_t(I_data: np.ndarray, DM_data: np.ndarray, prefix: str, dt_us: int):
    imgplot_Inoisy = plt.imshow(
        I_data, aspect="auto",
        extent=(0, I_data.shape[1] * dt_us / 1000, DM_data[-1], DM_data[0]),
    )
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.savefig(f"{prefix}_{dt_us}us_I_DM_t.png")
    plt.clf()


def plot_I_at_max(I_series: np.ndarray, I_smooth_series: np.ndarray,
                  prefix: str, dt_us: int):
    plt.plot([i * dt_us / 1000 for i in range(len(I_series))], I_series,
             alpha=0.5, color='blue')
    plt.plot([i * dt_us / 1000 for i in range(len(I_smooth_series))],
             I_smooth_series, color='blue')
    plt.xlabel("Time (ms)")
    plt.ylabel("I")
    plt.savefig(f"{prefix}_{dt_us}us_I_max.png")
    plt.clf()


def plot_DCT_spectrum(CI_data: np.ndarray, prefix: str, dt_us: int, kc: int):
    plt.plot(np.transpose(np.abs(CI_data)), '.')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r'$\mathregular{k}$ Index')
    plt.ylabel(r'DCT coefficients, $\mathregular{|C^Ti|}$')
    plt.axvline(x=kc)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.savefig(f"{prefix}_{dt_us}us_DCT_v_k_index.png")
    plt.clf()


def plot_smooth_I_DM_t(I_smooth: np.ndarray, DM_data: np.ndarray,
                       prefix: str, dt_us: int):
    imgplot_Ismooth = plt.imshow(
        I_smooth, aspect="auto",
        extent=(0, I_smooth.shape[1] * dt_us / 1000, DM_data[-1], DM_data[0]),
    )
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.savefig(f"{prefix}_{dt_us}us_I_DM_t_smoothed.png")
    plt.clf()


def plot_detrended_noise(delta_I: np.ndarray, DM_data: np.ndarray,
                         prefix: str, dt_us: int):
    imgplot_DeltaI = plt.imshow(
        delta_I, aspect="auto",
        extent=(0, delta_I.shape[1] * dt_us / 1000, DM_data[-1], DM_data[0]),
    )
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.savefig(f"{prefix}_{dt_us}us_detrended_noise.png")
    plt.clf()


def plot_relative_detrended_noise(delta_delta_I: np.ndarray, DM_data: np.ndarray,
                                  prefix: str, dt_us: int):
    imgplot_DeltaDeltaI = plt.imshow(
        delta_delta_I, aspect="auto",
        extent=(0, delta_delta_I.shape[1] * dt_us / 1000, DM_data[-1], DM_data[0]),
    )
    plt.xlabel("Time (ms)")
    plt.ylabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.savefig(f"{prefix}_{dt_us}us_relative_detrended_noise.png")
    plt.clf()


def plot_noise_at_max(noise_at_max: np.ndarray, prefix: str, dt_us: int):
    plt.plot([time / 1000 for time in range(len(noise_at_max * dt_us))],
             noise_at_max)
    plt.xlabel("Time (ms)")
    plt.ylabel("I")
    plt.savefig(f"{prefix}_{dt_us}us_noise_max.png")
    plt.clf()


def plot_uncertainty(uncertainty: np.ndarray, DM_data: np.ndarray,
                     prefix: str, dt_us: int):
    plt.plot(DM_data, 100 * uncertainty, '-')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Uncertainty (%)')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.savefig(f"{prefix}_{dt_us}us_uncertainty.png")
    plt.clf()


def plot_relative_uncertainty(relative_uncertainty: np.ndarray, DM_data: np.ndarray,
                              prefix: str, dt_us: int):
    plt.plot(DM_data, 100 * relative_uncertainty, '-')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Relative Uncertainty (%)')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.savefig(f"{prefix}_{dt_us}us_relative_uncertainty.png")
    plt.clf()


def plot_compare_uncertainty(uncertainty: np.ndarray, relative_uncertainty: np.ndarray,
                             DM_data: np.ndarray, prefix: str, dt_us: int):
    plt.plot(DM_data, np.sqrt(2) * 100 * uncertainty, '-',
             label=r'$\sqrt{2}$ Uncertainty')
    plt.plot(DM_data, 100 * relative_uncertainty, '-',
             label='Relative Uncertainty')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Uncertainty (%)')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f"{prefix}_{dt_us}us_compare_uncertainty.png")
    plt.clf()


def plot_SP(SP_data: np.ndarray, DM_data: np.ndarray, prefix: str, dt_us: int):
    plt.plot(DM_data, SP_data, '-')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Structure Parameter')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{dt_us}us_structure_parameter.png")
    plt.clf()


def plot_adjusted_SP(adjusted_SPs: np.ndarray, DM_data: np.ndarray,
                     prefix: str, dt_us: int):
    plt.plot(DM_data, adjusted_SPs, '-')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Relative Uncertainty Adjusted Structure Parameter')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{prefix}_{dt_us}us_adjusted_SP.png")
    plt.clf()


def plot_norm_CI_f_adjusted_SP(norm_CI_filtered: np.ndarray, adjusted_SPs: np.ndarray,
                               DM_data: np.ndarray, prefix: str, dt_us: int):
    plt.plot(DM_data, norm_CI_filtered, '-', label='Structure Parameter')
    plt.plot(DM_data, adjusted_SPs, '-',
             label='Relative Uncertainty Adjusted Structure Parameter')
    plt.xlabel(r"$\Delta$DM ($\mathregular{pc\ cm^{-3}}$)")
    plt.ylabel('Structure Parameter')
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_{dt_us}us_norm_CI_f_adjusted_SP.png")
    plt.clf()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHRINE structure maximisation — streamlined CLI",
    )

    # -- Input data --
    data = parser.add_argument_group("Input data")
    data.add_argument(
        "--fires-dir", default=None,
        help="FIRES output directory (expects out_I.npy, freq.npy, time.npy). "
             "Shorthand for --stokes-i <dir>/out_I.npy --freq <dir>/freq.npy --time <dir>/time.npy.",
    )
    data.add_argument(
        "-d", "--stokes-cube", default=None,
        help="Path to Stokes I/Q/U/V cube (.npy or .npz). "
             "Accepted .npy shapes: (4,freq,time), (3,freq,time), (freq,time,4), (freq,time,3). "
             ".npz archives must contain a 'stokes' array.",
    )
    data.add_argument("--stokes-i", default=None, help="Stokes I .npy (freq x time)")
    data.add_argument("--stokes-q", default=None, help="Stokes Q .npy (freq x time)")
    data.add_argument("--stokes-u", default=None, help="Stokes U .npy (freq x time)")
    data.add_argument("--stokes-v", default=None, help="Stokes V .npy (freq x time)")
    data.add_argument("--freq", default="freq.npy", help="Frequency array [MHz]")
    data.add_argument("--time", default="time.npy", help="Time array [ms]")

    # -- DM grid --
    dm = parser.add_argument_group("DM grid")
    dm.add_argument("--dm-min", type=float, default=None, help="Min DM (pc cm^-3)")
    dm.add_argument("--dm-max", type=float, default=None, help="Max DM (pc cm^-3)")
    dm.add_argument("--dm-step", type=float, default=None, help="DM grid step")
    dm.add_argument("--dm-guess", type=float, default=None,
                    help="DM guess — builds range as guess +/- 50")
    dm.add_argument("--ref-freq", type=float, default=None, help="Reference freq [MHz]")
    dm.add_argument("--input-dm", type=float, default=0.0,
                    help="DM already applied to data")

    # -- Filter parameters --
    filt = parser.add_argument_group("Filter")
    filt.add_argument("-O", "--order", type=int, default=3,
                      help="Butterworth filter order (default: 3)")
    filt.add_argument("--kc", type=int, default=None,
                      help="Spectral cutoff (default: auto-detect)")

    # -- Peaks --
    peaks = parser.add_argument_group("Peak selection")
    peaks.add_argument("--peak-indices", nargs="*", type=int, default=None,
                       help="Peak index pairs: start1 end1 [start2 end2 ...]")
    peaks.add_argument("--manual-peaks", action="store_true",
                       help="Interactive peak selection")
    peaks.add_argument("--separate-peaks", action="store_true",
                       help="Auto-detect separate peaks")

    # -- Scrunching --
    scr = parser.add_argument_group("Scrunching")
    scr.add_argument("--tscrunch", "-ts", type=int, default=None,
                     help="Time scrunch factor")
    scr.add_argument("--fscrunch", "-fs", type=int, default=None,
                     help="Frequency scrunch factor")

    # -- Output --
    out = parser.add_argument_group("Output")
    out.add_argument("--label", default="frb", help="Output label")
    out.add_argument("--save-all", action="store_true", help="Save all plots")
    out.add_argument("--ext", default="png", help="Plot extension (default: png)")
    out.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible noise fill during dedispersion (default: 1234)",
    )
    out.add_argument(
        "--dedisp-mode",
        type=str,
        choices=["expand", "expand_zero", "expand_nan", "crop"],
        default="expand_nan",
        help="Dedispersion mode: 'expand' (fill edges with noise), 'expand_zero' (zero-fill new bins), 'expand_nan' (NaN-fill new bins), or 'crop' (trim to common valid region)",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load data ──
    print("Loading data...")
    derived_freq = None
    derived_time = None

    if args.fires_dir:
        fires_path = Path(args.fires_dir)
        stokes_i = np.load(fires_path / "out_I.npy")
        freq_mhz = np.load(fires_path / "freq.npy")
        time_ms = np.load(fires_path / "time.npy")
        stokes_q = fires_path / "out_Q.npy"
        stokes_q = np.load(stokes_q) if stokes_q.exists() else None
        stokes_u = fires_path / "out_U.npy"
        stokes_u = np.load(stokes_u) if stokes_u.exists() else None
        stokes_v = fires_path / "out_V.npy"
        stokes_v = np.load(stokes_v) if stokes_v.exists() else None
        print(f"  Loaded FIRES directory: {fires_path}")
        print(f"  Stokes I shape: {stokes_i.shape}  (freq x time)")
        print(f"  Freq: {freq_mhz[0]:.1f} - {freq_mhz[-1]:.1f} MHz")
        print(f"  Time: {time_ms[0]:.3f} - {time_ms[-1]:.3f} ms")
    elif args.stokes_cube:
        cube_path = Path(args.stokes_cube)
        if cube_path.suffix.lower() == ".npz":
            cube, meta = _load_stokes_cube_from_npz(cube_path)
            stokes_i, stokes_q, stokes_u, stokes_v, _, pol = \
                _extract_stokes_components(cube, pol_order=meta.get("pol_order"),
                                           nchan=meta.get("nchan"), nsamp=meta.get("nsamp"))
            derived_freq, derived_time = _derive_axes_from_metadata(meta, stokes_i.shape[0], stokes_i.shape[1])
            print(f"  Loaded Stokes archive: {cube_path}  (pol_order={pol!r})")
        else:
            cube = np.load(cube_path)
            stokes_i, stokes_q, stokes_u, stokes_v, _, _ = _extract_stokes_components(cube)
    else:
        if not args.stokes_i:
            print("Error: provide --fires-dir, --stokes-i, or --stokes-cube", file=sys.stderr)
            sys.exit(1)
        stokes_q = stokes_u = stokes_v = None
        stokes_i = np.load(args.stokes_i)
        if stokes_i.ndim == 3:
            # (pol, freq, time) or (freq, time, pol) style array
            stokes_i, q3d, u3d, v3d, _, pol3d = _extract_stokes_components(stokes_i)
            print(f"  Extracted Stokes I from 3D array (pol_order={pol3d!r})")
            stokes_q, stokes_u, stokes_v = q3d, u3d, v3d
        if args.stokes_q:
            stokes_q = np.load(args.stokes_q)
        if args.stokes_u:
            stokes_u = np.load(args.stokes_u)
        if args.stokes_v:
            stokes_v = np.load(args.stokes_v)

    # Load / derive axes (skip if --fires-dir already loaded them)
    if not args.fires_dir:
        freq_path = Path(args.freq) if args.freq else None
        if freq_path and freq_path.exists():
            freq_mhz = np.load(freq_path)
        else:
            freq_mhz = derived_freq
        if freq_mhz is None:
            print("Error: frequency axis required (--freq or in cube metadata)", file=sys.stderr)
            sys.exit(1)

        time_path = Path(args.time) if args.time else None
        if time_path and time_path.exists():
            time_ms = np.load(time_path)
        else:
            time_ms = derived_time
        if time_ms is None:
            print("Error: time axis required (--time or in cube metadata)", file=sys.stderr)
            sys.exit(1)

    if freq_mhz.ndim != 1:
        raise ValueError("Frequency array must be 1D")
    if not np.all(np.diff(freq_mhz) >= 0):
        order = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[order]
        stokes_i = stokes_i[order]
        if stokes_q is not None:
            stokes_q = stokes_q[order]
        if stokes_u is not None:
            stokes_u = stokes_u[order]
        if stokes_v is not None:
            stokes_v = stokes_v[order]

    print(f"  Stokes I shape: {stokes_i.shape}  (freq x time)")
    print(f"  Freq: {freq_mhz[0]:.1f} - {freq_mhz[-1]:.1f} MHz")
    print(f"  Time: {time_ms[0]:.3f} - {time_ms[-1]:.3f} ms")

    # ── Scrunching ──
    if args.fscrunch and args.fscrunch > 1:
        n_orig = freq_mhz.size
        # fscrunch: average along freq axis (axis 0)
        n_new = stokes_i.shape[0] // args.fscrunch
        stokes_i = np.nanmean(stokes_i[: n_new * args.fscrunch].reshape(n_new, args.fscrunch, -1), axis=1)
        if stokes_q is not None:
            stokes_q = np.nanmean(stokes_q[: n_new * args.fscrunch].reshape(n_new, args.fscrunch, -1), axis=1)
        if stokes_u is not None:
            stokes_u = np.nanmean(stokes_u[: n_new * args.fscrunch].reshape(n_new, args.fscrunch, -1), axis=1)
        if stokes_v is not None:
            stokes_v = np.nanmean(stokes_v[: n_new * args.fscrunch].reshape(n_new, args.fscrunch, -1), axis=1)
        freq_mhz = np.nanmean(freq_mhz[: n_new * args.fscrunch].reshape(n_new, args.fscrunch), axis=1)
        print(f"  Freq scrunch {n_orig} -> {freq_mhz.size}")

    if args.tscrunch and args.tscrunch > 1:
        n_orig = stokes_i.shape[1]
        n_new = n_orig // args.tscrunch
        stokes_i = np.nanmean(stokes_i[:, : n_new * args.tscrunch].reshape(stokes_i.shape[0], n_new, args.tscrunch), axis=2)
        if stokes_q is not None:
            stokes_q = np.nanmean(stokes_q[:, : n_new * args.tscrunch].reshape(stokes_q.shape[0], n_new, args.tscrunch), axis=2)
        if stokes_u is not None:
            stokes_u = np.nanmean(stokes_u[:, : n_new * args.tscrunch].reshape(stokes_u.shape[0], n_new, args.tscrunch), axis=2)
        if stokes_v is not None:
            stokes_v = np.nanmean(stokes_v[:, : n_new * args.tscrunch].reshape(stokes_v.shape[0], n_new, args.tscrunch), axis=2)
        time_ms = np.nanmean(time_ms[: n_new * args.tscrunch].reshape(n_new, args.tscrunch), axis=1)
        print(f"  Time scrunch {n_orig} -> {time_ms.size}")
        if args.peak_indices is not None:
            orig_indices = list(args.peak_indices)
            args.peak_indices = _rescale_peak_indices(orig_indices, args.tscrunch)
            print(f"  Peak indices scaled {orig_indices} -> {args.peak_indices} (scrunched axis)")

    # Match frbop DMOptimiser: zero NaNs after scrunching
    for arr in (stokes_i, stokes_q, stokes_u, stokes_v):
        if arr is not None:
            arr[np.isnan(arr)] = 0.0

    # ── DM grid ──
    dt_ms = float(np.median(np.diff(time_ms)))
    reference_freq = float(args.ref_freq) if args.ref_freq is not None else float(np.max(freq_mhz))
    delta_inv_f2 = np.abs(1.0 / freq_mhz**2 - 1.0 / reference_freq**2)
    max_delta_inv_f2 = float(np.max(delta_inv_f2))
    recommended_step = (dt_ms) / (_DM_CONSTANT * max_delta_inv_f2) if max_delta_inv_f2 > 0 else 0.01

    if args.dm_min is not None and args.dm_max is not None:
        dm_range = (args.dm_min, args.dm_max)
    elif args.dm_guess is not None:
        span = 50.0
        dm_range = (args.dm_guess - span, args.dm_guess + span)
    else:
        dm_range = (300.0, 400.0)

    if args.dm_step is not None:
        dm_step = args.dm_step
    else:
        dm_step = recommended_step

    dm_values = _build_dm_values(dm_range, n_points=100, dm_step=dm_step)
    print(f"  DM range: {dm_range[0]:.2f} - {dm_range[1]:.2f} pc cm^-3")
    print(f"  DM grid: {len(dm_values)} points, step ~{dm_step:.6g} pc cm^-3")
    print(f"  Recommended step: {recommended_step:.6g} pc cm^-3")
    print(f"  Reference frequency: {reference_freq:.3f} MHz")

    # ── Peak selection ──
    print("\nPeak selection...")
    if args.peak_indices is not None:
        peak_regions = _parse_peak_index_pairs(args.peak_indices, stokes_i.shape[1])
        print(f"  Using {len(peak_regions)} manually specified peak region(s)")
    elif args.manual_peaks:
        profile = np.nansum(stokes_i, axis=0)
        peak_regions = _select_peaks_manual(time_ms, profile)
        print(f"  Selected {len(peak_regions)} peak region(s) interactively")
    elif args.separate_peaks:
        # Simple auto-detection: find contiguous on-pulse regions
        profile = np.nansum(stokes_i, axis=0)
        median_val = np.median(profile)
        std_val = np.std(profile[:max(1, len(profile) // 10)])
        threshold = median_val + 3.0 * std_val
        on_pulse = profile > threshold
        regions = []
        in_region = False
        for i in range(len(on_pulse)):
            if on_pulse[i] and not in_region:
                start = i
                in_region = True
            elif not on_pulse[i] and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(on_pulse)))
        peak_regions = regions if regions else [(0, stokes_i.shape[1])]
        print(f"  Auto-detected {len(peak_regions)} peak region(s)")
    else:
        peak_regions = [(0, stokes_i.shape[1])]
        print("  Using full dataset (no peak separation)")

    for i, (s, e) in enumerate(peak_regions):
        e_disp = min(e - 1, len(time_ms) - 1)
        print(f"    Peak {i+1}: time indices {s}-{e} "
              f"({time_ms[s]:.3f} - {time_ms[e_disp]:.3f} ms)")

    # ── Run structure maximisation per peak ──
    print(f"\nFilter: order={args.order}, kc={'auto' if args.kc is None else args.kc}")
    print("=" * 70)

    dedisperser = _Dedisperser(
        stokes_i, freq_mhz, time_ms,
        input_dm=args.input_dm,
        reference_freq=reference_freq,
        seed=args.seed,
    )

    for i, (start, end) in enumerate(peak_regions):
        tag = f"peak{i+1}" if len(peak_regions) > 1 else ""
        print(f"\n── Peak {i+1} ──")

        # Crop the dspec to this peak first, then dedisperse — cropping
        # I(DM, t) instead would let each peak sample the other peak's
        # swept signal at large |delta-DM|
        si_peak = stokes_i[:, start:end]

        print("  Dedispersing over DM grid...")
        output_size = dedisperser.max_output_size_for_dm_range(si_peak, dm_range)
        i_dm_t = np.zeros((len(dm_values), output_size), dtype=float)
        for idx, dm in enumerate(dm_values):
            dedisp = dedisperser.dedisperse(
                si_peak, dm, output_size=output_size, mode=args.dedisp_mode,
            )
            i_dm_t[idx] = np.nansum(dedisp, axis=0)
        print(f"  I(DM,t) shape: {i_dm_t.shape}")
        #plt.imshow(i_dm_t, aspect="auto", extent=(0, i_dm_t.shape[1], dm_values[-1], dm_values[0]))
        #plt.xlabel("Time (ms)")
        #plt.ylabel("DM (pc cm^-3)")
        #plt.title("Dedispersed Intensity")
        #plt.show()

        try:
            result = structure_param(i_dm_t, order=args.order, kc=args.kc)
        except RuntimeError as e:
            raise SystemExit(
                f"Error: {e}. The DCT spectrum has no detectable noise floor; "
                "pass --kc to force a cutoff") from e

        dm_best = recovered_dm(dm_values, result["sp"])
        low_dm, high_dm = dm_range_from_relative_uncertainty(
            result["sp"], result["relative_uncertainty"], dm_values
        )
        unc_minus = None if low_dm is None else dm_best - low_dm
        unc_plus = None if high_dm is None else high_dm - dm_best

        def _fmt(dm, minus, plus):
            s = f"{dm:.6f}"
            if minus is not None and plus is not None:
                return f"{s} (-{minus:.6f}/+{plus:.6f})"
            return s + " (-?/+)"

        print(f"  DM = {_fmt(dm_best, unc_minus, unc_plus)} pc cm^-3")
        print(f"  SP = {result['sp'][int(np.argmax(result['sp']))]:.6f}")
        print(f"  kc = {result['kc']}, order = {result['order']}")

        # Diagnostic plots (identical recipe/filenames to maximise_structure.py,
        # so figures are directly comparable to frbop shrine_logs output)
        CI_data = dct(i_dm_t, norm="ortho")
        max_index = int(np.argmax(result["sp"]))
        delta_I = result["delta_I"]
        delta_delta_I = delta_I - delta_I[max_index]
        adjusted_SPs = result["sp"] + result["sp"] * result["relative_uncertainty"]

        prefix = f"{args.label}_{tag}" if tag else args.label
        dt_us = max(1, int(round(dt_ms * 1000)))

        plot_DCT_spectrum(CI_data, prefix, dt_us, result["kc"])

        if args.save_all:
            plot_DM_index(dm_values, prefix, dt_us)
            plot_I_at_max(i_dm_t[max_index], result["i_smooth"][max_index], prefix, dt_us)
            plot_noise_at_max(delta_I[max_index], prefix, dt_us)
            plot_noisy_I_DM_t(i_dm_t, dm_values, prefix, dt_us)
            plot_smooth_I_DM_t(result["i_smooth"], dm_values, prefix, dt_us)
            plot_detrended_noise(delta_I, dm_values, prefix, dt_us)
            plot_relative_detrended_noise(delta_delta_I, dm_values, prefix, dt_us)
            plot_uncertainty(result["uncertainty"], dm_values, prefix, dt_us)
            plot_relative_uncertainty(result["relative_uncertainty"], dm_values, prefix, dt_us)
            plot_compare_uncertainty(
                result["uncertainty"], result["relative_uncertainty"],
                dm_values, prefix, dt_us,
            )
            plot_SP(result["sp"], dm_values, prefix, dt_us)
            plot_adjusted_SP(adjusted_SPs, dm_values, prefix, dt_us)
            plot_norm_CI_f_adjusted_SP(result["sp"], adjusted_SPs, dm_values, prefix, dt_us)

        # Save data files
        np.savetxt(f"{prefix}_SPs.dat", result["sp"])
        np.savetxt(f"{prefix}_Uncertainties.dat", result["uncertainty"])
        np.savetxt(f"{prefix}_Relative_Uncertainties.dat", result["relative_uncertainty"])

        # Save I(DM, t) inputs (same convention as frbop shrine_logs runs)
        np.save(f"{prefix}_DMs.npy", dm_values)
        np.save(f"{prefix}_I_{dt_us}us.npy", i_dm_t)

        # Summary file
        with open(f"{prefix}_structure_summaryfile.txt", "w") as f:
            f.write(f"//begin maximise_structure summary//\n/*\n")
            f.write(f"FRB Label: {args.label}\n")
            f.write(f"Filter order: {result['order']}\n")
            f.write(f"kc: {result['kc']} (auto={args.kc is None})\n")
            f.write(f"DM range: {dm_range[0]} - {dm_range[1]}\n")
            f.write(f"Input DM: {args.input_dm}\n")
            f.write(f"Reference frequency: {reference_freq}\n")
            f.write(f"Dedispersion mode: {args.dedisp_mode}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Structure-max DM: {dm_best}\n")
            f.write(f"Uncertainty: {_fmt(dm_best, unc_minus, unc_plus)}\n")
            f.write(f"*/\n//end summary//\n")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
