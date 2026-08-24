#!/usr/bin/env python3
"""Filter design & order sweep test suite for SHRINE structure maximisation.

Sweeps Butterworth order (O) and spectral cutoff (kc) over a 2-D grid.

Two modes, with deliberately different diagnostics:

* **real** — one observation, components cropped via peak indices. Per-
  component DM_SHRINE / stability maps plus the differential-DM result:
  DM2 - DM1, its formal uncertainty sigma_dDM = sqrt(s1^2+s2^2), and the
  signed significance (DM2-DM1)/sigma_dDM across the (O, kc) grid.
* **synthetic** — operates exactly like real mode on a user-supplied two-
  component dynamic spectrum with known component DMs (--true-dms DM1 DM2;
  --input-dm = prep correction of the sim, usually 0 for raw sims). Noise
  realisations are added and recovery is judged against the KNOWN truth:
  accuracy, bias, error calibration (empirical/formal scatter ratio and
  normalised errors z), and detection sensitivity incl. the false-positive
  rate. Different injections come from running different sim configs.

Usage examples
--------------
# Two-component differential simulation (known truths via --true-dms;
# --input-dm = prep correction of the sim, usually 0 for raw sims)
python test_filter_sweep.py --mode synthetic \
    --stokes-i ds.npy --freq freq.npy --time time.npy \
    --true-dms 335.5 336.2 --input-dm 0 \
    --peak-indices 321 482 491 880 \
    --dm-min 334 --dm-max 338 --target-snr 300 -n 50

# Real data sweep with differential output (>= 2 peaks required)
python test_filter_sweep.py --mode real \
    --stokes-i ds.npy --freq freq.npy --time time.npy \
    --input-dm 335.87 --peak-indices 321 482 491 880 \
    --dm-min 334.87 --dm-max 336.87 --dm-step 0.004

# Custom filter grid (both modes)
python test_filter_sweep.py ... -o 1 --o-max 20 --o-step 2 \
    --kc-min 10 --kc-max 100 --kc-step 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time as _time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from struct_max import _build_dm_values, _Dedisperser
from struct_max_core import (dm_range_from_relative_uncertainty, get_kc,
                             lowpass_response, recovered_dm, structure_param,
                             structure_param_from_dct)

# ──────────────────────────────────────────────────────────────────────
# Small shared utilities
# ──────────────────────────────────────────────────────────────────────

_DM_CONSTANT = 4.148808e6
_GAUSS_TAIL_3SIG = 2 * (1 - 0.9986501019683699)  # |z|>3 two-sided rate


def _save_fig(output_dir: Path, name: str, ext: str, bbox_inches=None):
    path = output_dir / f"{name}.{ext}"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches=bbox_inches)
    plt.close()
    print(f"Saved {path}")


def _robust_rms(x):
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return 1.4826 * mad if mad > 0 else float(np.nanstd(x))


def rows_to_grid(rows, key, orders=None, kcs=None):
    """Scatter row dicts into a 2-D (order, kc) grid.

    Returns (grid, orders, kcs); orders/kcs inferred from the rows when not
    given. Missing cells stay NaN."""
    if orders is None:
        orders = sorted(set(r["order"] for r in rows))
    if kcs is None:
        kcs = sorted(set(r["kc"] for r in rows))
    grid = np.full((len(orders), len(kcs)), np.nan)
    for r in rows:
        v = r.get(key)
        if v is not None:
            grid[orders.index(r["order"]), kcs.index(r["kc"])] = v
    return grid, orders, kcs


def draw_heatmap(ax, grid, kcs, orders, cmap, vmin, vmax, clabel=None,
                 title=None, xlabel="kc", ylabel="Order O"):
    """imshow a (order, kc) grid with colourbar; returns True if drawn."""
    if not np.any(np.isfinite(grid)):
        if title:
            ax.set_title(title)
        return False
    kc_step = kcs[1] - kcs[0] if len(kcs) > 1 else 1
    extent = [kcs[0] - 0.5 * kc_step, kcs[-1] + 0.5 * kc_step,
              orders[0] - 0.5, orders[-1] + 0.5]
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label=clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return True


def annotate_cells(ax, kcs, orders, values, fmt, vmin, vmax, mode="seq",
                   fontsize=4.5):
    """Overlay per-cell values on a heatmap when the grid is small.

    mode: "seq"      — sequential cmap (white text at high end)
          "centred"  — diverging cmap centred between vmin/vmax
                       (white near both extremes)
          "extremes" — white near the data's own min/max
    """
    values = np.asarray(values)
    if values.size > 300 or max(values.shape) > 40:
        return
    scale = max(float(vmax) - float(vmin), 1e-12)
    fin = values[np.isfinite(values)]
    dmin, dmax = (float(np.min(fin)), float(np.max(fin))) if fin.size else (0, 0)
    span = max(dmax - dmin, 1e-12)
    for oi in range(values.shape[0]):
        for ki in range(values.shape[1]):
            v = values[oi, ki]
            if not np.isfinite(v):
                continue
            frac = (float(v) - vmin) / scale
            if mode == "centred":
                white = abs(frac - 0.5) > 0.32
            elif mode == "extremes":
                white = (abs(float(v) - dmin) < 0.15 * span or
                         abs(float(v) - dmax) < 0.15 * span)
            else:
                white = frac > 0.62
            ax.text(kcs[ki], orders[oi], fmt(float(v)),
                    ha="center", va="center", fontsize=fontsize,
                    color="w" if white else "k")


def _progress(done, total, msg):
    sys.stderr.write(f"\r  [{done}/{total}] {msg}   ")
    sys.stderr.flush()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SHRINE filter order / kc sweep test suite",
    )

    # Sweep grid
    g = p.add_argument_group("Sweep grid")
    g.add_argument("-o", "--o-min", type=int, default=1, help="Min filter order (default: 1)")
    g.add_argument("--o-max", type=int, default=20, help="Max filter order (default: 20)")
    g.add_argument("--o-step", type=int, default=1, help="Filter order step (default: 1)")
    g.add_argument("--kc-min", type=int, default=10, help="Min kc (default: 10)")
    g.add_argument("--kc-max", type=int, default=100, help="Max kc (default: 100)")
    g.add_argument("--kc-step", type=int, default=5, help="kc step (default: 5)")

    # Mode
    m = p.add_argument_group("Mode")
    m.add_argument("--mode", choices=["synthetic", "real", "both"], default="synthetic",
                   help="synthetic = two-component differential-DM tests, "
                        "real = single file, both = both")

    # Synthetic parameters (two-component differential-DM simulations)
    s = p.add_argument_group("Synthetic data (two-component differential-DM sims)")
    s.add_argument("--true-dms", nargs=2, type=float, default=None,
                   help="Known true DMs of the two simulated components: "
                        "--true-dms DM1 DM2 [pc cm^-3]. Used as the reference "
                        "truth for recovery, bias and z statistics.")
    s.add_argument("--n-realisations", "-n", type=int, default=50,
                   help="Noise realisations per (O, kc) in synthetic mode")
    s.add_argument("--target-snr", type=float, default=None,
                   help="FIRES-style analytic noise: derive the SEFD from the "
                        "clean dynamic spectrum (on-pulse boxcar method) and "
                        "add per-channel white noise sigma = SEFD / "
                        "sqrt(n_pol * df * dt) so the on-pulse S/N matches "
                        "this value. Takes precedence over --noise-level.")
    s.add_argument("--buffer-frac", type=float, default=1.0,
                   help="On/off-pulse buffer in units of the pulse width "
                        "(FIRES buffer_fraction; default 1.0)")
    s.add_argument("--noise-level", type=float, default=None,
                   help="Fallback simple noise: RMS relative to data std. "
                        "Ignored when --target-snr is given")
    s.add_argument("--seed", type=int, default=42, help="Base random seed")

    # Real data
    r = p.add_argument_group("Real data")
    r.add_argument("--stokes-i", default=None, help="Stokes I .npy (freq x time)")
    r.add_argument("--stokes-cube", default=None, help="Stokes cube (.npy or .npz)")
    r.add_argument("--freq", default="freq.npy", help="Frequency array [MHz]")
    r.add_argument("--time", default="time.npy", help="Time array [ms]")
    r.add_argument("--dm-min", type=float, default=None, help="DM min for real data")
    r.add_argument("--dm-max", type=float, default=None, help="DM max for real data")
    r.add_argument("--dm-step", type=float, default=None, help="DM grid step (overrides auto)")
    r.add_argument("--ndm", type=int, default=None, help="Number of DM grid points (overrides auto)")
    r.add_argument("--input-dm", type=float, default=0.0,
                   help="DM previously used to correct the full time series "
                        "[pc cm^-3]. Recovered DMs are only absolute if this "
                        "matches the correction actually applied to the data.")
    r.add_argument("--ref-freq", type=float, default=None, help="Reference freq [MHz], default is max(freq)")
    r.add_argument("--true-dm", type=float, default=None,
                   help="Known true DM for real data (for bias computation)")
    r.add_argument("--dedisp-mode", choices=["expand", "expand_zero", "expand_nan", "crop"],
                   default="expand_nan",
                   help="Dedispersion edge mode (default: expand; frbop run used expand_nan)")

    # Peak selection
    pk = p.add_argument_group("Peak selection")
    pk.add_argument("--peak-indices", nargs="*", type=int, default=None,
                    help="Peak index pairs: start1 end1 [start2 end2 ...]")
    pk.add_argument("--manual-peaks", action="store_true",
                    help="Interactive peak selection via matplotlib clicks")
    pk.add_argument("--separate-peaks", action="store_true",
                    help="Auto-detect separate peaks (Gaussian smoothing + find_peaks)")

    # Top-hat comparison
    t = p.add_argument_group("Top-hat comparison")
    t.add_argument("--check-tophat", action="store_true",
                   help="Plot filter responses O=5,10,15,20 vs ideal top-hat")

    # Output
    o = p.add_argument_group("Output")
    o.add_argument("--output-dir", "--out", default="SHRINE_test",
                   help="Output directory")
    o.add_argument("--ext", default="png", help="Plot extension")
    o.add_argument("--verbose", action="store_true", help="Print per-trial results")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Shared data-loading helpers (same path for synthetic and real)
# ──────────────────────────────────────────────────────────────────────


def _load_stokes_i(args):
    """Return (stokes_i, freq_mhz, time_ms) from CLI args."""
    # ── Real data ──
    derived_freq = derived_time = None

    if args.stokes_cube:
        from struct_max import (_derive_axes_from_metadata,
                                _extract_stokes_components,
                                _load_stokes_cube_from_npz)
        cube_path = Path(args.stokes_cube)
        if cube_path.suffix.lower() == ".npz":
            cube, meta = _load_stokes_cube_from_npz(cube_path)
            si, _, _, _, _, _ = _extract_stokes_components(
                cube, pol_order=meta.get("pol_order"),
                nchan=meta.get("nchan"), nsamp=meta.get("nsamp"),
            )
            derived_freq, derived_time = _derive_axes_from_metadata(
                meta, si.shape[0], si.shape[1],
            )
        else:
            cube = np.load(cube_path)
            if cube.ndim == 3:
                si, _, _, _, _, _ = _extract_stokes_components(cube)
            elif cube.ndim == 2:
                si = cube
            else:
                raise SystemExit(f"Error: unrecognised cube shape {cube.shape}")
    else:
        if not args.stokes_i:
            raise SystemExit("Error: provide --stokes-i or --stokes-cube")
        si = np.load(args.stokes_i)
        if si.ndim == 3:
            from struct_max import _extract_stokes_components
            si, _, _, _, _, pol = _extract_stokes_components(si)
            print(f"  Extracted Stokes I from 3D array (pol_order={pol!r})")

    freq_mhz = None
    fp = Path(args.freq)
    if fp.exists():
        freq_mhz = np.load(fp)
    if freq_mhz is None:
        freq_mhz = derived_freq
    if freq_mhz is None:
        raise SystemExit("Error: frequency axis required")

    time_ms = None
    tp = Path(args.time)
    if tp.exists():
        time_ms = np.load(tp)
    if time_ms is None:
        time_ms = derived_time
    if time_ms is None:
        raise SystemExit("Error: time axis required")

    if not np.all(np.diff(freq_mhz) >= 0):
        order = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[order]
        si = si[order]

    return si, freq_mhz, time_ms


def _build_dm_grid(freq_mhz, time_ms, dm_min, dm_max, ref_freq=None,
                   dm_step=None, ndm=None):
    """Build a DM grid from freq/time arrays and a DM range.

    Mirrors frbop's production path (dm_optimisation.py): default step is
    the recommended lowest step (~1 time sample across band) and the grid
    is arange-with-endpoint-append via MetricsMixin._build_dm_values.
    --ndm still allows an explicit linspace override.
    """
    if ndm is not None:
        return np.linspace(dm_min, dm_max, ndm)

    dt_ms = float(np.median(np.diff(time_ms)))
    if ref_freq is None:
        ref_freq = float(np.max(freq_mhz))
    delta_inv_f2 = np.abs(1.0 / freq_mhz**2 - 1.0 / ref_freq**2)
    max_d = float(np.max(delta_inv_f2))
    recommended_step = dt_ms / (_DM_CONSTANT * max_d) if max_d > 0 else 0.01

    return _build_dm_values((dm_min, dm_max), n_points=100,
                            dm_step=dm_step if dm_step is not None else recommended_step)


# ──────────────────────────────────────────────────────────────────────
# Peak selection  (ported from frbop/utils/peaks.py)
# ──────────────────────────────────────────────────────────────────────

def _parse_peak_index_pairs(peak_indices, n_time):
    """Normalise a flat list of indices into clipped (start, end) regions."""
    if peak_indices is None:
        return []
    values = list(peak_indices)
    if len(values) == 0:
        raise ValueError("--peak-indices requires at least one pair")
    if len(values) % 2 != 0:
        raise ValueError("--peak-indices requires an even number of values")
    if n_time <= 0:
        raise ValueError("Cannot normalise peak indices for an empty time axis")

    regions = []
    for i in range(0, len(values), 2):
        start = int(np.clip(min(values[i], values[i + 1]), 0, n_time - 1))
        stop = int(np.clip(max(values[i], values[i + 1]), 0, n_time))
        if stop <= start:
            stop = min(n_time, start + 1)
        regions.append((start, stop))
    return regions


def _select_peaks_manual(time_ms, stokes_i):
    """Interactive peak selection via matplotlib clicks (with dspec overlay)."""
    profile = np.nansum(stokes_i, axis=0)
    freq_mhz = None

    fig, (ax, ax_spec) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0},
    )
    ax.plot(time_ms, profile, color="k", linewidth=1)
    ax.set_title("Click start/end bounds for each peak (close window to finish)")
    ax.set_ylabel("S [arb.]")
    ax.grid(True, alpha=0.3)

    extent = [time_ms[0], time_ms[-1], 0, stokes_i.shape[0]]
    ax_spec.imshow(stokes_i, aspect="auto", extent=extent, origin="lower", cmap="plasma")
    ax_spec.set_xlabel("Time [ms]")
    ax_spec.set_ylabel("Frequency channel")
    ax_spec.grid(True, alpha=0.3)

    cursor_line = ax.axvline(time_ms[0], color="tab:blue", alpha=0.4, linewidth=1)
    cursor_spec = ax_spec.axvline(time_ms[0], color="tab:blue", alpha=0.4, linewidth=1)
    clicks: List[float] = []

    def on_move(event):
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        cursor_line.set_xdata([x, x])
        cursor_spec.set_xdata([x, x])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        clicks.append(x)
        for a in (ax, ax_spec):
            a.axvline(x, color="tab:red", alpha=0.7, linewidth=1)
        if len(clicks) % 2 == 0:
            lo, hi = sorted((clicks[-2], clicks[-1]))
            for a in (ax, ax_spec):
                a.axvspan(lo, hi, color="tab:orange", alpha=0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if not clicks:
        return [(0, stokes_i.shape[1])]
    if len(clicks) % 2 != 0:
        clicks = clicks[:-1]

    regions = []
    for i in range(0, len(clicks), 2):
        lo_t, hi_t = sorted((clicks[i], clicks[i + 1]))
        s = int(np.argmin(np.abs(time_ms - lo_t)))
        e = int(np.argmin(np.abs(time_ms - hi_t)))
        s, e = min(s, e), max(s, e) + 1
        e = min(time_ms.size, e)
        if e <= s:
            e = min(time_ms.size, s + 1)
        regions.append((s, e))
    return regions if regions else [(0, stokes_i.shape[1])]


def _separate_peaks(time_ms, stokes_i, min_separation_ms=5.0):
    """Auto-detect separate peaks via Gaussian smoothing + scipy find_peaks."""
    profile = np.nansum(stokes_i, axis=0)
    smoothed = gaussian_filter1d(profile, sigma=4)

    dt = float(np.median(np.diff(time_ms)))
    min_distance = max(1, int(min_separation_ms / dt))
    n_edge = max(1, int(0.05 * len(smoothed)))

    peaks, _ = find_peaks(
        smoothed, distance=min_distance,
        prominence=2 * np.std(smoothed[:n_edge]),
    )

    if len(peaks) == 0:
        return [(0, stokes_i.shape[1])]

    peak_regions = []
    for peak in peaks:
        half_max = (smoothed[peak] - np.min(smoothed)) / 2 + np.min(smoothed)
        start = peak
        while start > 0 and smoothed[start] > half_max:
            start -= 1
        end = peak
        while end < len(smoothed) - 1 and smoothed[end] > half_max:
            end += 1
        peak_regions.append((max(0, start - 20), min(stokes_i.shape[1], end + 80)))

    return peak_regions


def _resolve_peak_regions(args, time_ms, stokes_i):
    """Dispatch peak selection based on CLI flags, return list of (start, end)."""
    if args.peak_indices is not None:
        regions = _parse_peak_index_pairs(args.peak_indices, stokes_i.shape[1])
        print(f"  Using {len(regions)} manually specified peak region(s)")
    elif args.manual_peaks:
        regions = _select_peaks_manual(time_ms, stokes_i)
        print(f"  Selected {len(regions)} peak region(s) interactively")
    elif args.separate_peaks:
        regions = _separate_peaks(time_ms, stokes_i)
        print(f"  Auto-detected {len(regions)} peak region(s)")
    else:
        regions = [(0, stokes_i.shape[1])]
        print("  Using full dataset (no peak separation)")

    for i, (s, e) in enumerate(regions):
        e_disp = min(e - 1, len(time_ms) - 1)
        print(f"    Peak {i+1}: time indices {s}-{e} "
              f"({time_ms[s]:.3f} - {time_ms[e_disp]:.3f} ms)")
    return regions


# ──────────────────────────────────────────────────────────────────────
# Sweep logic
# ──────────────────────────────────────────────────────────────────────

def _eval_one(dm_values, order, kc, CI_data):
    """Run structure_param for one (O, kc) and return metrics.

    If *CI_data* (pre-computed DCT of i_dm_t) is provided, uses the fast
    :func:`structure_param_from_dct` path to avoid redundant DCTs.
    """
    result = structure_param_from_dct(CI_data, kc=kc, order=order)
    result["dm_data"] = dm_values
    dm_best = recovered_dm(result["dm_data"], result["sp"])
    low_dm, high_dm = dm_range_from_relative_uncertainty(
        result["sp"], result["relative_uncertainty"], result["dm_data"]
    )
    unc_minus = 0.0 if low_dm is None else dm_best - low_dm
    unc_plus = 0.0 if high_dm is None else high_dm - dm_best
    formal_unc = 0.5 * (unc_minus + unc_plus)
    return dm_best, formal_unc, unc_minus, unc_plus


def _sweep_grids(args):
    """Order and kc axes of the sweep, from CLI bounds."""
    orders = list(range(args.o_min, args.o_max + 1, args.o_step))
    kc_values = list(range(args.kc_min, args.kc_max + 1, args.kc_step))
    return orders, kc_values


def _prepare_component(args, stokes_i, freq_mhz, time_ms, window,
                       dedisperser):
    """Dedisperse one peak window over the DM grid and DCT it (done once per
    component; the O/kc sweep then runs on the coefficients).

    Returns (dm_values, CI_data)."""
    s, e = window
    tm_peak = time_ms[s:e]
    dm_values = _build_dm_grid(freq_mhz, tm_peak, args.dm_min, args.dm_max,
                               ref_freq=args.ref_freq,
                               dm_step=args.dm_step, ndm=args.ndm)
    output_size = dedisperser.max_output_size_for_dm_range(
        stokes_i[:, s:e], (args.dm_min, args.dm_max))
    i_dm_t = np.zeros((len(dm_values), output_size), dtype=float)
    for idx, dm in enumerate(dm_values):
        dedisp = dedisperser.dedisperse(
            stokes_i[:, s:e], dm, output_size=output_size,
            mode=args.dedisp_mode)
        i_dm_t[idx] = np.nansum(dedisp, axis=0)
    print(f"  I(DM,t) shape: {i_dm_t.shape}  "
          f"(DM grid {len(dm_values)} pts [{dm_values[0]:.3g}, "
          f"{dm_values[-1]:.3g}], out {output_size})")
    CI_data = dct(i_dm_t, norm="ortho")
    return dm_values, CI_data


# ──────────────────────────────────────────────────────────────────────
# Synthetic sweep — two-component differential-DM simulations
# ──────────────────────────────────────────────────────────────────────

def _boxcar_window(profile, frac=0.95):
    """Contiguous window around the brightest bin containing *frac* of the
    total profile energy (port of FIRES' boxcar_width). Returns (left, right)
    inclusive bin indices."""
    prof = np.where(np.isfinite(profile), profile, 0.0)
    peak = int(np.argmax(prof))
    total = float(prof.sum())
    if total <= 0:
        return 0, len(prof) - 1
    lo = hi = peak
    captured = float(prof[peak])
    while captured < frac * total and (lo > 0 or hi < len(prof) - 1):
        lv = prof[lo - 1] if lo > 0 else -np.inf
        rv = prof[hi + 1] if hi < len(prof) - 1 else -np.inf
        if lv >= rv:
            lo -= 1
            captured += float(lv)
        else:
            hi += 1
            captured += float(rv)
    return lo, hi


def _onpulse_snr(stokes_i, left, right, buffer_bins):
    """Measured on-pulse S/N of a noisy Stokes-I dynamic spectrum: energy in
    the on-pulse boxcar divided by robust off-pulse RMS (FIRES snr_onpulse).
    The buffer only excludes off-pulse bins near the pulse."""
    profile = np.nansum(stokes_i, axis=0)
    lo = max(left - buffer_bins, 0)
    hi = min(right + buffer_bins, profile.size - 1)
    off_mask = np.ones(profile.size, dtype=bool)
    off_mask[lo:hi + 1] = False
    on = profile[left:right + 1]
    n_on = on.size
    off = profile[off_mask]
    if off.size == 0 or n_on <= 0:
        return np.nan
    sigma = _robust_rms(off)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(np.nansum(on) / (sigma * np.sqrt(n_on)))


def _fires_noise_setup(stokes_i, freq_mhz, time_ms, target_snr,
                       buffer_frac=1.0, n_pol=2):
    """Analytic SEFD derivation (port of FIRES compute_required_sefd).

    Finds the 95%-energy boxcar of the Stokes-I pulse profile, then solves
        sefd = F_on * sqrt(n_pol * df * dt) / (SNR * sqrt(N_chan * N_on))
    so that adding white noise with sigma_ch = sefd / sqrt(n_pol * df * dt)
    yields on-pulse S/N ~= target_snr.

    Returns (sigma_ch, info) with sigma_ch the per-channel noise std.
    """
    profile = np.nansum(stokes_i, axis=0)
    left, right = _boxcar_window(profile, frac=0.95)
    n_chan = stokes_i.shape[0]
    n_on = right - left + 1
    F_on = float(np.nansum(profile[left:right + 1]))
    if F_on <= 0 or n_on <= 0:
        raise SystemExit("Error: insufficient on-pulse energy for SEFD "
                         "estimation (--target-snr)")
    ch_bw_hz = float(np.median(np.abs(np.diff(freq_mhz)))) * 1e6
    t_res_s = float(np.median(np.abs(np.diff(time_ms)))) / 1000.0
    buffer_bins = max(int(float(buffer_frac) * n_on), 0)
    sefd = F_on * np.sqrt(n_pol * ch_bw_hz * t_res_s) \
        / (target_snr * np.sqrt(n_chan * n_on))
    sigma_ch = sefd / np.sqrt(n_pol * np.full(n_chan, ch_bw_hz) * t_res_s)
    info = {"left": left, "right": right, "n_on": n_on, "F_on": F_on,
            "ch_bw_hz": ch_bw_hz, "t_res_s": t_res_s, "sefd": sefd,
            "buffer_bins": buffer_bins}
    return sigma_ch, info


def run_synthetic_differential(args, output_dir: Path):
    """Two-component simulation operating exactly like the real pipeline.

    The user supplies a NOISELESS dynamic spectrum (e.g. FIRES output) with
    two components whose true DMs are known, crops them with peak indices,
    and passes the truths via --true-dms DM1 DM2. --input-dm is the DM at
    which the simulated time series was pre-corrected (0 for raw sims).
    Every run adds a fresh noise realisation; both windows are dedispersed
    and evaluated over the same (O, kc) grid as real mode, and recovery is
    judged against the true component DMs and their difference
    dDM_inj = DM2 - DM1. Different injections come from running different
    FIRES configs, not from this script.

    Returns (rows, meta): rows carry per-(O, kc) aggregates; meta holds the
    pooled z scores and false-positive count for the plots.
    """
    orders, kc_values = _sweep_grids(args)
    n_reals = args.n_realisations

    print(f"\n{'=' * 70}")
    print(f"SYNTHETIC DIFFERENTIAL SWEEP: {n_reals} realisations x "
          f"{len(orders)} O x {len(kc_values)} kc")
    print(f"{'=' * 70}")

    stokes_i, freq_mhz, time_ms = _load_stokes_i(args)
    print(f"  Shape: {stokes_i.shape}  "
          f"Freq: {freq_mhz[0]:.1f}-{freq_mhz[-1]:.1f} MHz")

    peak_regions = _resolve_peak_regions(args, time_ms, stokes_i)
    if len(peak_regions) < 2:
        raise SystemExit("Error: synthetic differential sims require >= 2 "
                         "peak regions (use --peak-indices)")
    if args.true_dms is None or len(args.true_dms) != 2:
        raise SystemExit("Error: --true-dms DM1 DM2 required for synthetic "
                         "mode (known component DMs)")
    T1, T2 = float(args.true_dms[0]), float(args.true_dms[1])
    dd_inj = T2 - T1
    input_dm = float(args.input_dm)

    if args.dm_min is None or args.dm_max is None:
        raise SystemExit("Error: --dm-min and --dm-max required for "
                         "synthetic mode")

    # Static per-component setup: DM grids, dedispersers and output sizes
    # are built once; each realisation re-dedisperses its own noisy cube
    # through them. One shared input DM (the sim's prep correction),
    # exactly like the real pipeline.
    comps = []
    for pi in range(2):
        s, e = peak_regions[pi]
        tm_peak = time_ms[s:e]
        dm_values = _build_dm_grid(freq_mhz, tm_peak, args.dm_min,
                                   args.dm_max, ref_freq=args.ref_freq,
                                   dm_step=args.dm_step, ndm=args.ndm)
        dedisperser = _Dedisperser(
            stokes_i, freq_mhz, time_ms,
            input_dm=input_dm,
            reference_freq=args.ref_freq,
            seed=args.seed,
        )
        out_size = dedisperser.max_output_size_for_dm_range(
            stokes_i[:, s:e], (args.dm_min, args.dm_max))
        comps.append({"window": (s, e), "dms": dm_values,
                      "dedisp": dedisperser, "out_size": out_size})
        print(f"  Component {pi+1}: window {s}-{e}, true DM {[T1, T2][pi]}, "
              f"grid {len(dm_values)} pts [{dm_values[0]:.2f}, "
              f"{dm_values[-1]:.2f}]")
    print(f"  Input DM (prep correction): {input_dm}; "
          f"dDM_inj = {dd_inj:+.4g} pc cm^-3")

    # Noise model: FIRES analytic S/N matching (preferred) or simple
    # data-std scaling as fallback.
    snr_measured = []
    ninfo = None
    sigma_ch = None
    if args.target_snr is not None and args.target_snr > 0:
        sigma_ch, ninfo = _fires_noise_setup(
            stokes_i, freq_mhz, time_ms, args.target_snr,
            buffer_frac=args.buffer_frac)
        noise_sigma = None
        print(f"  Noise: FIRES analytic mode (target S/N {args.target_snr:g})")
        print(f"    on-pulse boxcar [{ninfo['left']}, {ninfo['right']}] "
              f"(N_on={ninfo['n_on']}), F_on={ninfo['F_on']:.4g}")
        print(f"    df={ninfo['ch_bw_hz']/1e6:.3f} MHz, "
              f"dt={ninfo['t_res_s']*1e3:.3f} ms -> "
              f"SEFD={ninfo['sefd']:.4g} Jy")
        print(f"    per-channel sigma = {float(np.mean(sigma_ch)):.4g} "
              f"(uniform)")
    elif args.noise_level is not None:
        sigma_ch = None
        noise_sigma = args.noise_level * float(np.std(stokes_i))
        print(f"  Noise sigma: {noise_sigma:.4g} "
              f"({args.noise_level:.3g} x data std)")
    else:
        raise SystemExit("Error: specify --target-snr (FIRES analytic noise) "
                         "or --noise-level (simple relative noise)")

    n_dedisp = n_reals * sum(len(c["dms"]) for c in comps)
    print(f"  Dedispersal calls planned: {n_dedisp}")

    from collections import defaultdict
    collect = defaultdict(lambda: {"diffs": [], "sigmas": [],
                                   "u1s": [], "u2s": [],
                                   "dm1s": [], "dm2s": []})
    z_pool = []
    fp_counts = [0, 0]

    done = 0
    for ridx in range(n_reals):
        rng = np.random.default_rng([args.seed, ridx])
        clean = np.asarray(stokes_i, dtype=float)
        if sigma_ch is not None:
            noise = rng.standard_normal(np.shape(clean)) * sigma_ch[:, None]
        else:
            noise = rng.standard_normal(np.shape(clean)) * noise_sigma
        noisy = clean + noise
        if ninfo is not None:
            snr_measured.append(_onpulse_snr(
                noisy, ninfo["left"], ninfo["right"], ninfo["buffer_bins"]))

        CIs = []
        for c in comps:
            s, e = c["window"]
            i_dm_t = np.zeros((len(c["dms"]), c["out_size"]), dtype=float)
            for idx, dm in enumerate(c["dms"]):
                dedisp = c["dedisp"].dedisperse(
                    noisy[:, s:e], dm,
                    output_size=c["out_size"], mode=args.dedisp_mode)
                i_dm_t[idx] = np.nansum(dedisp, axis=0)
            CIs.append((dct(i_dm_t, norm="ortho"), c["dms"]))

        for order in orders:
            for kc in kc_values:
                try:
                    dm1, u1, _, _ = _eval_one(CIs[0][1], order, kc, CIs[0][0])
                    dm2, u2, _, _ = _eval_one(CIs[1][1], order, kc, CIs[1][0])
                except Exception:
                    continue
                sig = float(np.hypot(u1, u2))
                diff = float(dm2 - dm1)
                cell = collect[(order, kc)]
                cell["diffs"].append(diff)
                cell["sigmas"].append(sig)
                cell["u1s"].append(float(u1))
                cell["u2s"].append(float(u2))
                cell["dm1s"].append(float(dm1))
                cell["dm2s"].append(float(dm2))
                z_pool.append((diff - dd_inj) / sig if sig > 0 else np.nan)
                S = abs(diff) / sig if sig > 0 else np.nan
                fp_counts[1] += 1
                if np.isfinite(S) and S > 3:
                    fp_counts[0] += 1

        done += 1
        _progress(done, n_reals,
                  f"realisation {ridx+1}/{n_reals}")

    sys.stderr.write("\n")

    rows = []
    for (order, kc), cell in sorted(collect.items()):
        diffs = np.array(cell["diffs"])
        sigs = np.array(cell["sigmas"])
        u1s = np.array(cell["u1s"])
        u2s = np.array(cell["u2s"])
        dm1s = np.array(cell["dm1s"])
        dm2s = np.array(cell["dm2s"])
        mean_diff = float(np.mean(diffs))
        sigma_emp = float(np.std(diffs))
        mf = float(np.mean(sigs))
        z = (diffs - dd_inj) / sigs
        S = np.abs(diffs) / sigs
        rows.append({
            "dmdm_inj": dd_inj,
            "order": order,
            "kc": kc,
            "mean_diff": mean_diff,
            "bias": mean_diff - dd_inj,
            "sigma_empirical": sigma_emp,
            "sigma_formal_mean": mf,
            "r_sigma": (sigma_emp / mf) if mf > 0 else np.nan,
            "z_std": float(np.std(z)),
            "coverage": float(np.mean(np.abs(diffs - dd_inj) <= sigs)),
            "frac_S_gt3": float(np.mean(S > 3)),
            "n_realisations": len(diffs),
            "sigma1_formal_mean": float(np.mean(u1s)),
            "sigma2_formal_mean": float(np.mean(u2s)),
            "dm1_scatter": float(np.std(dm1s - T1)) if len(dm1s) else np.nan,
            "dm2_scatter": float(np.std(dm2s - T2)) if len(dm2s) else np.nan,
        })

    meta = {
        "orders": orders,
        "kcs": kc_values,
        "true_dms": (T1, T2),
        "dd_inj": dd_inj,
        "z_pool": np.array(z_pool),
        "fp_counts": fp_counts,
        "target_snr": args.target_snr,
        "snr_measured_mean": (float(np.mean(snr_measured))
                              if snr_measured else None),
    }
    _print_sim_summary(args, rows, meta, snr_measured)
    return rows, meta


def _print_sim_summary(args, rows, meta, snr_measured):
    """Compact post-run statistics: measured S/N plus pooled recovery,
    calibration and detection numbers across the (O, kc) grid."""
    if not rows:
        return
    z = meta["z_pool"]
    z = z[np.isfinite(z)]
    n_fp, n_tot = meta["fp_counts"]

    print(f"\n{'─' * 26} SYNTHETIC RUN SUMMARY {'─' * 26}")
    print(f"  Truth: DM1={meta['true_dms'][0]:+.4g}, "
          f"DM2={meta['true_dms'][1]:+.4g}  ->  "
          f"dDM_inj = {meta['dd_inj']:+.4g} pc cm^-3")
    if args.target_snr is not None:
        noise_desc = f"FIRES analytic, target S/N {args.target_snr:g}"
    else:
        noise_desc = f"simple {args.noise_level:.3g} x data std"
    print(f"  Noise: {noise_desc}; "
          f"{len(rows)} of {len(meta['orders']) * len(meta['kcs'])} "
          f"(O, kc) cells evaluated")

    if snr_measured:
        sm = np.array([s for s in snr_measured if np.isfinite(s)])
        if sm.size:
            print(f"  Measured S/N : mean {sm.mean():.1f}  median "
                  f"{np.median(sm):.1f}  std {sm.std():.1f}  "
                  f"range [{sm.min():.1f}, {sm.max():.1f}] (N={sm.size})")

    bias = np.array([r["bias"] for r in rows])
    semp = np.array([r["sigma_empirical"] for r in rows])
    sform = np.array([r["sigma_formal_mean"] for r in rows])
    rsig = np.array([r["r_sigma"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    with np.errstate(invalid="ignore"):
        print(f"  Delta-DM recovery (median over cells unless stated):")
        print(f"    bias         : median {np.nanmedian(bias):+.4g}   "
              f"mean {np.nanmean(bias):+.4g}   "
              f"|max| {np.nanmax(np.abs(bias)):.4g} pc cm^-3")
        print(f"    sigma empir. : median {np.nanmedian(semp):.4g}   "
              f"(best cell {np.nanmin(semp):.4g})")
        print(f"    sigma formal : median {np.nanmedian(sform):.4g}   "
              f"mean {np.nanmean(sform):.4g} pc cm^-3")
        print(f"    R_sigma      : median {np.nanmedian(rsig):.3f}   "
              f"range [{np.nanmin(rsig):.3f}, {np.nanmax(rsig):.3f}]")
        print(f"  Calibration:")
        print(f"    z-score      : mean {z.mean():+.3f}  std {z.std():.3f}  "
              f"|z|>3: {(np.abs(z) > 3).mean():.2%} (N={z.size})")
        print(f"    coverage     : median {np.nanmedian(cov):.1%}   "
              f"min {np.nanmin(cov):.1%}")
    fp_frac = n_fp / n_tot if n_tot else float("nan")
    print(f"  Detection:")
    print(f"    S>3 rate     : {fp_frac:.2%} of trials "
          f"(Gaussian expectation {_GAUSS_TAIL_3SIG:.2%}; "
          f"n={n_fp}/{n_tot})")
    print("─" * 78)


def plot_sim_recovery(rows, meta, output_dir: Path, ext: str):
    """Sim plot 1: distribution of recovered dDM over the (O, kc) grid
    against the known truth, plus the bias heatmap."""
    if not rows:
        return
    dd_inj = meta["dd_inj"]
    means = np.array([r["mean_diff"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(means, bins=30, alpha=0.75, color="C0", edgecolor="k",
            linewidth=0.4)
    ax.axvline(dd_inj, color="k", linestyle="--", linewidth=1.2,
               label=f"truth $={dd_inj:+.3g}$")
    med = float(np.median(means))
    ax.axvline(med, color="C1", linewidth=1.0, label=f"median $={med:+.4f}$")
    closest = float(np.nanmin(np.abs(means - dd_inj)))
    ax.set_xlabel(r"Recovered $\langle\Delta\mathrm{DM}_{21}\rangle$ "
                  "[pc cm$^{-3}$]")
    ax.set_ylabel("Cells in (O, kc) grid")
    ax.set_title(r"$\Delta\mathrm{DM}_{21}=\mathrm{DM}_2-\mathrm{DM}_1$"
                 f" across {len(rows)} (O, kc) settings"
                 f"\n(closest cell within {closest:.2g} of truth)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    bias_g, orders, kcs = rows_to_grid(rows, "bias", meta["orders"],
                                       meta["kcs"])
    vabs = np.nanmax(np.abs(bias_g)) * 1.05 or 1e-3
    draw_heatmap(axes[1], bias_g, kcs, orders, "RdBu_r", -vabs, vabs,
                 clabel=r"$b$ [pc cm$^{-3}$]",
                 title=r"$b=\langle\Delta\mathrm{DM}_{21}^{\,\rm rec}\rangle"
                       r"-\Delta\mathrm{DM}_{21}^{\,\rm inj}$")
    axes[1].axhline(y=3, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    _save_fig(output_dir, "sim_recovery", ext)


def plot_sim_calibration(rows, meta, output_dir: Path, ext: str):
    """Sim calibration figure (2 cols x 3 rows): per-component formal
    uncertainties, pooled empirical vs formal differential scatter, then
    the calibration ratio map and normalised-error distribution."""
    if not rows:
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 17))
    orders, kcs = meta["orders"], meta["kcs"]

    def _mean_map(key):
        g = np.full((len(orders), len(kcs)), np.nan)
        for row in rows:
            g[orders.index(row["order"]), kcs.index(row["kc"])] = row[key]
        return g

    # Row 1: per-component formal uncertainties on a shared colour scale
    s1_map = _mean_map("sigma1_formal_mean")
    s2_map = _mean_map("sigma2_formal_mean")
    smax = 1.05 * float(np.nanmax([np.nanmax(s1_map), np.nanmax(s2_map)])
                        ) or 1.0
    draw_heatmap(axes[0, 0], s1_map, kcs, orders, "viridis", 0.0, smax,
                 clabel=r"$\sigma_1$ [pc cm$^{-3}$]",
                 title=r"Component 1 formal uncertainty"
                       "\n"
                       r"$\sigma_1=\langle u_1\rangle_{\rm realisations}$")
    draw_heatmap(axes[0, 1], s2_map, kcs, orders, "viridis", 0.0, smax,
                 clabel=r"$\sigma_2$ [pc cm$^{-3}$]",
                 title=r"Component 2 formal uncertainty"
                       "\n"
                       r"$\sigma_2=\langle u_2\rangle_{\rm realisations}$")

    # Row 2: pooled differential scatter, empirical vs formal
    # (shared colour scale so the two maps are directly comparable)
    se_map = _mean_map("sigma_empirical")
    sf_map = _mean_map("sigma_formal_mean")
    dmax = 1.05 * float(np.nanmax([np.nanmax(se_map), np.nanmax(sf_map)])
                        ) or 1.0
    draw_heatmap(axes[1, 0], se_map, kcs, orders, "viridis", 0.0, dmax,
                 clabel=r"$\sigma_{\rm empirical}$ [pc cm$^{-3}$]",
                 title=r"Empirical differential scatter"
                       "\n"
                       r"$\sigma_{\rm empirical}=\mathrm{std}("
                       r"\Delta\mathrm{DM}_{21})$")
    draw_heatmap(axes[1, 1], sf_map, kcs, orders, "viridis", 0.0, dmax,
                 clabel=r"$\langle\sigma_{\rm formal}\rangle$ "
                        r"[pc cm$^{-3}$]",
                 title=r"Mean formal differential uncertainty"
                       "\n"
                       r"$\langle\sigma_{\rm formal}\rangle="
                       r"\langle\sqrt{\sigma_1^2+\sigma_2^2}\rangle$")

    # Row 3: calibration ratio + z distribution
    r_map = np.full((len(orders), len(kcs)), np.nan)
    counts = np.zeros((len(orders), len(kcs)))
    for row in rows:
        oi, ki = orders.index(row["order"]), kcs.index(row["kc"])
        if np.isfinite(row["r_sigma"]):
            r_map[oi, ki] = ((r_map[oi, ki] if np.isfinite(r_map[oi, ki])
                              else 0.0) + row["r_sigma"])
            counts[oi, ki] += 1
    r_map = np.where(counts > 0, r_map / np.maximum(counts, 1), np.nan)
    draw_heatmap(axes[2, 0], r_map, kcs, orders, "coolwarm",
                 vmin=0, vmax=None,
                 clabel=r"$R_\sigma$",
                 title=r"$R_\sigma=\sigma_{\rm empirical}/"
                       r"\langle\sigma_{\rm formal}\rangle$"
                       "\n(1 = well calibrated, >1 = overconfident, "
                       "<1 = conservative)")
    axes[2, 0].axhline(y=3, color="k", linestyle="--", linewidth=0.7,
                       alpha=0.5)

    z = meta["z_pool"]
    z = z[np.isfinite(z)]
    ax = axes[2, 1]
    ax.hist(z, bins=60, range=(-6, 6), density=True, alpha=0.7,
            color="C0", edgecolor="k", linewidth=0.3)
    xs = np.linspace(-6, 6, 300)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), "k--",
            linewidth=1.2, label=r"$\mathcal{N}(0,1)$")
    frac3 = float(np.mean(np.abs(z) > 3))
    ax.set_xlabel(r"$z$")
    ax.set_ylabel("Probability density")
    ax.set_title(r"$z=\left(\Delta\mathrm{DM}_{21}^{\,\rm rec}-"
                 r"\Delta\mathrm{DM}_{21}^{\,\rm inj}\right)/"
                 r"\sigma_{\rm formal}$"
                 f"\nN={z.size}, std={np.std(z):.3f}, "
                 f"|z|>3: {frac3:.2%}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(output_dir, "sim_calibration", ext)


def plot_sim_sensitivity(rows, meta, output_dir: Path, ext: str):
    """Sim plot 4: significance behaviour at this injection -- distribution
    of per-cell S and the false-positive rate versus Gaussian expectation."""
    if not rows:
        return

    s_cells = np.array([abs(r["mean_diff"]) /
                        max(r["sigma_formal_mean"], 1e-12) for r in rows])

    n_fp, n_tot = meta["fp_counts"]
    fp_frac = n_fp / n_tot if n_tot else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.hist(s_cells, bins=25, alpha=0.75, color="C0", edgecolor="k",
            linewidth=0.4)
    ax.axvline(3, color="k", linestyle="--", linewidth=1,
               label=r"3$\sigma$ threshold")
    ax.set_xlabel(r"$S_\Delta$ (per-cell median over realisations)")
    ax.set_ylabel("Cells in (O, kc) grid")
    ax.set_title(r"Detection significance: "
                 r"$S_\Delta=|\Delta\mathrm{DM}_{21}^{\,\rm rec}|/"
                 r"\sqrt{\sigma_1^2+\sigma_2^2}$"
                 "\nat "
                 rf"$\Delta\mathrm{{DM}}_{{\rm inj}}="
                 rf"{meta['dd_inj']:+.3g}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(["observed", "Gaussian\nexpectation"],
                  [100 * fp_frac, 100 * _GAUSS_TAIL_3SIG],
                  color=["C3", "0.7"], width=0.55)
    for b, v in zip(bars, [fp_frac, _GAUSS_TAIL_3SIG]):
        ax.annotate(f"{v:.3%}", xy=(b.get_x() + b.get_width() / 2,
                    b.get_height()), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(r"Fraction of trials with $S_\Delta>3$ [%]")
    ax.set_ylim(0, max(fp_frac, _GAUSS_TAIL_3SIG) * 130 or 1)
    ax.set_title(r"False-positive rate vs "
                 r"$P(|z|>3)=2\Phi(-3)\approx0.27\%$")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(output_dir, "sim_sensitivity", ext)


# ──────────────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────────────

def run_real_sweep(args, output_dir: Path):
    orders, kc_values = _sweep_grids(args)

    print(f"\n{'=' * 70}")
    print(f"REAL DATA SWEEP: {len(orders)} orders x {len(kc_values)} kc values")
    print(f"{'=' * 70}")

    stokes_i, freq_mhz, time_ms = _load_stokes_i(args)
    print(f"  Shape: {stokes_i.shape}  "
          f"Freq: {freq_mhz[0]:.1f}-{freq_mhz[-1]:.1f} MHz")

    print("\nPeak selection...")
    peak_regions = _resolve_peak_regions(args, time_ms, stokes_i)

    if args.dm_min is None or args.dm_max is None:
        raise SystemExit("Error: --dm-min and --dm-max required for real mode")

    # Shared dedisperser mirroring DMOptimiser exactly (noise stats from the
    # FULL time axis, seeded fill, endpoint-based output sizing). One input
    # DM for the whole file: the DM used to pre-correct the time series.
    dedisperser = _Dedisperser(
        stokes_i, freq_mhz, time_ms,
        input_dm=args.input_dm,
        reference_freq=args.ref_freq,
        seed=args.seed,
    )

    all_rows = []

    for pi, (start, end) in enumerate(peak_regions):
        tag = f"C{pi+1}" if len(peak_regions) > 1 else ""
        print(f"\n── Peak {pi+1}/{len(peak_regions)} ── "
              f"(input DM {args.input_dm})")

        print("  Dedispersing over DM grid (once)...")
        dm_values, CI_data = _prepare_component(
            args, stokes_i, freq_mhz, time_ms, (start, end), dedisperser)
        plot_dct_spectrum(CI_data, output_dir, args.ext, tag=tag)

        total = len(orders) * len(kc_values)
        done = 0

        for order in orders:
            for kc in kc_values:
                done += 1
                try:
                    dm_best, formal_unc, unc_minus, unc_plus = _eval_one(
                        dm_values, order, kc, CI_data)
                except Exception as e:
                    sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} "
                                     f"kc={kc:3d}  FAILED: {e}   \n")
                    sys.stderr.flush()
                    continue

                row = {
                    "order": order,
                    "kc": kc,
                    "recovered_dm": dm_best,
                    "formal_unc": formal_unc,
                    "unc_minus": unc_minus,
                    "unc_plus": unc_plus,
                }
                if args.true_dm is not None:
                    row["bias"] = dm_best - args.true_dm
                if tag:
                    row["peak"] = tag

                all_rows.append(row)
                bias_str = f"  bias={row['bias']:+.6f}" if "bias" in row else ""
                sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} kc={kc:3d}  "
                                 f"DM={dm_best:.6f}  "
                                 f"unc=-{unc_minus:.4g}/+{unc_plus:.4g}"
                                 f"{bias_str}  ")
                sys.stderr.flush()

        # ── Default SHRINE reference (O=3, auto-kc) ──
        sys.stderr.write("\n")
        print(f"  Running default SHRINE (O=3, auto-kc) for reference...")
        try:
            auto_kc = int(get_kc(CI_data))
            dm_ref, unc_ref, _, _ = _eval_one(dm_values, 3, auto_kc, CI_data)
            ref_kc = auto_kc
            print(f"  Default SHRINE: DM={dm_ref:.6f}  unc={unc_ref:.6f}  "
                  f"auto-kc={ref_kc}")
        except Exception as e:
            print(f"  Default SHRINE failed: {e}")
            import traceback; traceback.print_exc()
            dm_ref, unc_ref, ref_kc = None, None, None

        # Attach reference to rows for plotting
        if dm_ref is not None:
            for row in all_rows:
                if row.get("peak", "") == tag or (not tag and "peak" not in row):
                    row["_ref_dm"] = dm_ref
                    row["_ref_unc"] = unc_ref
                    row["_ref_kc"] = ref_kc

    return all_rows


# ──────────────────────────────────────────────────────────────────────
# DCT spectrum
# ──────────────────────────────────────────────────────────────────────

def plot_dct_spectrum(CI_data, output_dir: Path, ext: str, tag=""):
    """Save the DCT coefficient spectrum vs k index (original SHRINE style)."""
    try:
        kc = get_kc(CI_data)
    except RuntimeError:
        kc = None
    plt.plot(np.transpose(np.abs(CI_data)), '.')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r'$\mathregular{k}$ Index')
    plt.ylabel(r'DCT coefficients, $\mathregular{|C^Ti|}$')
    if kc is not None:
        plt.axvline(x=kc)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    _save_fig(output_dir, f"DCT_v_k_index_{tag}" if tag else "DCT_v_k_index",
              ext)


# ──────────────────────────────────────────────────────────────────────
# Top-hat comparison
# ──────────────────────────────────────────────────────────────────────

def plot_tophat_comparison(output_dir: Path, ext: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    k_length = 200
    kc = 50
    k = np.linspace(1, k_length, k_length)

    for order, color in [(3, "C0"), (5, "C1"), (10, "C2"), (15, "C3"), (20, "C4")]:
        fL = lowpass_response(k_length, kc, order)
        ax.plot(k, fL, label=f"O={order}", color=color, linewidth=1.5)

    tophat = np.where(k <= kc, 1.0, 0.0)
    ax.plot(k, tophat, "--", color="k", linewidth=1.2, alpha=0.6, label="Top-hat")

    ax.set_xlabel("k index")
    ax.set_ylabel("Filter response fL(k)")
    ax.set_title(f"Butterworth low-pass filter responses (kc={kc})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(output_dir, "tophat_comparison", ext)


# ──────────────────────────────────────────────────────────────────────
# Real-data summary plots
# ──────────────────────────────────────────────────────────────────────

def plot_real_heatmaps(rows, output_dir: Path, ext: str, input_dm=0.0):
    """Per-component DM_SHRINE (centred on the input DM) and formal
    uncertainty heatmaps, with the default-SHRINE reference marked."""
    if not rows:
        print("No results to plot.")
        return

    peak_labels = sorted(set(r.get("peak", "") for r in rows))
    n_peaks = len(peak_labels)
    if n_peaks == 0:
        return

    # Collect per-peak grids first so colour limits can be shared across
    # components (identical saturation keeps colours directly comparable)
    peak_data = []
    for plabel in peak_labels:
        prow = [r for r in rows if r.get("peak", "") == plabel]
        if not prow:
            peak_data.append(None)
            continue
        dm_grid, porders, pkcs = rows_to_grid(prow, "recovered_dm")
        unc_grid, _, _ = rows_to_grid(prow, "formal_unc", porders, pkcs)
        ref_rows = [r for r in prow if r.get("_ref_dm") is not None]
        peak_data.append({
            "orders": porders, "kcs": pkcs,
            "dm_grid": dm_grid, "unc_grid": unc_grid,
            "ref": ref_rows[0] if ref_rows else None, "label": plabel,
        })

    # Shared DM scale centred on the input DM: white always marks exact
    # recovery of the prep-corrected value
    dm_centre = input_dm
    halves = [float(np.max(np.abs(d["dm_grid"][np.isfinite(d["dm_grid"])]
                                  - dm_centre)))
              for d in peak_data
              if d is not None and np.any(np.isfinite(d["dm_grid"]))]
    dm_half = max(halves) * 1.05 if halves else 1.0
    if not dm_half > 0:
        dm_half = 1.0
    dm_vmin, dm_vmax = dm_centre - dm_half, dm_centre + dm_half

    fig, axes = plt.subplots(n_peaks, 2,
                             figsize=(13, 5 * n_peaks), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for row_idx, pd in enumerate(peak_data):
        if pd is None:
            for col in range(2):
                axes[row_idx, col].set_visible(False)
            continue

        porders, pkcs = pd["orders"], pd["kcs"]
        dm_grid, unc_grid = pd["dm_grid"], pd["unc_grid"]
        label = pd["label"]
        tag = f" — {label}" if label else ""

        ufin = unc_grid[np.isfinite(unc_grid)]
        unc_vmax = float(np.max(ufin)) * 1.05 if ufin.size else 1.0
        unc_vmax = unc_vmax if unc_vmax > 0 else 1.0

        kc_step = pkcs[1] - pkcs[0] if len(pkcs) > 1 else 1
        img_ext = [pkcs[0] - 0.5 * kc_step, pkcs[-1] + 0.5 * kc_step,
                   porders[0] - 0.5, porders[-1] + 0.5]

        # DM_SHRINE panel
        ax_dm = axes[row_idx, 0]
        draw_heatmap(
            ax_dm, dm_grid, pkcs, porders, "PuOr", dm_vmin, dm_vmax,
            clabel=r"$\mathrm{DM}_{\rm SHRINE}$ [pc cm$^{-3}$]",
            title=rf"$\mathrm{{DM}}_{{\rm SHRINE}}{tag} "
                  rf"(centred on input $\pm{dm_half:.3g}$)")
        annotate_cells(ax_dm, pkcs, porders, dm_grid - dm_centre,
                       lambda v: f"{v:+.3g}", dm_vmin, dm_vmax, "centred")

        # Formal-uncertainty panel
        ax_unc = axes[row_idx, 1]
        draw_heatmap(
            ax_unc, unc_grid, pkcs, porders, "Reds", 0.0, unc_vmax,
            clabel=r"$(\Delta_{\rm DM}^{+}{+}|\Delta_{\rm DM}^{-}|)/2$ "
                   r"[pc cm$^{-3}$]",
            title=rf"DM range half-width{tag} (max {unc_vmax:.3g})")
        annotate_cells(ax_unc, pkcs, porders, unc_grid,
                       lambda v: f"{v:.3g}", 0.0, unc_vmax, "seq")

        if row_idx == n_peaks - 1:
            for ax in (ax_dm, ax_unc):
                ax.set_xlabel("kc")

        # Overlay default-SHRINE reference (star + crosshair + info box)
        ref = pd["ref"]
        if ref is None:
            continue
        ref_kc = ref["_ref_kc"]
        ref_dm = ref["_ref_dm"]
        ref_unc = ref["_ref_unc"]

        star_kc = ref_kc
        kc_in_grid = pkcs[0] <= ref_kc <= pkcs[-1]
        if not kc_in_grid:
            star_kc = pkcs[0] if ref_kc < pkcs[0] else pkcs[-1]

        for ax in (ax_dm, ax_unc):
            ax.axhline(y=3, color="red", linestyle="--", linewidth=0.8,
                       alpha=0.5)
            if kc_in_grid:
                ax.axvline(x=ref_kc, color="red", linestyle="--",
                           linewidth=0.8, alpha=0.5)
            ax.plot(star_kc, 3, marker="*", markersize=20, color="red",
                    markeredgecolor="white", markeredgewidth=1.0, zorder=10)

        kc_note = f"  (swept {pkcs[0]}-{pkcs[-1]})" if not kc_in_grid else ""

        # Pick annotation offset so the box stays inside the axes
        kc_frac = (star_kc - pkcs[0]) / (pkcs[-1] - pkcs[0]) \
            if len(pkcs) > 1 else 0.5
        o_frac = (3 - porders[0]) / (porders[-1] - porders[0]) \
            if len(porders) > 1 else 0.5
        x_off = -50 if kc_frac > 0.6 else 20
        y_off = -50 if o_frac > 0.6 else 20

        ax_dm.annotate(
            f"O=3, kc={ref_kc}\nDM={ref_dm:.4f}\n"
            + r"$\sigma=$" + f"{ref_unc:.4f}",
            xy=(star_kc, 3), xycoords="data",
            xytext=(x_off, y_off), textcoords="offset points",
            fontsize=8, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red",
                      alpha=0.85),
        )
        print(f"  Reference {label or 'single'}: O=3, kc={ref_kc}, "
              f"DM={ref_dm:.6f}, unc={ref_unc:.6f}{kc_note}")

    plt.savefig(output_dir / f"real_sweep.{ext}", dpi=150)
    plt.close()
    print(f"Saved {output_dir / f'real_sweep.{ext}'}")


# ──────────────────────────────────────────────────────────────────────
# Real-data differential DM (DM2 - DM1)
# ──────────────────────────────────────────────────────────────────────

def compute_differential_rows(all_rows):
    """Combine per-component sweep rows into differential-DM rows.

    Returns (diff_rows, ref_pair) where diff_rows have one entry per
    (order, kc) with dm1/u1/dm2/u2, diff_dm = DM2-DM1,
    sigma_diff = sqrt(u1^2+u2^2), and signed/absolute significances.
    ref_pair is (row1, row2) default-SHRINE reference rows, or Nones.
    """
    labels = sorted(set(r.get("peak", "") for r in all_rows))
    if len(labels) < 2:
        return [], (None, None)

    def _index(label):
        out = {}
        for r in all_rows:
            if r.get("peak", "") == label:
                out[(r["order"], r["kc"])] = r
        return out

    idx1, idx2 = _index(labels[0]), _index(labels[1])
    common = sorted(set(idx1) & set(idx2))

    diff_rows = []
    for key in common:
        r1, r2 = idx1[key], idx2[key]
        dm1, u1 = r1["recovered_dm"], r1["formal_unc"]
        dm2, u2 = r2["recovered_dm"], r2["formal_unc"]
        sigma = float(np.hypot(u1, u2))
        ddm = float(dm2 - dm1)
        sig = ddm / sigma if sigma > 0 else np.nan
        diff_rows.append({
            "order": key[0], "kc": key[1],
            "dm1": dm1, "unc1": u1, "dm2": dm2, "unc2": u2,
            "diff_dm": ddm, "sigma_diff": sigma,
            "sig_signed": sig, "sig_abs": abs(sig),
        })

    def _ref(label_idx):
        for r in all_rows:
            if (r.get("peak", "") == labels[label_idx]
                    and r.get("_ref_dm") is not None):
                return r
        return None

    return diff_rows, (_ref(0), _ref(1))


def plot_real_differential(diff_rows, refs, output_dir: Path, ext: str):
    """Main real-data result figure: component stability, differential DM,
    its significance, and the formal differential uncertainty."""
    if not diff_rows:
        print("Differential plot skipped (needs >= 2 peaks).")
        return

    dm1_g, orders, kcs = rows_to_grid(diff_rows, "dm1")
    dm2_g, _, _ = rows_to_grid(diff_rows, "dm2", orders, kcs)
    ddm_g, _, _ = rows_to_grid(diff_rows, "diff_dm", orders, kcs)
    sig_g, _, _ = rows_to_grid(diff_rows, "sigma_diff", orders, kcs)
    s_g, _, _ = rows_to_grid(diff_rows, "sig_signed", orders, kcs)
    absg_g, _, _ = rows_to_grid(diff_rows, "sig_abs", orders, kcs)

    kc_step = kcs[1] - kcs[0] if len(kcs) > 1 else 1
    img_ext = [kcs[0] - 0.5 * kc_step, kcs[-1] + 0.5 * kc_step,
               orders[0] - 0.5, orders[-1] + 0.5]

    def _lim(key_grid, default):
        fin = key_grid[np.isfinite(key_grid)]
        return float(np.max(np.abs(fin))) * 1.05 if fin.size else default

    # Shared stability scale across the two components: each panel centred
    # on that component's median, identical half-width
    halves = []
    for g in (dm1_g, dm2_g):
        fin = g[np.isfinite(g)]
        if fin.size:
            halves.append(float(np.max(np.abs(fin - np.median(fin)))))
    stab_half = (max(halves) * 1.05) if halves else 1.0
    med1 = float(np.median(dm1_g[np.isfinite(dm1_g)]))
    med2 = float(np.median(dm2_g[np.isfinite(dm2_g)]))

    dd_abs = _lim(ddm_g, 1.0)
    s_abs = _lim(s_g, 5.0)
    sg_max = _lim(sig_g, 1.0)
    asg_max = _lim(absg_g, 5.0)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), squeeze=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    panels = [
        (axes[0, 0], dm1_g, "PuOr", med1 - stab_half, med1 + stab_half,
         rf"$\mathrm{{DM}}_1$ (centred on median {med1:.4g})",
         r"$\mathrm{DM}$ [pc cm$^{-3}$]", lambda v: f"{v:+.3g}", "centred"),
        (axes[0, 1], dm2_g, "PuOr", med2 - stab_half, med2 + stab_half,
         rf"$\mathrm{{DM}}_2$ (centred on median {med2:.4g})",
         r"$\mathrm{DM}$ [pc cm$^{-3}$]", lambda v: f"{v:+.3g}", "centred"),
        (axes[0, 2], ddm_g, "PuOr", -dd_abs, dd_abs,
         r"$\Delta\mathrm{DM}_{21}=\mathrm{DM}_2-\mathrm{DM}_1$",
         r"$\Delta\mathrm{DM}_{21}$ [pc cm$^{-3}$]", lambda v: f"{v:+.3g}",
         "extremes"),
        (axes[1, 0], s_g, "PuOr", -s_abs, s_abs,
         r"Significance $S=\Delta\mathrm{DM}_{21}/\sigma_{\Delta}$",
         r"$S=(\mathrm{DM}_2-\mathrm{DM}_1)/\sigma_\Delta$",
         lambda v: f"{v:+.2f}", "centred"),
        (axes[1, 1], sig_g, "Reds", 0.0, sg_max,
         r"Formal differential uncertainty $\sigma_{\Delta}"
         r"=\sqrt{\sigma_1^2+\sigma_2^2}$",
         r"$\sigma_{\Delta}$ [pc cm$^{-3}$]", lambda v: f"{v:.3g}", "seq"),
        (axes[1, 2], absg_g, "viridis", 0.0, asg_max,
         r"Absolute significance $|S|$",
         r"$|S|$", lambda v: f"{v:.2f}", "seq"),
    ]

    for ax, grid, cmap, vmin, vmax, title, clabel, fmt, amode in panels:
        if np.any(np.isfinite(grid)):
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax, extent=img_ext,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, label=clabel)
            annotate_cells(ax, kcs, orders, grid, fmt, vmin, vmax, amode)
        if "Significance" in title or "|S|" in title:
            ax.axhline(y=3, color="k", linestyle="--", linewidth=0.7,
                       alpha=0.5)
        ax.set_xlabel("kc")
        ax.set_ylabel("Order O")
        ax.set_title(title)

    # Reference summary box on the main-result panel
    r1, r2 = refs
    if r1 is not None and r2 is not None:
        d_ref = r2["_ref_dm"] - r1["_ref_dm"]
        s_ref = float(np.hypot(r1["_ref_unc"], r2["_ref_unc"]))
        axes[0, 2].annotate(
            f"default SHRINE\n"
            rf"$\Delta\mathrm{{DM}}_{{21}}$={d_ref:.4f}"
            rf" $\pm$ {s_ref:.4f}" "\n"
            rf"$S={d_ref/s_ref:+.2f}\sigma$",
            xy=(0.03, 0.96), xycoords="axes fraction",
            va="top", fontsize=8, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red",
                      alpha=0.85),
        )

    plt.savefig(output_dir / f"real_differential.{ext}", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / f'real_differential.{ext}'}")

    # Reference summary box on the main-result panel
    r1, r2 = refs
    if r1 is not None and r2 is not None:
        d_ref = r2["_ref_dm"] - r1["_ref_dm"]
        s_ref = float(np.hypot(r1["_ref_unc"], r2["_ref_unc"]))
        axes[0, 2].annotate(
            f"default SHRINE\n"
            rf"$\Delta\mathrm{{DM}}_{{21}}$={d_ref:.4f}"
            rf" $\pm$ {s_ref:.4f}" "\n"
            rf"$S={d_ref/s_ref:+.2f}\sigma$",
            xy=(0.03, 0.96), xycoords="axes fraction",
            va="top", fontsize=8, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red",
                      alpha=0.85),
        )

    plt.savefig(output_dir / f"real_differential.{ext}", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / f'real_differential.{ext}'}")


def save_csv(rows, output_dir: Path, filename: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    path = output_dir / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.mode in ("synthetic", "both"):
        synth_rows, synth_meta = run_synthetic_differential(args, output_dir)
        save_csv(synth_rows, output_dir, "synthetic_differential.csv")
        if synth_rows:
            plot_sim_recovery(synth_rows, synth_meta, output_dir, args.ext)
            plot_sim_calibration(synth_rows, synth_meta, output_dir, args.ext)
            plot_sim_sensitivity(synth_rows, synth_meta, output_dir, args.ext)

    if args.mode in ("real", "both"):
        real_rows = run_real_sweep(args, output_dir)
        save_csv(real_rows, output_dir, "real_sweep.csv")

        diff_rows, refs = compute_differential_rows(real_rows)
        save_csv(diff_rows, output_dir, "real_differential.csv")
        plot_real_differential(diff_rows, refs, output_dir, args.ext)

        if real_rows:
            plot_real_heatmaps(real_rows, output_dir, args.ext,
                               input_dm=args.input_dm)

    if args.check_tophat:
        plot_tophat_comparison(output_dir, args.ext)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
