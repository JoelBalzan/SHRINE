#!/usr/bin/env python3
"""Filter design & order sweep test suite for SHRINE structure maximisation.

Sweeps Butterworth order (O) and spectral cutoff (kc) over a 2-D grid,
measuring bias, scatter, formal uncertainty, and coverage across noise
realisations (synthetic mode) or on a single real dataset (real mode).

In both modes the data is handled identically to a real observation:
the .npy files are loaded, the DM grid is built, dedisperion produces
I(DM, t), and :func:`struct_max_core.structure_param` evaluates each
(O, kc) pair.

Usage examples
--------------
# Synthetic sweep using FIRES output as signal template
python test_filter_sweep.py --mode synthetic \
    --fires-dir /path/to/fires_output/ --n-realisations 50

# Real data sweep
python test_filter_sweep.py --mode real \
    --fires-dir /path/to/fires_output/ \
    --dm-min 300 --dm-max 400 --true-dm 335.87

# Or with individual files
python test_filter_sweep.py --mode real \
    --stokes-i /path/to/stokes_i.npy \
    --freq /path/to/freq.npy --time /path/to/time.npy \
    --dm-min 300 --dm-max 400

# Custom grid
python test_filter_sweep.py --mode synthetic \
    --fires-dir /path/to/fires_output/ \
    -o 1 --o-max 20 --o-step 2 --kc-min 10 --kc-max 100 --kc-step 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time as _time
from pathlib import Path
from typing import List, Tuple

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
                   help="synthetic = injected-DM tests, real = single file, both = both")

    # Synthetic parameters
    s = p.add_argument_group("Synthetic data (uses FIRES output as signal template)")
    s.add_argument("--fires-dir", default=None,
                   help="FIRES output directory — clean signal for noise realisations")
    s.add_argument("--injected-dm", type=float, default=0.0,
                   help="DM already applied to FIRES data (pc cm^-3)")
    s.add_argument("--n-realisations", "-n", type=int, default=50,
                   help="Noise realisations per (O, kc) in synthetic mode")
    s.add_argument("--noise-level", type=float, default=1.0,
                   help="RMS noise to add per realisation (relative to signal std)")
    s.add_argument("--seed", type=int, default=42, help="Base random seed")
    s.add_argument("--dm-range-half", type=float, default=5.0,
                   help="Half-width of delta-DM search range for synthetic data")

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
                   help="DM already applied to real data")
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

_DM_CONSTANT = 4.148808e6


def _load_stokes_i(args, synth_dir=None, synth_label=None):
    """Return (stokes_i, freq_mhz, time_ms) from CLI args or synthetic .npy files."""
    if synth_dir is not None:
        # Load from a directory of .npy files (FIRES or manually saved)
        fires_path = Path(synth_dir)
        si = np.load(fires_path / "out_I.npy")
        freq_mhz = np.load(fires_path / "freq.npy")
        time_ms = np.load(fires_path / "time.npy")
        return si, freq_mhz, time_ms

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
            raise SystemExit("Error: provide --fires-dir, --stokes-i, or --stokes-cube")
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


def _dedisperse_i_dm_t(stokes_i, freq_mhz, time_ms, dm_values, input_dm=0.0):
    """Dedisperse stokes_i at each trial DM, return I(DM, t)."""
    n_freq, n_time = stokes_i.shape
    dt = float(np.median(np.diff(time_ms)))
    ref_freq = float(np.max(freq_mhz))

    # Compute output size
    delays_all = []
    for dm in dm_values:
        eff = input_dm - dm
        delays_all.append(
            _DM_CONSTANT * eff * (1.0 / freq_mhz**2 - 1.0 / ref_freq**2)
        )
    delays_all = np.concatenate(delays_all)
    min_shift = int(np.floor(np.min(delays_all) / dt))
    max_shift = int(np.ceil(np.max(delays_all) / dt))
    out_size = n_time + abs(max_shift - min_shift)

    i_dm_t = np.zeros((len(dm_values), out_size), dtype=float)

    for idx, dm in enumerate(dm_values):
        eff = input_dm - dm
        delays_ms = _DM_CONSTANT * eff * (1.0 / freq_mhz**2 - 1.0 / ref_freq**2)
        delay_samp = np.round(delays_ms / dt).astype(int)
        min_s = int(np.min(delay_samp))
        counts = np.zeros(out_size, dtype=float)

        for ch in range(n_freq):
            sh = delay_samp[ch] - min_s
            cp = min(n_time, out_size - sh)
            if cp > 0 and 0 <= sh < out_size:
                i_dm_t[idx, sh : sh + cp] += stokes_i[ch, :cp]
                counts[sh : sh + cp] += 1.0

        valid = counts > 0
        i_dm_t[idx, valid] /= counts[valid]

    return i_dm_t


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

def _run_single(args, dm_values, stokes_i, freq_mhz, time_ms, input_dm=0.0):
    """Dedisperse once and return the I(DM, t) array (reused across O/kc)."""
    return _dedisperse_i_dm_t(stokes_i, freq_mhz, time_ms, dm_values, input_dm)


def _eval_one(i_dm_t, dm_values, order, kc, CI_data=None):
    """Run structure_param for one (O, kc) and return metrics.

    If *CI_data* (pre-computed DCT of i_dm_t) is provided, uses the fast
    :func:`structure_param_from_dct` path to avoid redundant DCTs.
    """
    if CI_data is not None:
        result = structure_param_from_dct(CI_data, kc=kc, order=order)
        result["dm_data"] = dm_values
    else:
        result = structure_param(i_dm_t, order=order, kc=kc)
        result["dm_data"] = dm_values
    dm_best = recovered_dm(result["dm_data"], result["sp"])
    low_dm, high_dm = dm_range_from_relative_uncertainty(
        result["sp"], result["relative_uncertainty"], result["dm_data"]
    )
    unc_minus = 0.0 if low_dm is None else dm_best - low_dm
    unc_plus = 0.0 if high_dm is None else high_dm - dm_best
    formal_unc = 0.5 * (unc_minus + unc_plus)
    return dm_best, formal_unc


# ──────────────────────────────────────────────────────────────────────
# Synthetic sweep
# ──────────────────────────────────────────────────────────────────────

def run_synthetic_sweep(args, output_dir: Path):
    if not args.fires_dir:
        raise SystemExit("Error: --fires-dir required for synthetic mode "
                         "(provide a FIRES output directory)")

    orders = list(range(args.o_min, args.o_max + 1, args.o_step))
    kc_values = list(range(args.kc_min, args.kc_max + 1, args.kc_step))
    n_reals = args.n_realisations
    seed_base = args.seed

    print(f"\n{'=' * 70}")
    print(f"SYNTHETIC SWEEP: {len(orders)} orders x {len(kc_values)} kc "
          f"x {n_reals} realisations = {len(orders)*len(kc_values)*n_reals} runs")
    print(f"{'=' * 70}")

    # Load FIRES output as signal template
    fires_path = Path(args.fires_dir)
    signal = np.load(fires_path / "out_I.npy")
    freq_mhz = np.load(fires_path / "freq.npy")
    time_ms  = np.load(fires_path / "time_ms.npy") if (fires_path / "time_ms.npy").exists() \
               else np.load(fires_path / "time.npy")
    print(f"  Signal shape: {signal.shape}  "
          f"Freq: {freq_mhz[0]:.1f}-{freq_mhz[-1]:.1f} MHz  "
          f"Time: {time_ms[0]:.3f}-{time_ms[-1]:.3f} ms")

    # DM grid centred on 0 (delta-DM relative to input_dm)
    dm_half = args.dm_range_half
    dm_values = np.linspace(-dm_half, dm_half, 300)
    print(f"  DM grid: {len(dm_values)} points, "
          f"[{dm_values[0]:.3f}, {dm_values[-1]:.3f}] pc cm^-3")

    # Scale signal so its RMS = 1 (noise will be added at level args.noise_level)
    sig_rms = float(np.std(signal))
    if sig_rms > 0:
        signal_scaled = signal / sig_rms
    else:
        signal_scaled = signal

    rows = []
    total = len(orders) * len(kc_values)
    first_CI = None

    # Collect results per (order, kc) across realisations
    from collections import defaultdict
    collect = defaultdict(lambda: {"dms": [], "uncs": []})
    done = 0

    for real in range(n_reals):
        rng = np.random.default_rng(seed_base + real)
        noise = rng.standard_normal(signal.shape) * args.noise_level
        noisy = signal_scaled + noise

        i_dm_t = _dedisperse_i_dm_t(noisy, freq_mhz, time_ms,
                                    dm_values, input_dm=args.injected_dm)
        CI_data = dct(i_dm_t, norm="ortho")
        if first_CI is None:
            first_CI = CI_data

        for order in orders:
            for kc in kc_values:
                try:
                    dm_best, formal_unc = _eval_one(i_dm_t, dm_values, order, kc,
                                                    CI_data=CI_data)
                except Exception:
                    continue
                collect[(order, kc)]["dms"].append(dm_best)
                collect[(order, kc)]["uncs"].append(formal_unc)

    plot_dct_spectrum(first_CI, output_dir, args.ext, tag="synthetic")

    done = 0
    for order in orders:
        for kc in kc_values:
            done += 1
            dms_recovered = collect[(order, kc)]["dms"]
            form_uncs = collect[(order, kc)]["uncs"]

            if not dms_recovered:
                if args.verbose:
                    sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} kc={kc:3d}  FAILED   \n")
                    sys.stderr.flush()
                continue

            dms = np.array(dms_recovered)
            uncs = np.array(form_uncs)
            bias = float(np.mean(dms - args.injected_dm))
            scatter = float(np.std(dms))
            mean_unc = float(np.mean(uncs))
            within = np.abs(dms - args.injected_dm) <= uncs
            coverage = float(np.mean(within))

            rows.append({
                "order": order,
                "kc": kc,
                "bias": bias,
                "scatter": scatter,
                "mean_formal_unc": mean_unc,
                "coverage": coverage,
                "n_realisations": len(dms),
                "mean_recovered_dm": float(np.mean(dms)),
            })

            sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} kc={kc:3d}  "
                             f"bias={bias:+.6f}  scatter={scatter:.6f}  "
                             f"form_unc={mean_unc:.6f}  coverage={coverage:.1%}  ")
            sys.stderr.flush()

    sys.stderr.write("\n")
    return rows


# ──────────────────────────────────────────────────────────────────────
# Real sweep
# ──────────────────────────────────────────────────────────────────────

def run_real_sweep(args, output_dir: Path):
    orders = list(range(args.o_min, args.o_max + 1, args.o_step))
    kc_values = list(range(args.kc_min, args.kc_max + 1, args.kc_step))

    print(f"\n{'=' * 70}")
    print(f"REAL DATA SWEEP: {len(orders)} orders x {len(kc_values)} kc values")
    print(f"{'=' * 70}")

    stokes_i, freq_mhz, time_ms = _load_stokes_i(args)
    print(f"  Shape: {stokes_i.shape}  "
          f"Freq: {freq_mhz[0]:.1f}-{freq_mhz[-1]:.1f} MHz")

    #plt.imshow(stokes_i, aspect="auto", extent=(time_ms[0], time_ms[-1], freq_mhz[0], freq_mhz[-1]), origin="lower", cmap="plasma")
    #plt.xlabel("Time (ms)")
    #plt.ylabel("Frequency (MHz)")
    #plt.show()

    dt = float(np.median(np.diff(time_ms)))
    # Peak selection
    print("\nPeak selection...")
    peak_regions = _resolve_peak_regions(args, time_ms, stokes_i)

    if args.dm_min is None or args.dm_max is None:
        raise SystemExit("Error: --dm-min and --dm-max required for real mode")

    # Shared dedisperser mirroring DMOptimiser exactly (noise stats from the
    # FULL time axis, seeded fill, endpoint-based output sizing).
    dedisperser = _Dedisperser(
        stokes_i, freq_mhz, time_ms,
        input_dm=args.input_dm,
        reference_freq=args.ref_freq,
        seed=args.seed,
    )

    all_rows = []

    for pi, (start, end) in enumerate(peak_regions):
        tag = f"C{pi+1}" if len(peak_regions) > 1 else ""
        print(f"\n── Peak {pi+1}/{len(peak_regions)} ──")

        si_peak = stokes_i[:, start:end]
        tm_peak = time_ms[start:end]

        #plt.imshow(si_peak, aspect="auto", extent=(tm_peak[0], tm_peak[-1], freq_mhz[0], freq_mhz[-1]), origin="lower", cmap="plasma")
        #plt.xlabel("Time (ms)")
        #plt.ylabel("Frequency (MHz)")
        #plt.show()

        dm_values = _build_dm_grid(freq_mhz, tm_peak, args.dm_min, args.dm_max,
                                   ref_freq=args.ref_freq,
                                   dm_step=args.dm_step, ndm=args.ndm)
        print(f"  DM grid: {len(dm_values)} points, "
              f"[{dm_values[0]:.2f}, {dm_values[-1]:.2f}]")

        print("  Dedispersing over DM grid (once)...")
        output_size = dedisperser.max_output_size_for_dm_range(
            si_peak, (args.dm_min, args.dm_max))
        i_dm_t = np.zeros((len(dm_values), output_size), dtype=float)
        for idx, dm in enumerate(dm_values):
            dedisp = dedisperser.dedisperse(
                si_peak, dm, output_size=output_size, mode=args.dedisp_mode)
            i_dm_t[idx] = np.nansum(dedisp, axis=0)
        print(f"  I(DM,t) shape: {i_dm_t.shape}")
        
        plt.imshow(i_dm_t, aspect="auto", extent=(0, i_dm_t.shape[1]*dt, dm_values[-1], dm_values[0]))
        plt.xlabel("Time (ms)")
        plt.ylabel("DM (pc cm^-3)")
        plt.title("Dedispersed Intensity")
        plt.show()

        # Pre-compute DCT once (avoids redundant DCTs in the O/kc sweep)
        print("  Computing DCT (once for all O/kc)...")
        CI_data = dct(i_dm_t, norm="ortho")
        print(f"  DCT shape: {CI_data.shape}")

        plot_dct_spectrum(CI_data, output_dir, args.ext, tag=tag)

        total = len(orders) * len(kc_values)
        done = 0

        for order in orders:
            for kc in kc_values:
                done += 1
                t0 = _time.time()
                try:
                    dm_best, formal_unc = _eval_one(i_dm_t, dm_values, order, kc,
                                                    CI_data=CI_data)
                except Exception as e:
                    sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} kc={kc:3d}  FAILED: {e}   \n")
                    sys.stderr.flush()
                    continue

                elapsed = _time.time() - t0
                row = {
                    "order": order,
                    "kc": kc,
                    "recovered_dm": dm_best,
                    "formal_unc": formal_unc,
                }
                if args.true_dm is not None:
                    row["bias"] = dm_best - args.true_dm
                if tag:
                    row["peak"] = tag

                all_rows.append(row)
                bias_str = f"  bias={row['bias']:+.6f}" if "bias" in row else ""
                sys.stderr.write(f"\r  [{done}/{total}] O={order:2d} kc={kc:3d}  "
                                 f"DM={dm_best:.6f}  unc={formal_unc:.6f}"
                                 f"{bias_str}  ")
                sys.stderr.flush()

        # ── Default SHRINE reference (O=3, auto-kc) ──
        sys.stderr.write("\n")
        print(f"  Running default SHRINE (O=3, auto-kc) for reference...")
        try:
            auto_kc = int(get_kc(CI_data))
            dm_ref, unc_ref = _eval_one(i_dm_t, dm_values, 3, auto_kc, CI_data=CI_data)
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
    prefix = f"DCT_v_k_index_{tag}" if tag else "DCT_v_k_index"
    path = output_dir / f"{prefix}.{ext}"
    plt.savefig(path)
    plt.clf()
    print(f"Saved {path}")


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
    plt.tight_layout()
    plt.savefig(output_dir / f"tophat_comparison.{ext}", dpi=150)
    plt.close()
    print(f"Saved top-hat comparison to {output_dir / f'tophat_comparison.{ext}'}")


# ──────────────────────────────────────────────────────────────────────
# Summary plots
# ──────────────────────────────────────────────────────────────────────

def plot_sweep_heatmaps(rows, output_dir: Path, ext: str, mode: str, input_dm=0.0):
    if not rows:
        print("No results to plot.")
        return

    orders = sorted(set(r["order"] for r in rows))
    kcs = sorted(set(r["kc"] for r in rows))

    def _grid(key):
        g = np.full((len(orders), len(kcs)), np.nan)
        for r in rows:
            if key in r and r[key] is not None:
                oi = orders.index(r["order"])
                ki = kcs.index(r["kc"])
                g[oi, ki] = r[key]
        return g

    bias_grid = _grid("bias")
    scatter_grid = _grid("scatter")
    unc_grid = _grid("mean_formal_unc")
    if not np.any(np.isfinite(unc_grid)):
        unc_grid = _grid("formal_unc")
    coverage_grid = _grid("coverage")

    has_bias = np.any(np.isfinite(bias_grid))
    has_scatter = np.any(np.isfinite(scatter_grid))
    has_coverage = np.any(np.isfinite(coverage_grid))

    if mode == "synthetic":
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        def _imshow(ax, data, cmap, label, title, **kw):
            if np.any(np.isfinite(data)):
                im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                               extent=[kcs[0], kcs[-1], orders[0], orders[-1]],
                               **kw)
                plt.colorbar(im, ax=ax, label=label)
            ax.set_xlabel("kc")
            ax.set_ylabel("Order O")
            ax.set_title(title)

        if has_bias:
            vabs = max(np.nanpercentile(np.abs(bias_grid[np.isfinite(bias_grid)]), 95), 1e-6)
            _imshow(axes[0], bias_grid, "RdBu_r",
                    r"$\langle DM_{\rm rec} - DM_{\rm true}\rangle$ [pc cm$^{-3}$]",
                    "Bias", vmin=-vabs, vmax=vabs)
        else:
            axes[0].set_title("Bias (no data)")

        _imshow(axes[1], scatter_grid, "viridis",
                r"Scatter $\sigma(DM_{\rm rec})$ [pc cm$^{-3}$]",
                "Scatter (std of recovered DM)")

        _imshow(axes[2], unc_grid, "viridis",
                r"Mean formal $\sigma$ [pc cm$^{-3}$]",
                "Mean formal uncertainty")

        if has_coverage:
            _imshow(axes[3], coverage_grid, "RdYlGn",
                    "Coverage fraction",
                    r"Coverage (within formal $\sigma$)", vmin=0, vmax=1)
        else:
            axes[3].set_title("Coverage (no data)")

        plt.suptitle("SHRINE filter sweep — synthetic", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / f"sweep_heatmaps.{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_dir / f'sweep_heatmaps.{ext}'}")

        # Scatter vs formal uncertainty
        if has_scatter and np.any(np.isfinite(unc_grid)):
            fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes2[0]
            for oi, order in enumerate(orders):
                s, u = scatter_grid[oi], unc_grid[oi]
                v = np.isfinite(s) & np.isfinite(u)
                if np.any(v):
                    ax.plot(u[v], s[v], "o-", label=f"O={order}", markersize=3)
            lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05]
            ax.plot(lims, lims, "--", color="k", alpha=0.5)
            ax.set_xlabel(r"Mean formal $\sigma$")
            ax.set_ylabel(r"Scatter $\sigma(DM_{\rm rec})$")
            ax.set_title("Scatter vs formal uncertainty\n(diagonal = ideal)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            ax = axes2[1]
            for ki, kc in enumerate(kcs):
                s, u = scatter_grid[:, ki], unc_grid[:, ki]
                v = np.isfinite(s) & np.isfinite(u)
                if np.any(v):
                    ax.plot(u[v], s[v], "o-", label=f"kc={kc}", markersize=3)
            ax.plot(lims, lims, "--", color="k", alpha=0.5)
            ax.set_xlabel(r"Mean formal $\sigma$")
            ax.set_ylabel(r"Scatter $\sigma(DM_{\rm rec})$")
            ax.set_title("Scatter vs formal uncertainty\n(coloured by kc)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_vs_formal.{ext}", dpi=150)
            plt.close()
            print(f"Saved {output_dir / f'scatter_vs_formal.{ext}'}")

    else:
        # Real mode — single figure, one row per peak
        peak_labels = sorted(set(r.get("peak", "") for r in rows))
        n_peaks = len(peak_labels)
        if n_peaks == 0:
            return

        # Collect per-peak data first so we can use shared colour limits
        peak_data = []
        for plabel in peak_labels:
            prow = [r for r in rows if r.get("peak", "") == plabel]
            if not prow:
                peak_data.append(None)
                continue
            porders = sorted(set(r["order"] for r in prow))
            pkcs = sorted(set(r["kc"] for r in prow))

            def _pgrid(key):
                g = np.full((len(porders), len(pkcs)), np.nan)
                for r in prow:
                    if key in r and r[key] is not None and r["order"] in porders and r["kc"] in pkcs:
                        oi = porders.index(r["order"])
                        ki = pkcs.index(r["kc"])
                        g[oi, ki] = r[key]
                return g

            dm_grid = _pgrid("recovered_dm")
            unc_grid = _pgrid("formal_unc")
            ref_rows = [r for r in prow if "_ref_dm" in r and r["_ref_dm"] is not None]
            ref = ref_rows[0] if ref_rows else None
            peak_data.append({
                "porders": porders, "pkcs": pkcs,
                "dm_grid": dm_grid, "unc_grid": unc_grid,
                "ref": ref, "label": plabel,
            })

        # Shared colour limits across all peaks
        all_dm_finite = np.concatenate([d["dm_grid"][np.isfinite(d["dm_grid"])]
                                        for d in peak_data if d is not None])
        all_unc_finite = np.concatenate([d["unc_grid"][np.isfinite(d["unc_grid"])]
                                         for d in peak_data if d is not None])
        dm_centre = input_dm
        dm_half = max(abs(np.min(all_dm_finite) - dm_centre),
                      abs(np.max(all_dm_finite) - dm_centre)) * 1.05 if len(all_dm_finite) else 1.0
        unc_vmax = np.max(all_unc_finite) * 1.05 if len(all_unc_finite) else 1.0

        fig, axes = plt.subplots(n_peaks, 2, figsize=(14, 5 * n_peaks),
                                 squeeze=False)
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        for row_idx, pd in enumerate(peak_data):
            if pd is None:
                axes[row_idx, 0].set_visible(False)
                axes[row_idx, 1].set_visible(False)
                continue

            porders, pkcs = pd["porders"], pd["pkcs"]
            dm_grid, unc_grid = pd["dm_grid"], pd["unc_grid"]
            label = pd["label"]
            tag = f" — {label}" if label else ""

            kc_step = pkcs[1] - pkcs[0] if len(pkcs) > 1 else 1
            img_ext = [pkcs[0] - 0.5 * kc_step, pkcs[-1] + 0.5 * kc_step,
                       porders[0] - 0.5, porders[-1] + 0.5]

            # Recovered DM
            ax_dm = axes[row_idx, 0]
            if np.any(np.isfinite(dm_grid)):
                im_dm = ax_dm.imshow(dm_grid, aspect="auto", origin="lower",
                                     cmap="PuOr",
                                     vmin=dm_centre - dm_half, vmax=dm_centre + dm_half,
                                     extent=img_ext)
                plt.colorbar(im_dm, ax=ax_dm, label=r"DM [pc cm$^{-3}$]")
            ax_dm.set_ylabel("Order O")
            ax_dm.set_title(f"Recovered DM{tag}")

            # Formal uncertainty
            ax_unc = axes[row_idx, 1]
            if np.any(np.isfinite(unc_grid)):
                im_unc = ax_unc.imshow(unc_grid, aspect="auto", origin="lower",
                                       cmap="Reds", vmin=0, vmax=unc_vmax,
                                       extent=img_ext)
                plt.colorbar(im_unc, ax=ax_unc, label=r"$\sigma$ [pc cm$^{-3}$]")
            ax_unc.set_title(f"Formal uncertainty{tag}")

            # x-label only on bottom row
            if row_idx == n_peaks - 1:
                ax_dm.set_xlabel("kc")
                ax_unc.set_xlabel("kc")

            # Overlay default SHRINE reference
            ref = pd["ref"]
            if ref is not None:
                ref_kc = ref["_ref_kc"]
                ref_dm = ref["_ref_dm"]
                ref_unc = ref["_ref_unc"]

                star_kc = ref_kc
                kc_in_grid = pkcs[0] <= ref_kc <= pkcs[-1]
                if not kc_in_grid:
                    star_kc = pkcs[0] if ref_kc < pkcs[0] else pkcs[-1]

                for ax in (ax_dm, ax_unc):
                    ax.axhline(y=3, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
                    if kc_in_grid:
                        ax.axvline(x=ref_kc, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
                    ax.plot(star_kc, 3, marker="*", markersize=20, color="red",
                            markeredgecolor="white", markeredgewidth=1.0, zorder=10)

                kc_note = f"  (swept {pkcs[0]}-{pkcs[-1]})" if not kc_in_grid else ""

                # Pick annotation offset so the box stays inside the axes
                kc_frac = (star_kc - pkcs[0]) / (pkcs[-1] - pkcs[0]) if len(pkcs) > 1 else 0.5
                o_frac = (3 - porders[0]) / (porders[-1] - porders[0]) if len(porders) > 1 else 0.5
                x_off = -50 if kc_frac > 0.6 else 20
                y_off = -50 if o_frac > 0.6 else 20

                ax_dm.annotate(
                    f"O=3, kc={ref_kc}\nDM={ref_dm:.4f}\n"
                    + r"$\sigma=$" + f"{ref_unc:.4f}",
                    xy=(star_kc, 3), xycoords="data",
                    xytext=(x_off, y_off), textcoords="offset points",
                    fontsize=8, color="red", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.85),
                )
                print(f"  Reference {label or 'single'}: O=3, kc={ref_kc}, "
                      f"DM={ref_dm:.6f}, unc={ref_unc:.6f}{kc_note}")

        plt.savefig(output_dir / f"real_sweep.{ext}", dpi=150)
        plt.close()
        print(f"Saved {output_dir / f'real_sweep.{ext}'}")


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

    synth_rows, real_rows = [], []

    if args.mode in ("synthetic", "both"):
        synth_rows = run_synthetic_sweep(args, output_dir)
        save_csv(synth_rows, output_dir, "synthetic_sweep.csv")

    if args.mode in ("real", "both"):
        real_rows = run_real_sweep(args, output_dir)
        save_csv(real_rows, output_dir, "real_sweep.csv")

    if args.mode in ("synthetic", "both") and synth_rows:
        plot_sweep_heatmaps(synth_rows, output_dir, args.ext, "synthetic",
                            input_dm=args.injected_dm)

    if args.mode in ("real", "both") and real_rows:
        plot_sweep_heatmaps(real_rows, output_dir, args.ext, "real",
                            input_dm=args.input_dm)

    if args.check_tophat:
        plot_tophat_comparison(output_dir, args.ext)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
