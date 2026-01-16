import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from .config import (
    BAND_RANGES,
    K_LIST,
    N_POINTS,
    SINGLE_BAND_MODE,
    COVERAGE_MEAN_MIN,
    COVERAGE_MIN_MIN,
    FREQ_START_HZ,
    FREQ_STEP_HZ,
)

# ============ Baseline / Envelope configuration ============
# RRS smoothing (default off)
RRS_SMOOTH_ENABLED = False
RRS_SMOOTH_WINDOW = 15
RRS_SMOOTH_POLY = 3

# Quantile envelope search
QUANTILE_COVERAGE_GRID = np.arange(0.94, 0.996, 0.004)
SLIDING_WINDOW_SIZE = 41
SLIDING_COVERAGE_MIN = 0.93

# Smoothing (in Hz)
ENVELOPE_SMOOTH_SIGMA_HZ = 200e6
WIDTH_SMOOTH_SIGMA_HZ = 400e6

# Soft prior width (vendor tolerance guidance)
WIDTH_PRIOR_ALPHA = 0.8

# Coverage expansion
WIDTH_EXPAND_STEP = 0.01
WIDTH_EXPAND_MAX_ITERS = 50


def load_and_align(folder_path, use_spectrum_column=True):
    """
    Load CSV responses and align to a unified frequency grid.

    Parameters
    ----------
    folder_path : str
        Path to folder containing CSV files.
    use_spectrum_column : bool
        If True, uses the second-to-last column (spectrum analyzer reading).
        If False, uses column 1 (power dBm).

    Returns
    -------
    frequency : np.ndarray
    traces : np.ndarray
    file_names : list
    """
    traces = []
    names = []
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            try:
                for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
                    try:
                        df = pd.read_csv(os.path.join(folder_path, f), encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    df = pd.read_csv(os.path.join(folder_path, f), encoding="utf-8", errors="ignore")

                if df.shape[1] >= 2:
                    freq = df.iloc[:, 0].values.astype(float)
                    if use_spectrum_column and df.shape[1] >= 3:
                        amp = df.iloc[:, -2].values.astype(float)
                    else:
                        amp = df.iloc[:, 1].values.astype(float)

                    traces.append((freq, amp))
                    names.append(f)
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
                continue

    if not traces:
        raise FileNotFoundError("未找到有效 CSV 频响数据")

    all_freq = [t[0] for t in traces]
    min_f = max(np.min(f) for f in all_freq)
    max_f = min(np.max(f) for f in all_freq)
    frequency = np.linspace(min_f, max_f, N_POINTS)
    aligned = []
    for freq, amp in traces:
        interp = interp1d(freq, amp, kind="linear", fill_value="extrapolate")
        aligned.append(interp(frequency))
    return frequency, np.vstack(aligned), names


def align_to_frequency(target_frequency, freq, amp):
    """Interpolate a curve to the target frequency grid."""
    interp = interp1d(freq, amp, kind="linear", fill_value="extrapolate")
    return interp(target_frequency)


def compute_coverage(traces, upper, lower):
    """Compute coverage statistics for an envelope."""
    n_traces = traces.shape[0]
    n_points = traces.shape[1]
    coverages = []

    for i in range(n_traces):
        trace = traces[i]
        in_bounds = (trace >= lower) & (trace <= upper)
        coverages.append(np.mean(in_bounds))

    point_coverages = []
    for j in range(n_points):
        col = traces[:, j]
        in_bounds = (col >= lower[j]) & (col <= upper[j])
        point_coverages.append(np.mean(in_bounds))

    return {
        "coverage_mean": float(np.mean(coverages)),
        "coverage_min": float(np.min(coverages)),
        "coverage_per_trace": coverages,
        "coverage_per_point": point_coverages,
        "coverage_point_5th": float(np.percentile(point_coverages, 5)),
        "coverage_point_50th": float(np.percentile(point_coverages, 50)),
        "coverage_point_95th": float(np.percentile(point_coverages, 95)),
    }


def compute_sliding_coverage(traces, upper, lower, window_size=SLIDING_WINDOW_SIZE):
    """Compute sliding window coverage for envelope validation."""
    n_traces, n_points = traces.shape
    n_windows = n_points - window_size + 1
    if n_windows <= 0:
        return np.array([1.0])

    window_coverages = []
    for start in range(n_windows):
        end = start + window_size
        window_in_bounds = 0
        window_total = n_traces * window_size
        upper_window = upper[start:end]
        lower_window = lower[start:end]
        for i in range(n_traces):
            trace_window = traces[i, start:end]
            window_in_bounds += np.sum((trace_window >= lower_window) & (trace_window <= upper_window))
        window_coverages.append(window_in_bounds / window_total)

    return np.array(window_coverages)


def compute_rrs_robust(
    traces,
    smooth=RRS_SMOOTH_ENABLED,
    smooth_window=RRS_SMOOTH_WINDOW,
    smooth_poly=RRS_SMOOTH_POLY,
):
    """Compute RRS as pointwise median with optional very-light smoothing."""
    n_traces, n_points = traces.shape
    rrs = np.median(traces, axis=0)
    if smooth and n_points >= smooth_window:
        rrs = savgol_filter(rrs, smooth_window, smooth_poly, mode="nearest")
    return rrs


def vendor_tolerance_dbm(frequency_hz: np.ndarray) -> np.ndarray:
    """Vendor tolerance (half-width) in dB for each frequency point."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    tol = np.full_like(f, 0.80, dtype=np.float64)
    tol[(f >= 10e6) & (f < 100e6)] = 0.80
    tol[(f >= 100e6) & (f < 3.25e9)] = 0.40
    tol[(f >= 3.25e9) & (f < 5.25e9)] = 0.60
    tol[(f >= 5.25e9) & (f <= 8.2e9)] = 0.80
    return tol


def _sigma_points(frequency_hz: np.ndarray, sigma_hz: float) -> float:
    df = np.median(np.diff(frequency_hz)) if len(frequency_hz) > 1 else FREQ_STEP_HZ
    return float(sigma_hz / df) if df > 0 else 0.0


def _build_quantile_envelope(
    residuals: np.ndarray,
    rrs: np.ndarray,
    vendor_tol: np.ndarray,
    q_low: float,
    q_high: float,
    smooth_sigma_points: float,
    width_smooth_sigma_points: float,
    width_prior_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_raw = rrs + np.quantile(residuals, q_low, axis=0)
    upper_raw = rrs + np.quantile(residuals, q_high, axis=0)

    if smooth_sigma_points > 0:
        lower_raw = gaussian_filter1d(lower_raw, sigma=smooth_sigma_points, mode="reflect")
        upper_raw = gaussian_filter1d(upper_raw, sigma=smooth_sigma_points, mode="reflect")

    width_data = upper_raw - lower_raw
    width_prior = np.maximum(width_data, width_prior_alpha * 2.0 * vendor_tol)

    if width_smooth_sigma_points > 0:
        width_smooth = gaussian_filter1d(width_prior, sigma=width_smooth_sigma_points, mode="reflect")
    else:
        width_smooth = width_prior

    upper = rrs + width_smooth / 2.0
    lower = rrs - width_smooth / 2.0

    return upper, lower, width_smooth, width_data


def compute_quantile_envelope(
    frequency: np.ndarray,
    traces: np.ndarray,
    target_coverage_mean: float = COVERAGE_MEAN_MIN,
    target_coverage_min: float = COVERAGE_MIN_MIN,
    sliding_coverage_min: float = SLIDING_COVERAGE_MIN,
    quantile_grid: np.ndarray = QUANTILE_COVERAGE_GRID,
    smooth_sigma_hz: float = ENVELOPE_SMOOTH_SIGMA_HZ,
    width_smooth_sigma_hz: float = WIDTH_SMOOTH_SIGMA_HZ,
    width_prior_alpha: float = WIDTH_PRIOR_ALPHA,
    rrs_smooth: bool = RRS_SMOOTH_ENABLED,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, float]]:
    n_traces, n_points = traces.shape
    print(f"[Baseline] Computing RRS from {n_traces} traces, {n_points} points")
    print(f"[Baseline] RRS smoothing: {'enabled' if rrs_smooth else 'disabled (pointwise median)'}")

    rrs = compute_rrs_robust(traces, smooth=rrs_smooth)
    pointwise_median = np.median(traces, axis=0)
    rrs_mae = float(np.mean(np.abs(rrs - pointwise_median)))

    residuals = traces - rrs
    vendor_tol = vendor_tolerance_dbm(frequency)

    smooth_sigma_points = _sigma_points(frequency, smooth_sigma_hz)
    width_smooth_sigma_points = _sigma_points(frequency, width_smooth_sigma_hz)

    chosen = None
    coverage = None
    sliding_cov = None
    width = None

    for coverage_target in quantile_grid:
        q_low = (1.0 - coverage_target) / 2.0
        q_high = 1.0 - q_low

        upper, lower, width, _ = _build_quantile_envelope(
            residuals,
            rrs,
            vendor_tol,
            q_low,
            q_high,
            smooth_sigma_points,
            width_smooth_sigma_points,
            width_prior_alpha,
        )

        coverage = compute_coverage(traces, upper, lower)
        sliding_cov = compute_sliding_coverage(traces, upper, lower)
        sliding_cov_min = float(np.min(sliding_cov))

        if (
            coverage["coverage_mean"] >= target_coverage_mean
            and coverage["coverage_min"] >= target_coverage_min
            and sliding_cov_min >= sliding_coverage_min
        ):
            chosen = (q_low, q_high, coverage_target)
            break

    if chosen is None:
        q_low = (1.0 - quantile_grid[-1]) / 2.0
        q_high = 1.0 - q_low
        chosen = (q_low, q_high, float(quantile_grid[-1]))
        upper, lower, width, _ = _build_quantile_envelope(
            residuals,
            rrs,
            vendor_tol,
            q_low,
            q_high,
            smooth_sigma_points,
            width_smooth_sigma_points,
            width_prior_alpha,
        )
        coverage = compute_coverage(traces, upper, lower)
        sliding_cov = compute_sliding_coverage(traces, upper, lower)

    # Expand width if needed to satisfy coverage constraints after smoothing
    expansion_iters = 0
    sliding_cov_min = float(np.min(sliding_cov))
    while (
        coverage["coverage_mean"] < target_coverage_mean
        or coverage["coverage_min"] < target_coverage_min
        or sliding_cov_min < sliding_coverage_min
    ) and expansion_iters < WIDTH_EXPAND_MAX_ITERS:
        expansion_iters += 1
        width = width + WIDTH_EXPAND_STEP
        if width_smooth_sigma_points > 0:
            width = gaussian_filter1d(width, sigma=width_smooth_sigma_points, mode="reflect")
        upper = rrs + width / 2.0
        lower = rrs - width / 2.0
        coverage = compute_coverage(traces, upper, lower)
        sliding_cov = compute_sliding_coverage(traces, upper, lower)
        sliding_cov_min = float(np.min(sliding_cov))

    width_smoothness = float(np.std(np.diff(width))) if width is not None else 0.0

    print(f"[Baseline] RRS vs pointwise median MAE: {rrs_mae:.6f} dB")
    print(
        f"[Baseline] Chosen quantiles: q_low={chosen[0]:.3f}, q_high={chosen[1]:.3f} (coverage target {chosen[2]:.3f})"
    )
    print(
        f"[Baseline] Coverage: mean={coverage['coverage_mean']:.4f}, min={coverage['coverage_min']:.4f}, "
        f"sliding_min={sliding_cov_min:.4f}"
    )
    print(
        f"[Baseline] Width stats: min={width.min():.4f}, median={np.median(width):.4f}, "
        f"max={width.max():.4f} dB, smoothness={width_smoothness:.4f}"
    )

    coverage_info = {
        **coverage,
        "rrs_mae": rrs_mae,
        "rrs_smooth_enabled": rrs_smooth,
        "chosen_quantiles": {
            "q_low": chosen[0],
            "q_high": chosen[1],
            "coverage_target": chosen[2],
        },
        "width_min": float(width.min()),
        "width_median": float(np.median(width)),
        "width_max": float(width.max()),
        "width_smoothness": width_smoothness,
        "sliding_coverage_min": sliding_cov_min,
        "n_normal_traces": n_traces,
        "width_prior_alpha": width_prior_alpha,
        "smooth_params": {
            "rrs_smooth_enabled": rrs_smooth,
            "smooth_sigma_hz": smooth_sigma_hz,
            "width_smooth_sigma_hz": width_smooth_sigma_hz,
        },
        "expansion_iters": expansion_iters,
    }

    return rrs, (upper, lower), coverage_info


def compute_rrs_bounds(
    frequency,
    traces,
    band_ranges=BAND_RANGES,
    k_list=K_LIST,
    validate_coverage=True,
):
    """Compute RRS and envelope bounds.

    In single-band mode, use quantile envelope search.
    """
    if SINGLE_BAND_MODE and validate_coverage:
        return compute_quantile_envelope(frequency, traces)

    assert len(band_ranges) == len(k_list)
    rrs = np.zeros_like(frequency)
    upper = np.zeros_like(frequency)
    lower = np.zeros_like(frequency)
    for (start, end), k in zip(band_ranges, k_list):
        mask = (frequency >= start) & (frequency <= end)
        if not np.any(mask):
            continue
        band = traces[:, mask]
        m = np.mean(band, axis=0)
        s = np.std(band, axis=0)
        rrs[mask] = m
        upper[mask] = m + k * s
        lower[mask] = m - k * s

    coverage = compute_coverage(traces, upper, lower)
    return rrs, (upper, lower), coverage


def detect_switch_steps(frequency, traces, band_ranges=BAND_RANGES, tol=0.2):
    """Detect switching steps; returns empty list in single-band mode."""
    if SINGLE_BAND_MODE:
        return []

    feats = []
    for i in range(len(band_ranges) - 1):
        end_f = band_ranges[i][1]
        next_f = band_ranges[i + 1][0]
        m_end = np.argmin(np.abs(frequency - end_f))
        m_next = np.argmin(np.abs(frequency - next_f))
        current_vals = traces[:, m_end]
        next_vals = traces[:, m_next]
        diffs = next_vals - current_vals
        step_mean = float(np.mean(diffs))
        step_std = float(np.std(diffs))
        is_ok = np.abs(step_mean) <= tol
        feats.append(
            {
                "end_freq": float(frequency[m_end]),
                "start_freq": float(frequency[m_next]),
                "step_mean": step_mean,
                "step_std": step_std,
                "tolerance": tol,
                "is_within_tolerance": bool(is_ok),
            }
        )
    return feats


def build_frequency_axis_hz(n_points: int) -> np.ndarray:
    """Build a frequency axis from config for n_points samples."""
    stop_hz = FREQ_START_HZ + (n_points - 1) * FREQ_STEP_HZ
    return np.linspace(FREQ_START_HZ, stop_hz, n_points)
