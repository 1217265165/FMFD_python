"""
RRS (Reference Response Spectrum) and Dynamic Envelope Implementation

This module implements a robust baseline and envelope computation for
single-band spectrum data (10 MHz → 8.2 GHz, 820 points).

Key design principles:
1. RRS uses pointwise median (no secondary smoothing) - preserves real curve detail
2. Envelope uses vendor tolerance as minimum width, with residual-based expansion
3. Global offset drift (±0.4 dB) is absorbed as normal variation to avoid false positives
4. Only envelope width is smoothed (not RRS)

Vendor tolerance by frequency band:
- 10 MHz ~ 100 MHz: ±0.80 dB
- 100 MHz ~ 3.25 GHz: ±0.40 dB
- 3.25 GHz ~ 5.25 GHz: ±0.60 dB
- 5.25 GHz ~ 8.2 GHz: ±0.80 dB
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =============================================================================
# Frequency Axis Construction
# =============================================================================
def build_frequency_axis_hz(start_hz=10e6, stop_hz=8.2e9, step_hz=10e6) -> np.ndarray:
    """Build frequency axis from 10 MHz to 8.2 GHz with 10 MHz step.
    
    Returns
    -------
    np.ndarray
        Frequency axis in Hz, shape (820,) for default parameters.
    """
    n = int(round((stop_hz - start_hz) / step_hz)) + 1
    return start_hz + step_hz * np.arange(n, dtype=np.float64)


# =============================================================================
# Vendor Tolerance by Frequency Segment
# =============================================================================
def vendor_tolerance_db(freq_hz: np.ndarray) -> np.ndarray:
    """Get vendor specification tolerance (dB) for each frequency point.
    
    Based on equipment specifications:
    - 10 MHz ~ 100 MHz: ±0.80 dB
    - 100 MHz ~ 3.25 GHz: ±0.40 dB
    - 3.25 GHz ~ 5.25 GHz: ±0.60 dB
    - 5.25 GHz ~ 8.2 GHz: ±0.80 dB
    
    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency axis in Hz.
        
    Returns
    -------
    np.ndarray
        Tolerance in dB for each frequency point.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    tol = np.full_like(f, 0.80, dtype=np.float64)
    
    tol[(f >= 10e6) & (f < 100e6)] = 0.80
    tol[(f >= 100e6) & (f < 3.25e9)] = 0.40
    tol[(f >= 3.25e9) & (f < 5.25e9)] = 0.60
    tol[(f >= 5.25e9) & (f <= 8.2e9)] = 0.80
    
    return tol


# =============================================================================
# Helper Functions for Smoothing
# =============================================================================
def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average with edge padding."""
    if win <= 1:
        return x.copy()
    win = int(win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(win, dtype=np.float64) / win
    return np.convolve(xp, w, mode="valid")[:len(x)]


def _smooth_1d(x: np.ndarray, *, gaussian_sigma_bins: float = 0.0,
               moving_avg_win: int = 0) -> np.ndarray:
    """Smooth 1D array using Gaussian and/or moving average.
    
    Parameters
    ----------
    x : np.ndarray
        Input array.
    gaussian_sigma_bins : float
        Gaussian filter sigma in bins (0 = disabled).
    moving_avg_win : int
        Moving average window size (0 = disabled).
        
    Returns
    -------
    np.ndarray
        Smoothed array.
    """
    out = x.copy()
    
    if gaussian_sigma_bins > 0 and _HAS_SCIPY:
        out = gaussian_filter1d(out, sigma=gaussian_sigma_bins, mode='reflect')
    
    if moving_avg_win > 1:
        out = _moving_average(out, moving_avg_win)
    
    return out


# =============================================================================
# Global Offset Estimation (Per-Trace Drift Removal)
# =============================================================================
def estimate_global_offset(trace: np.ndarray, rrs: np.ndarray,
                          max_offset_db: float = 0.4) -> float:
    """Estimate global offset of a single trace relative to RRS.
    
    This helps absorb time-dependent drift as normal variation,
    avoiding false positives for reference level errors.
    
    Parameters
    ----------
    trace : np.ndarray
        Single frequency response trace.
    rrs : np.ndarray
        Reference response spectrum.
    max_offset_db : float
        Maximum allowed offset (clips to ±max_offset_db).
        
    Returns
    -------
    float
        Estimated global offset in dB (clipped).
    """
    # Use robust median of difference
    diff = trace - rrs
    offset = np.median(diff)
    
    # Clip to maximum allowed offset
    offset = np.clip(offset, -max_offset_db, max_offset_db)
    
    return float(offset)


def remove_global_offsets(traces: np.ndarray, rrs: np.ndarray,
                         max_offset_db: float = 0.4) -> tuple:
    """Remove global offset from each trace for residual computation.
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points).
    rrs : np.ndarray
        Reference response spectrum.
    max_offset_db : float
        Maximum allowed offset.
        
    Returns
    -------
    tuple
        (corrected_traces, offsets)
    """
    n_traces = traces.shape[0]
    offsets = np.zeros(n_traces)
    corrected = np.zeros_like(traces)
    
    for i in range(n_traces):
        offset = estimate_global_offset(traces[i], rrs, max_offset_db)
        offsets[i] = offset
        corrected[i] = traces[i] - offset
    
    return corrected, offsets


# =============================================================================
# RRS Computation (Pointwise Median, No Secondary Smoothing)
# =============================================================================
def compute_rrs(traces: np.ndarray, smooth_sigma: float = 0.0) -> np.ndarray:
    """Compute RRS using pointwise median.
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points), normal response curves.
    smooth_sigma : float
        Optional Gaussian smoothing sigma (default 0 = no smoothing).
        
    Returns
    -------
    np.ndarray
        RRS (Reference Response Spectrum), shape (n_points,).
    """
    rrs = np.median(traces, axis=0)
    
    if smooth_sigma > 0 and _HAS_SCIPY:
        rrs = gaussian_filter1d(rrs, sigma=smooth_sigma, mode='reflect')
    
    return rrs


# =============================================================================
# Robust Sigma Estimation (MAD-based)
# =============================================================================
def compute_robust_sigma_from_residuals(residuals: np.ndarray,
                                        smooth_sigma: float = 6.0) -> np.ndarray:
    """Compute robust sigma for each frequency point from residuals.
    
    Uses MAD (Median Absolute Deviation) for robustness against outliers.
    
    Parameters
    ----------
    residuals : np.ndarray
        Shape (n_traces, n_points), residuals after offset removal.
    smooth_sigma : float
        Gaussian smoothing sigma for sigma estimate.
        
    Returns
    -------
    np.ndarray
        Robust sigma estimate for each frequency point.
    """
    n_points = residuals.shape[1]
    sigma_raw = np.zeros(n_points)
    
    for j in range(n_points):
        res_j = residuals[:, j]
        mad = np.median(np.abs(res_j - np.median(res_j)))
        sigma_raw[j] = 1.4826 * mad  # Scale factor for normal distribution
    
    # Ensure minimum sigma
    sigma_raw = np.maximum(sigma_raw, 1e-6)
    
    # Smooth sigma estimate
    if smooth_sigma > 0 and _HAS_SCIPY:
        sigma_smooth = gaussian_filter1d(sigma_raw, sigma=smooth_sigma, mode='reflect')
    else:
        sigma_smooth = sigma_raw
    
    return sigma_smooth


# =============================================================================
# Envelope Width Computation (Vendor Tolerance as Minimum)
# =============================================================================
def compute_envelope_width(sigma: np.ndarray, freq_hz: np.ndarray,
                          k: float = 2.5,
                          use_vendor_floor: bool = True,
                          smooth_sigma: float = 8.0,
                          width_max_db: float = 1.0) -> np.ndarray:
    """Compute envelope half-width.
    
    Parameters
    ----------
    sigma : np.ndarray
        Robust sigma estimate from residuals.
    freq_hz : np.ndarray
        Frequency axis in Hz.
    k : float
        Scale factor for sigma (width = k * sigma).
    use_vendor_floor : bool
        If True, use vendor tolerance as minimum width.
    smooth_sigma : float
        Gaussian smoothing sigma for width.
    width_max_db : float
        Maximum allowed half-width.
        
    Returns
    -------
    np.ndarray
        Envelope half-width in dB.
    """
    # Base width from residual sigma
    half_width = k * sigma
    
    # Apply vendor tolerance as minimum (floor)
    if use_vendor_floor:
        vendor_tol = vendor_tolerance_db(freq_hz)
        half_width = np.maximum(half_width, vendor_tol)
    
    # Apply maximum cap
    half_width = np.minimum(half_width, width_max_db)
    
    # Smooth width (this is what makes envelope slowly varying)
    if smooth_sigma > 0 and _HAS_SCIPY:
        half_width = gaussian_filter1d(half_width, sigma=smooth_sigma, mode='reflect')
    
    return half_width


# =============================================================================
# Coverage Computation
# =============================================================================
def compute_coverage(traces: np.ndarray, upper: np.ndarray, 
                    lower: np.ndarray) -> dict:
    """Compute coverage statistics.
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points).
    upper, lower : np.ndarray
        Envelope boundaries.
        
    Returns
    -------
    dict
        Coverage statistics.
    """
    n_traces, n_points = traces.shape
    
    # Per-trace coverage
    trace_coverages = []
    for i in range(n_traces):
        in_bounds = (traces[i] >= lower) & (traces[i] <= upper)
        trace_coverages.append(np.mean(in_bounds))
    
    # Per-point coverage
    point_coverages = []
    for j in range(n_points):
        col = traces[:, j]
        in_bounds = (col >= lower[j]) & (col <= upper[j])
        point_coverages.append(np.mean(in_bounds))
    
    return {
        'coverage_mean': float(np.mean(trace_coverages)),
        'coverage_min': float(np.min(trace_coverages)),
        'coverage_per_trace': trace_coverages,
        'coverage_per_point': point_coverages,
        'coverage_point_5th': float(np.percentile(point_coverages, 5)),
        'coverage_point_50th': float(np.percentile(point_coverages, 50)),
        'coverage_point_95th': float(np.percentile(point_coverages, 95)),
    }


# =============================================================================
# Main Function: Build RRS and Dynamic Envelope
# =============================================================================
def build_rrs_and_envelope(
    freq_hz: np.ndarray,
    traces: np.ndarray,
    *,
    # RRS parameters
    rrs_smooth_sigma: float = 0.0,  # Default: no smoothing for RRS
    # Global offset parameters
    absorb_global_offset: bool = True,
    max_offset_db: float = 0.4,
    # Envelope parameters
    k_sigma: float = 2.5,
    use_vendor_floor: bool = True,
    sigma_smooth: float = 6.0,
    width_smooth: float = 8.0,
    width_max_db: float = 1.0,
    # Coverage target
    target_coverage: float = 0.97,
) -> dict:
    """Build RRS and dynamic envelope from normal traces.
    
    Main entry point for baseline construction. Implements:
    1. RRS via pointwise median (no secondary smoothing by default)
    2. Global offset absorption to handle time drift
    3. Vendor tolerance as envelope floor
    4. Only smooth envelope width (not RRS)
    
    Parameters
    ----------
    freq_hz : np.ndarray
        Frequency axis in Hz.
    traces : np.ndarray
        Shape (n_traces, n_points), normal response curves.
    rrs_smooth_sigma : float
        Gaussian sigma for RRS smoothing (0 = disabled).
    absorb_global_offset : bool
        If True, remove global offset from each trace before residual computation.
    max_offset_db : float
        Maximum offset to absorb as normal drift.
    k_sigma : float
        Scale factor for sigma in envelope width.
    use_vendor_floor : bool
        If True, use vendor tolerance as minimum envelope width.
    sigma_smooth : float
        Gaussian sigma for smoothing residual-based sigma.
    width_smooth : float
        Gaussian sigma for smoothing final envelope width.
    width_max_db : float
        Maximum allowed envelope half-width.
    target_coverage : float
        Target coverage for envelope expansion.
        
    Returns
    -------
    dict
        Contains: rrs, upper, lower, coverage, metadata
    """
    n_traces, n_points = traces.shape
    print(f"[RRS/Envelope] Building from {n_traces} normal traces, {n_points} points")
    
    # Step 1: Compute RRS (pointwise median, no smoothing by default)
    rrs = compute_rrs(traces, smooth_sigma=rrs_smooth_sigma)
    print(f"[RRS/Envelope] RRS computed (smooth_sigma={rrs_smooth_sigma})")
    
    # Step 2: Estimate and remove global offsets
    if absorb_global_offset:
        corrected_traces, offsets = remove_global_offsets(traces, rrs, max_offset_db)
        print(f"[RRS/Envelope] Global offsets: mean={np.mean(offsets):.4f}, "
              f"std={np.std(offsets):.4f}, max_abs={np.max(np.abs(offsets)):.4f} dB")
    else:
        corrected_traces = traces
        offsets = np.zeros(n_traces)
    
    # Step 3: Compute residuals and robust sigma
    residuals = corrected_traces - rrs
    sigma = compute_robust_sigma_from_residuals(residuals, smooth_sigma=sigma_smooth)
    print(f"[RRS/Envelope] Sigma: min={sigma.min():.4f}, median={np.median(sigma):.4f}, "
          f"max={sigma.max():.4f} dB")
    
    # Step 4: Compute envelope width with vendor tolerance floor
    half_width = compute_envelope_width(
        sigma, freq_hz,
        k=k_sigma,
        use_vendor_floor=use_vendor_floor,
        smooth_sigma=width_smooth,
        width_max_db=width_max_db
    )
    
    # Step 5: Build envelope
    upper = rrs + half_width
    lower = rrs - half_width
    
    # Step 6: Check and expand coverage if needed
    coverage = compute_coverage(traces, upper, lower)
    
    if coverage['coverage_mean'] < target_coverage:
        print(f"[RRS/Envelope] Coverage {coverage['coverage_mean']:.4f} < target {target_coverage}, expanding...")
        
        # Iteratively expand until target coverage
        for extra_k in np.arange(0.1, 2.0, 0.1):
            half_width_exp = compute_envelope_width(
                sigma, freq_hz,
                k=k_sigma + extra_k,
                use_vendor_floor=use_vendor_floor,
                smooth_sigma=width_smooth,
                width_max_db=width_max_db + 0.2  # Allow slightly larger max
            )
            upper = rrs + half_width_exp
            lower = rrs - half_width_exp
            coverage = compute_coverage(traces, upper, lower)
            
            if coverage['coverage_mean'] >= target_coverage:
                half_width = half_width_exp
                k_sigma = k_sigma + extra_k
                print(f"[RRS/Envelope] Expanded to k={k_sigma:.2f}, coverage={coverage['coverage_mean']:.4f}")
                break
    
    # Compute final statistics
    width = upper - lower
    print(f"[RRS/Envelope] Width: min={width.min():.4f}, median={np.median(width):.4f}, "
          f"max={width.max():.4f} dB")
    print(f"[RRS/Envelope] Final coverage: mean={coverage['coverage_mean']:.4f}, "
          f"min={coverage['coverage_min']:.4f}")
    
    # Build result
    result = {
        'frequency_hz': freq_hz,
        'rrs': rrs,
        'upper': upper,
        'lower': lower,
        'coverage': coverage,
        'metadata': {
            'n_traces': n_traces,
            'n_points': n_points,
            'rrs_smooth_sigma': rrs_smooth_sigma,
            'absorb_global_offset': absorb_global_offset,
            'max_offset_db': max_offset_db,
            'offsets': offsets.tolist() if absorb_global_offset else [],
            'k_sigma': k_sigma,
            'use_vendor_floor': use_vendor_floor,
            'sigma_smooth': sigma_smooth,
            'width_smooth': width_smooth,
            'width_max_db': width_max_db,
            'target_coverage': target_coverage,
            'vendor_tolerance': vendor_tolerance_db(freq_hz).tolist(),
        },
        # For frontend display
        'center_level_db': float(np.median(rrs)),
        'spec_center_db': -10.0,
        'spec_tol_db': 0.4,
        'spec_upper_db': -9.6,
        'spec_lower_db': -10.4,
    }
    
    return result


# =============================================================================
# Wrapper for Compatibility with Existing Code
# =============================================================================
def compute_rrs_bounds_v2(frequency, traces, target_coverage=0.97):
    """Wrapper for compatibility with existing baseline.py interface.
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis.
    traces : np.ndarray
        Shape (n_traces, n_points).
    target_coverage : float
        Target coverage.
        
    Returns
    -------
    tuple
        (rrs, (upper, lower), coverage_info)
    """
    result = build_rrs_and_envelope(
        frequency, traces,
        target_coverage=target_coverage,
    )
    
    coverage_info = result['coverage']
    coverage_info.update({
        'k_final': result['metadata']['k_sigma'],
        'target_coverage_mean': target_coverage,
        'width_min': float(np.min(result['upper'] - result['lower'])),
        'width_median': float(np.median(result['upper'] - result['lower'])),
        'width_max': float(np.max(result['upper'] - result['lower'])),
        'center_level_db': result['center_level_db'],
        'spec_center_db': result['spec_center_db'],
        'spec_tol_db': result['spec_tol_db'],
    })
    
    return result['rrs'], (result['upper'], result['lower']), coverage_info
