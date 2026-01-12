#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Frequency Axis Utility
==============================

This module provides a standardized frequency axis for the entire project,
ensuring consistency across baseline construction, simulation, feature extraction,
and visualization.

Specifications:
- Single band mode: freq=10MHz→8.2GHz
- Step size: 10MHz
- Number of points: N=820
- freq[i] = 1e7 + i*1e7, for i=0..819
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# Default frequency axis parameters (single band mode)
DEFAULT_START_HZ = 1e7     # 10 MHz
DEFAULT_STEP_HZ = 1e7      # 10 MHz step
DEFAULT_N_POINTS = 820     # 820 points → 10MHz to 8.2GHz


def make_freq_axis(
    start_hz: float = DEFAULT_START_HZ,
    step_hz: float = DEFAULT_STEP_HZ,
    n_points: int = DEFAULT_N_POINTS
) -> np.ndarray:
    """Generate a unified frequency axis.
    
    Creates a frequency array with uniform spacing, starting from start_hz.
    
    Parameters
    ----------
    start_hz : float
        Starting frequency in Hz (default: 1e7 = 10 MHz).
    step_hz : float
        Frequency step size in Hz (default: 1e7 = 10 MHz).
    n_points : int
        Number of frequency points (default: 820).
        
    Returns
    -------
    np.ndarray
        Frequency array of shape (n_points,) with values:
        freq[i] = start_hz + i * step_hz
        
    Examples
    --------
    >>> freq = make_freq_axis()
    >>> len(freq)
    820
    >>> freq[0]
    10000000.0
    >>> freq[-1]
    8200000000.0
    """
    return start_hz + np.arange(n_points, dtype=float) * step_hz


def validate_freq_axis(freq: np.ndarray) -> Tuple[bool, str]:
    """Validate that a frequency axis meets the standard specifications.
    
    Checks:
    - Length is exactly 820 points
    - All differences are positive (strictly increasing)
    - Median difference is approximately 1e7 Hz
    - Starts at approximately 1e7 Hz
    
    Parameters
    ----------
    freq : np.ndarray
        Frequency array to validate.
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) where is_valid indicates if the frequency axis
        meets specifications, and message provides details.
    """
    if len(freq) != DEFAULT_N_POINTS:
        return False, f"Expected {DEFAULT_N_POINTS} points, got {len(freq)}"
    
    diffs = np.diff(freq)
    
    if not np.all(diffs > 0):
        return False, "Frequency axis is not strictly increasing"
    
    median_diff = np.median(diffs)
    expected_step = DEFAULT_STEP_HZ
    tolerance = expected_step * 0.01  # 1% tolerance
    
    if abs(median_diff - expected_step) > tolerance:
        return False, f"Median step {median_diff:.0f} Hz differs from expected {expected_step:.0f} Hz"
    
    if abs(freq[0] - DEFAULT_START_HZ) > tolerance:
        return False, f"Start frequency {freq[0]:.0f} Hz differs from expected {DEFAULT_START_HZ:.0f} Hz"
    
    expected_end = DEFAULT_START_HZ + (DEFAULT_N_POINTS - 1) * DEFAULT_STEP_HZ
    if abs(freq[-1] - expected_end) > tolerance:
        return False, f"End frequency {freq[-1]:.0f} Hz differs from expected {expected_end:.0f} Hz"
    
    return True, "Frequency axis is valid (single band mode, 820 points, 10MHz-8.2GHz)"


def get_freq_hz_to_index(freq: np.ndarray) -> callable:
    """Get a function that maps frequency (Hz) to nearest array index.
    
    Parameters
    ----------
    freq : np.ndarray
        Frequency array.
        
    Returns
    -------
    callable
        Function that takes frequency in Hz and returns nearest index.
    """
    def hz_to_idx(f_hz: float) -> int:
        return int(np.argmin(np.abs(freq - f_hz)))
    return hz_to_idx


def get_default_freq_axis() -> np.ndarray:
    """Get the default project frequency axis.
    
    This is a convenience function that returns the standard frequency
    axis used throughout the project.
    
    Returns
    -------
    np.ndarray
        Standard frequency axis (820 points, 10MHz to 8.2GHz).
    """
    return make_freq_axis()


def align_to_standard_freq(
    freq_original: np.ndarray,
    amp_original: np.ndarray,
    target_freq: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Align amplitude data to standard frequency axis via interpolation.
    
    Parameters
    ----------
    freq_original : np.ndarray
        Original frequency array.
    amp_original : np.ndarray
        Original amplitude array corresponding to freq_original.
    target_freq : np.ndarray, optional
        Target frequency axis. If None, uses the default standard axis.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (target_freq, aligned_amp) - the target frequency axis and
        interpolated amplitude values.
    """
    from scipy.interpolate import interp1d
    
    if target_freq is None:
        target_freq = get_default_freq_axis()
    
    interp_func = interp1d(
        freq_original, amp_original,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    aligned_amp = interp_func(target_freq)
    
    return target_freq, aligned_amp


# Module-level constants for easy import
FREQ_START_HZ = DEFAULT_START_HZ
FREQ_STEP_HZ = DEFAULT_STEP_HZ
FREQ_N_POINTS = DEFAULT_N_POINTS
FREQ_END_HZ = DEFAULT_START_HZ + (DEFAULT_N_POINTS - 1) * DEFAULT_STEP_HZ  # 8.2 GHz


if __name__ == "__main__":
    # Self-test
    freq = make_freq_axis()
    is_valid, msg = validate_freq_axis(freq)
    print(f"Validation: {msg}")
    print(f"Length: {len(freq)}")
    print(f"Start: {freq[0]:.0f} Hz = {freq[0]/1e6:.0f} MHz")
    print(f"End: {freq[-1]:.0f} Hz = {freq[-1]/1e9:.2f} GHz")
    print(f"Step: {np.median(np.diff(freq)):.0f} Hz = {np.median(np.diff(freq))/1e6:.0f} MHz")
