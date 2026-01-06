"""Feature pool generation for comparison methods.

This module creates a broader feature set (Pool features) from the same frequency 
response curves that the knowledge-driven method uses. This ensures fair comparison
where different methods can use different feature subsets from the same raw data.

The pool includes:
- amplitude_global: global amplitude statistics
- frequency_scale: frequency-domain features
- noise_ripple: noise and ripple characteristics
- switching: switching/transition features
- band_local: local band statistics
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def build_feature_pool_from_curve(freq: List[float], amp: List[float]) -> Dict[str, float]:
    """Build broad feature pool from frequency response curve.
    
    Args:
        freq: Frequency points
        amp: Amplitude values
        
    Returns:
        Dictionary with ~30-50 features across multiple groups
    """
    if not freq or not amp or len(freq) != len(amp):
        return _empty_feature_pool()
    
    freq_arr = np.array(freq)
    amp_arr = np.array(amp)
    n = len(amp_arr)
    
    features = {}
    
    # ========== Amplitude Global Features ==========
    features['amp_mean'] = float(np.mean(amp_arr))
    features['amp_std'] = float(np.std(amp_arr))
    features['amp_min'] = float(np.min(amp_arr))
    features['amp_max'] = float(np.max(amp_arr))
    features['amp_range'] = features['amp_max'] - features['amp_min']
    features['amp_median'] = float(np.median(amp_arr))
    features['amp_q25'] = float(np.percentile(amp_arr, 25))
    features['amp_q75'] = float(np.percentile(amp_arr, 75))
    features['amp_iqr'] = features['amp_q75'] - features['amp_q25']
    features['amp_skewness'] = _safe_skewness(amp_arr)
    features['amp_kurtosis'] = _safe_kurtosis(amp_arr)
    
    # ========== Frequency Scale Features ==========
    features['freq_min'] = float(np.min(freq_arr))
    features['freq_max'] = float(np.max(freq_arr))
    features['freq_span'] = features['freq_max'] - features['freq_min']
    freq_steps = np.diff(freq_arr)
    features['freq_step_mean'] = float(np.mean(freq_steps)) if len(freq_steps) > 0 else 0.0
    features['freq_step_std'] = float(np.std(freq_steps)) if len(freq_steps) > 0 else 0.0
    features['freq_step_cv'] = features['freq_step_std'] / (features['freq_step_mean'] + 1e-12)
    
    # ========== Noise & Ripple Features ==========
    # Detrend and compute ripple
    if n > 3:
        # Linear trend removal
        p = np.polyfit(freq_arr, amp_arr, 1)
        trend = np.polyval(p, freq_arr)
        detrended = amp_arr - trend
        features['ripple_var'] = float(np.var(detrended))
        features['ripple_std'] = float(np.std(detrended))
        features['ripple_max_dev'] = float(np.max(np.abs(detrended)))
        features['trend_slope'] = float(p[0])
        features['trend_intercept'] = float(p[1])
    else:
        features['ripple_var'] = 0.0
        features['ripple_std'] = 0.0
        features['ripple_max_dev'] = 0.0
        features['trend_slope'] = 0.0
        features['trend_intercept'] = float(np.mean(amp_arr))
    
    # High-frequency noise estimation (differences)
    amp_diffs = np.diff(amp_arr)
    features['noise_level'] = float(np.std(amp_diffs)) if len(amp_diffs) > 0 else 0.0
    features['noise_peak'] = float(np.max(np.abs(amp_diffs))) if len(amp_diffs) > 0 else 0.0
    
    # ========== Switching/Transition Features ==========
    # Count zero crossings of first derivative
    if len(amp_diffs) > 1:
        sign_changes = np.sum(np.diff(np.sign(amp_diffs)) != 0)
        features['switching_rate'] = float(sign_changes / len(amp_diffs))
    else:
        features['switching_rate'] = 0.0
    
    # ========== Band-Local Features ==========
    # Divide into 4 bands and compute local statistics
    band_size = max(1, n // 4)
    for i, band_name in enumerate(['band1', 'band2', 'band3', 'band4']):
        start_idx = i * band_size
        end_idx = min((i + 1) * band_size, n)
        if start_idx >= end_idx:
            band_amp = amp_arr[-band_size:] if n > 0 else np.array([0.0])
        else:
            band_amp = amp_arr[start_idx:end_idx]
        
        features[f'{band_name}_mean'] = float(np.mean(band_amp))
        features[f'{band_name}_std'] = float(np.std(band_amp))
        features[f'{band_name}_max'] = float(np.max(band_amp))
        features[f'{band_name}_min'] = float(np.min(band_amp))
    
    # Cross-band features
    features['band_consistency'] = _band_consistency([
        features['band1_mean'], features['band2_mean'], 
        features['band3_mean'], features['band4_mean']
    ])
    
    # ========== Spectral Shape Features ==========
    # High-frequency attenuation slope (last 25% vs first 25%)
    anchor_idx = max(1, n // 4)
    if n > anchor_idx:
        features['hf_attenuation_slope'] = (amp_arr[-1] - amp_arr[anchor_idx]) / (freq_arr[-1] - freq_arr[anchor_idx] + 1e-12)
    else:
        features['hf_attenuation_slope'] = 0.0
    
    # Energy in different bands
    band1_energy = float(np.sum(band_amp**2)) if len(band_amp) > 0 else 0.0
    total_energy = float(np.sum(amp_arr**2)) + 1e-12
    features['band1_energy_ratio'] = band1_energy / total_energy
    
    # ========== Also include knowledge-driven features for compatibility ==========
    features['bias'] = abs(features['amp_mean'])
    features['X1'] = features['bias']
    features['X2'] = features['ripple_var']
    features['X3'] = abs(features['hf_attenuation_slope'])
    features['X4'] = abs(features['freq_step_std'])
    features['scale_consistency'] = features['amp_range'] / (abs(features['amp_mean']) + 1e-6)
    features['X5'] = features['scale_consistency']
    features['df'] = features['freq_step_std']
    features['res_slope'] = features['hf_attenuation_slope']
    
    return features


def _safe_skewness(arr: np.ndarray) -> float:
    """Compute skewness safely."""
    if len(arr) < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute kurtosis safely."""
    if len(arr) < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


def _band_consistency(band_means: List[float]) -> float:
    """Measure consistency across bands (lower = more consistent)."""
    if not band_means:
        return 0.0
    return float(np.std(band_means) / (np.mean(band_means) + 1e-6))


def _empty_feature_pool() -> Dict[str, float]:
    """Return empty feature pool with all keys set to 0."""
    features = {}
    for key in ['amp_mean', 'amp_std', 'amp_min', 'amp_max', 'amp_range', 'amp_median',
                'amp_q25', 'amp_q75', 'amp_iqr', 'amp_skewness', 'amp_kurtosis',
                'freq_min', 'freq_max', 'freq_span', 'freq_step_mean', 'freq_step_std', 'freq_step_cv',
                'ripple_var', 'ripple_std', 'ripple_max_dev', 'trend_slope', 'trend_intercept',
                'noise_level', 'noise_peak', 'switching_rate',
                'band1_mean', 'band1_std', 'band1_max', 'band1_min',
                'band2_mean', 'band2_std', 'band2_max', 'band2_min',
                'band3_mean', 'band3_std', 'band3_max', 'band3_min',
                'band4_mean', 'band4_std', 'band4_max', 'band4_min',
                'band_consistency', 'hf_attenuation_slope', 'band1_energy_ratio',
                'bias', 'X1', 'X2', 'X3', 'X4', 'X5', 'scale_consistency', 'df', 'res_slope']:
        features[key] = 0.0
    return features


def read_curve_csv(path: Path) -> Tuple[List[float], List[float]]:
    """Read frequency response curve from CSV file.
    
    Args:
        path: Path to CSV file with freq, amp columns
        
    Returns:
        (freq_list, amp_list)
    """
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            with path.open('r', encoding=enc) as f:
                reader = csv.reader(f)
                rows = list(reader)
            break
        except Exception:
            rows = []
            continue
    
    if not rows:
        return [], []
    
    # Skip header if exists
    if rows and rows[0] and not _is_numeric(rows[0][0]):
        rows = rows[1:]
    
    freq_list = []
    amp_list = []
    for row in rows:
        if len(row) < 2:
            continue
        try:
            freq_list.append(float(row[0]))
            amp_list.append(float(row[1]))
        except (ValueError, IndexError):
            continue
    
    return freq_list, amp_list


def _is_numeric(s: str) -> bool:
    """Check if string is numeric."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def augment_features_with_pool(base_features: Dict[str, float], 
                                raw_curve_path: Path = None) -> Dict[str, float]:
    """Augment existing features with pool features.
    
    If raw curve is available, extract full pool. Otherwise, create
    derived pool features from existing base features.
    
    Args:
        base_features: Existing features dict
        raw_curve_path: Optional path to raw curve CSV
        
    Returns:
        Augmented feature dict
    """
    if raw_curve_path and raw_curve_path.exists():
        freq, amp = read_curve_csv(raw_curve_path)
        pool = build_feature_pool_from_curve(freq, amp)
        # Merge, preferring pool values
        return {**base_features, **pool}
    else:
        # Create synthetic pool from base features
        pool = _synthesize_pool_from_base(base_features)
        return {**base_features, **pool}


def _synthesize_pool_from_base(base: Dict[str, float]) -> Dict[str, float]:
    """Create synthetic pool features from base features when curve unavailable."""
    pool = {}
    
    # Extract base values
    bias = base.get('bias', base.get('X1', 0.0))
    ripple = base.get('ripple_var', base.get('X2', 0.0))
    slope = base.get('res_slope', base.get('X3', 0.0))
    df = base.get('df', base.get('X4', 0.0))
    scale = base.get('scale_consistency', base.get('X5', 0.0))
    
    # Amplitude features (derived)
    pool['amp_mean'] = bias
    pool['amp_std'] = math.sqrt(ripple) if ripple > 0 else 0.0
    pool['amp_range'] = scale * abs(bias) if bias != 0 else scale
    pool['amp_min'] = bias - pool['amp_range'] / 2
    pool['amp_max'] = bias + pool['amp_range'] / 2
    pool['amp_median'] = bias
    pool['amp_q25'] = bias - pool['amp_range'] / 4
    pool['amp_q75'] = bias + pool['amp_range'] / 4
    pool['amp_iqr'] = pool['amp_range'] / 2
    pool['amp_skewness'] = 0.0
    pool['amp_kurtosis'] = 0.0
    
    # Frequency features
    pool['freq_step_mean'] = 1e6  # Assume 1 MHz typical step
    pool['freq_step_std'] = df
    pool['freq_step_cv'] = df / (pool['freq_step_mean'] + 1e-12)
    
    # Noise & ripple
    pool['ripple_var'] = ripple
    pool['ripple_std'] = math.sqrt(ripple) if ripple > 0 else 0.0
    pool['ripple_max_dev'] = pool['ripple_std'] * 2.5
    pool['trend_slope'] = slope
    pool['trend_intercept'] = bias
    pool['noise_level'] = pool['ripple_std'] * 0.5
    pool['noise_peak'] = pool['ripple_std'] * 3.0
    
    # Switching
    pool['switching_rate'] = min(1.0, abs(slope) * 1e10)
    
    # Bands (rough approximation)
    for i in range(1, 5):
        band_offset = (i - 2.5) * 0.1 * bias
        pool[f'band{i}_mean'] = bias + band_offset
        pool[f'band{i}_std'] = pool['amp_std'] * 0.5
        pool[f'band{i}_max'] = pool[f'band{i}_mean'] + pool[f'band{i}_std']
        pool[f'band{i}_min'] = pool[f'band{i}_mean'] - pool[f'band{i}_std']
    
    pool['band_consistency'] = scale
    pool['hf_attenuation_slope'] = slope
    pool['band1_energy_ratio'] = 0.25
    
    # Keep original features
    for k in ['bias', 'X1', 'X2', 'X3', 'X4', 'X5', 'scale_consistency', 'df', 'res_slope']:
        if k in base:
            pool[k] = base[k]
    
    return pool
