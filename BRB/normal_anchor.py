#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-0 Normal Anchor Module (Soft Gating Version)
==================================================

This module implements the first stage of system-level diagnosis:
determining whether a sample is Normal vs Abnormal before proceeding
to fault-type classification.

Key improvements (v2):
- **Dual threshold (T_low, T_high)** with gray zone for soft gating
- **In-envelope anomaly detection** (not just envelope violations)
- **No hard bypass** - always runs three branches with Normal prior
- **Robust zscore** normalization using median/IQR
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NormalAnchorConfig:
    """Configuration for Normal Anchor detection.
    
    Attributes
    ----------
    normal_features : List[str]
        Features used for normal detection.
    T_low : float
        Lower threshold - below this, strong Normal prior.
    T_high : float
        Upper threshold - above this, minimal Normal prior.
    k_normal_prior : float
        Scaling factor for Normal logit prior.
    use_calibration : bool
        Whether to load thresholds from calibration.json.
    """
    normal_features: List[str] = field(default_factory=lambda: [
        'X11', 'X12', 'X13', 'X7', 'X2', 'X16', 'X17', 'X20'
    ])
    T_low: float = 0.10    # Below this = strong normal prior
    T_high: float = 0.35   # Above this = minimal normal prior
    k_normal_prior: float = 4.0  # Scaling for logit_normal
    use_calibration: bool = True


def _get_feature_value(features: Dict[str, float], *names: str, default: float = 0.0) -> float:
    """Get feature value with fallback to aliases."""
    for name in names:
        if name in features:
            return float(features[name])
    return default


def _robust_zscore(value: float, median: float, iqr: float, zcap: float = 5.0) -> float:
    """Compute robust z-score using median and IQR, capped at zcap."""
    if iqr <= 1e-12:
        return 0.0
    z = (value - median) / (iqr * 0.7413 + 1e-12)  # 0.7413 normalizes IQR to std
    return min(max(z, -zcap), zcap)


def compute_anchor_score(features: Dict[str, float], config: NormalAnchorConfig = None) -> Dict[str, float]:
    """Compute normal anchor score using GROUPED MAX fusion.
    
    This function computes a combined score indicating how "abnormal" a sample is.
    Higher scores indicate more anomalous behavior.
    
    v3: Changed from weighted average to GROUPED MAX to prevent
    Freq/Ref being diluted by Amp components.
    
    Groups:
    - amp_evidence: envelope, jump, flatness, kurtosis
    - freq_evidence: freq_shift, warp_scale, warp_bias, warp_residual, phase_slope
    - ref_evidence: compress_slope, scale_consist, high_quantile, piecewise, tail_asym
    
    Parameters
    ----------
    features : dict
        Feature dictionary containing X1-X28 or equivalent.
    config : NormalAnchorConfig, optional
        Configuration for normal detection.
        
    Returns
    -------
    dict
        Contains:
        - anchor_score: Combined anomaly score (higher = more abnormal)
        - score_amp, score_freq, score_ref: Group-level max scores
        - components: Individual component scores
        - T_low, T_high: Applied thresholds
    """
    if config is None:
        config = NormalAnchorConfig()
    
    components = {}
    
    # ==== AMP GROUP: Envelope + Internal Shape ====
    amp_group = {}
    
    # env_overrun_rate (X11) - actual values 0 to 1, median ~0.005
    x11 = abs(_get_feature_value(features, 'X11', 'env_overrun_rate', 'viol_rate'))
    amp_group['env_rate'] = min(1.0, x11 / 0.5)  # Adjusted - 50% violation rate = full score
    
    # env_overrun_max (X12) - actual values 0 to ~8, median ~0.08
    x12 = abs(_get_feature_value(features, 'X12', 'env_overrun_max'))
    amp_group['env_max'] = min(1.0, x12 / 4.0)  # Adjusted - 4dB max violation = full score
    
    # env_violation_energy (X13) - actual values 0 to ~6000, median ~16
    x13 = abs(_get_feature_value(features, 'X13', 'env_violation_energy'))
    amp_group['env_energy'] = min(1.0, x13 / 500.0)  # Adjusted - 500 energy = full score
    
    # gain_nonlinearity / jump_energy (X7) - actual values 0.001 to ~1.4, median ~0.08
    x7 = abs(_get_feature_value(features, 'X7', 'gain_nonlinearity', 'step_score'))
    amp_group['jump_energy'] = min(1.0, x7 / 0.5)  # Adjusted - 0.5 = full score
    
    # inband_flatness / ripple (X2)
    x2 = abs(_get_feature_value(features, 'X2', 'inband_flatness', 'ripple_var'))
    amp_group['flatness'] = min(1.0, x2 / 0.1)  # Adjusted
    
    # kurtosis_detrended (X20)
    x20 = abs(_get_feature_value(features, 'X20', 'kurtosis_detrended'))
    amp_group['kurtosis'] = min(1.0, max(0, x20 - 2.0) / 5.0)  # Adjusted
    
    components.update(amp_group)
    
    # ==== FREQ GROUP: Frequency Shift/Warp Evidence ====
    freq_group = {}
    
    # X16: corr_shift_bins (now in actual bins, not normalized)
    x16 = abs(_get_feature_value(features, 'X16', 'corr_shift_bins'))
    # For 820 points, 1 bin = ~1/820. With enhanced ppm, expect >= 1 bin shift
    # Adjusted: X16 can be large (up to 800) so use wider threshold
    freq_group['freq_shift'] = min(1.0, abs(x16) / 50.0)  # 50 bins = clear shift
    
    # X17: warp_scale
    x17 = abs(_get_feature_value(features, 'X17', 'warp_scale'))
    freq_group['warp_scale'] = min(1.0, x17 / 0.02)  # Adjusted based on actual values
    
    # X18: warp_bias
    x18 = abs(_get_feature_value(features, 'X18', 'warp_bias'))
    freq_group['warp_bias'] = min(1.0, x18 / 0.02)  # Adjusted
    
    # X23: warp_residual_energy (NEW)
    x23 = abs(_get_feature_value(features, 'X23', 'warp_residual_energy'))
    freq_group['warp_residual'] = min(1.0, x23 / 2.0)  # Adjusted - can be quite large
    
    # X24: phase_slope_diff (NEW)
    x24 = abs(_get_feature_value(features, 'X24', 'phase_slope_diff'))
    freq_group['phase_slope'] = min(1.0, x24 / 0.5)  # Adjusted
    
    # X25: interp_mse_after_shift (NEW)
    x25 = abs(_get_feature_value(features, 'X25', 'interp_mse_after_shift'))
    freq_group['interp_mse'] = min(1.0, x25 / 5.0)  # Adjusted
    
    components.update(freq_group)
    
    # ==== REF GROUP: Reference Level Evidence ====
    # NOTE: After data analysis, only X14 (low_band_residual) distinguishes ref_error from normal
    # Other features (X3, X26, X27, X28) do NOT discriminate - they add noise!
    # 
    # Data analysis results:
    #   X14: normal=0.027, ref_error=0.376 (14x difference) ✓
    #   X3:  normal=0.0002, ref_error=0.0002 (no difference)
    #   X26: normal=0.0004, ref_error=0.0004 (no difference)
    #   X27: normal=0.0001, ref_error=0.0001 (no difference)
    #   X28: normal=0.04,   ref_error=-0.02 (wrong direction!)
    
    ref_group = {}
    
    # X14: low_band_residual_mean - THE KEY FEATURE for ref detection
    # normal=0.027, ref_error=0.376 - 14x difference!
    # threshold 0.10: normal ~0.27, ref_error >1.0
    x14 = abs(_get_feature_value(features, 'X14', 'low_band_residual'))
    ref_group['low_band_res'] = min(1.0, x14 / 0.10)
    
    # Only keep X14 - other features add noise, not signal
    
    components.update(ref_group)
    
    # ==== GROUPED MAX FUSION ====
    # Instead of weighted average, use max within each group
    score_amp = max(amp_group.values()) if amp_group else 0.0
    score_freq = max(freq_group.values()) if freq_group else 0.0
    score_ref = max(ref_group.values()) if ref_group else 0.0
    
    # Final anchor_score = max of group scores (or weighted combination)
    # Using max to prevent dilution
    anchor_score = max(score_amp, score_freq, score_ref)
    
    # Also compute weighted version for calibration flexibility
    w_amp, w_freq, w_ref = 0.4, 0.3, 0.3  # Weights can be calibrated
    anchor_score_weighted = w_amp * score_amp + w_freq * score_freq + w_ref * score_ref
    
    return {
        'anchor_score': anchor_score,
        'anchor_score_weighted': anchor_score_weighted,
        'score_amp': score_amp,
        'score_freq': score_freq,
        'score_ref': score_ref,
        'components': components,
        'amp_group': amp_group,
        'freq_group': freq_group,
        'ref_group': ref_group,
        'T_low': config.T_low,
        'T_high': config.T_high,
    }


def compute_normal_logit(anchor_score: float, T_low: float, T_high: float, k: float) -> float:
    """Compute Normal logit prior based on anchor score.
    
    Implements soft gating with dual thresholds:
    - anchor_score <= T_low: strong Normal prior (logit = k)
    - anchor_score >= T_high: minimal Normal prior (logit = 0)
    - T_low < anchor_score < T_high: linearly interpolated
    
    Parameters
    ----------
    anchor_score : float
        Anomaly score from compute_anchor_score.
    T_low : float
        Lower threshold.
    T_high : float
        Upper threshold.
    k : float
        Maximum logit boost for Normal.
        
    Returns
    -------
    float
        Normal logit prior to be added before softmax.
    """
    if anchor_score <= T_low:
        return k  # Strong Normal prior
    elif anchor_score >= T_high:
        return 0.0  # Minimal Normal prior
    else:
        # Linear interpolation in gray zone
        ratio = (T_high - anchor_score) / (T_high - T_low + 1e-12)
        return k * ratio


def infer_normal_anchor(
    features: Dict[str, float],
    calibration: Optional[Dict] = None,
    config: Optional[NormalAnchorConfig] = None
) -> Dict[str, float]:
    """Perform Stage-0 Normal Anchor inference with soft gating.
    
    IMPORTANT: This version does NOT bypass classification!
    Instead, it returns a Normal logit prior that competes with
    fault logits in the final softmax.
    
    Parameters
    ----------
    features : dict
        Feature dictionary containing X1-X22 or equivalent.
    calibration : dict, optional
        Calibration parameters from calibration.json.
        Expected keys: T_low, T_high, k_normal_prior
    config : NormalAnchorConfig, optional
        Configuration for normal detection.
        
    Returns
    -------
    dict
        anchor_score : float
            Anomaly score (higher = more abnormal).
        normal_logit : float
            Logit prior for Normal class.
        is_low_zone : bool
            True if anchor_score <= T_low (strong normal prior).
        is_gray_zone : bool
            True if T_low < anchor_score < T_high.
        is_high_zone : bool
            True if anchor_score >= T_high (minimal normal prior).
        bypass_classification : bool
            ALWAYS False in v2 - we never bypass.
        components : dict
            Individual component scores for debugging.
    """
    if config is None:
        config = NormalAnchorConfig()
    
    # Apply calibration if provided
    if calibration is not None and config.use_calibration:
        config.T_low = calibration.get('T_low', config.T_low)
        config.T_high = calibration.get('T_high', config.T_high)
        config.k_normal_prior = calibration.get('k_normal_prior', config.k_normal_prior)
    
    # Compute anchor score with all components
    result = compute_anchor_score(features, config)
    anchor_score = result['anchor_score']
    
    # Compute Normal logit prior (soft gating)
    normal_logit = compute_normal_logit(
        anchor_score, 
        config.T_low, 
        config.T_high, 
        config.k_normal_prior
    )
    
    # Determine zone
    is_low_zone = anchor_score <= config.T_low
    is_high_zone = anchor_score >= config.T_high
    is_gray_zone = not is_low_zone and not is_high_zone
    
    # Compute uncertainty based on zone
    if is_low_zone:
        uncertainty = anchor_score / config.T_low if config.T_low > 0 else 0.0
    elif is_high_zone:
        uncertainty = 0.2  # Low uncertainty for clear faults
    else:
        # Gray zone - higher uncertainty
        uncertainty = 0.5
    
    return {
        'anchor_score': anchor_score,
        'score': anchor_score,  # Alias for compatibility
        'normal_logit': normal_logit,
        'is_low_zone': is_low_zone,
        'is_gray_zone': is_gray_zone,
        'is_high_zone': is_high_zone,
        'is_normal': is_low_zone,  # Compatibility
        'confidence': 1.0 - uncertainty,
        'uncertainty': uncertainty,
        'bypass_classification': False,  # NEVER bypass in v2
        'components': result['components'],
        'feature_scores': result['components'],  # Alias
        'T_low': config.T_low,
        'T_high': config.T_high,
        'k_normal_prior': config.k_normal_prior,
    }


def load_calibration(calibration_path: Path) -> Optional[Dict]:
    """Load calibration parameters from JSON file.
    
    Parameters
    ----------
    calibration_path : Path
        Path to calibration.json file.
        
    Returns
    -------
    dict or None
        Calibration parameters if file exists, None otherwise.
    """
    if not calibration_path.exists():
        return None
    
    try:
        with open(calibration_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def compute_normal_stats_from_data(
    normal_features_list: List[Dict[str, float]],
    quantile: float = 0.95
) -> Dict[str, float]:
    """Compute normal thresholds from a list of normal sample features.
    
    Parameters
    ----------
    normal_features_list : List[dict]
        List of feature dictionaries from normal samples.
    quantile : float
        Quantile to use for threshold computation.
        
    Returns
    -------
    dict
        Threshold values for normal detection.
    """
    if not normal_features_list:
        return {'T_normal': 0.15, 'T_prob': 0.30}
    
    config = NormalAnchorConfig()
    scores = []
    
    for features in normal_features_list:
        result = compute_normal_score(features, config)
        scores.append(result['normal_score'])
    
    scores = np.array(scores)
    
    return {
        'T_normal': float(np.percentile(scores, quantile * 100)),
        'T_prob': float(np.percentile(scores, quantile * 100) * 2),  # More conservative for prob
        'normal_score_mean': float(np.mean(scores)),
        'normal_score_std': float(np.std(scores)),
        'normal_score_max': float(np.max(scores)),
    }


if __name__ == "__main__":
    # Self-test with example features
    test_features = {
        'X1': 0.01, 'X2': 0.005, 'X3': 1e-11, 'X4': 1e6, 'X5': 0.02,
        'X6': 0.002, 'X7': 0.1, 'X8': 0.05, 'X9': 5000, 'X10': 0.03,
        'X11': 0.02, 'X12': 0.3, 'X13': 0.1, 'X14': 0.05, 'X15': 0.02,
        'X16': 0.005, 'X17': 0.01, 'X18': 0.005,
        'X19': 1e-11, 'X20': 0.8, 'X21': 3, 'X22': 0.2,
    }
    
    result = infer_normal_anchor(test_features)
    print("Normal Anchor Test Result:")
    print(f"  is_normal: {result['is_normal']}")
    print(f"  score: {result['score']:.4f}")
    print(f"  confidence: {result['confidence']:.4f}")
    print(f"  bypass_classification: {result['bypass_classification']}")
