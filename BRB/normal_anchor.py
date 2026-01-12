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
    """Compute normal anchor score with both envelope and in-envelope anomalies.
    
    This function computes a combined score indicating how "abnormal" a sample is.
    Higher scores indicate more anomalous behavior.
    
    Includes TWO types of evidence:
    1. Envelope-related (external anomalies): violations outside bounds
    2. In-envelope anomalies (internal shape anomalies): jumps, warping, kurtosis
    
    Parameters
    ----------
    features : dict
        Feature dictionary containing X1-X22 or equivalent.
    config : NormalAnchorConfig, optional
        Configuration for normal detection.
        
    Returns
    -------
    dict
        Contains:
        - anchor_score: Combined anomaly score (higher = more abnormal)
        - components: Individual component scores
        - T_low, T_high: Applied thresholds
    """
    if config is None:
        config = NormalAnchorConfig()
    
    components = {}
    
    # ==== Envelope-related features (external anomalies) ====
    # env_overrun_rate (X11)
    x11 = abs(_get_feature_value(features, 'X11', 'env_overrun_rate', 'viol_rate'))
    components['env_rate'] = min(1.0, x11 / 0.2)  # Cap at 0.2
    
    # env_overrun_max (X12)
    x12 = abs(_get_feature_value(features, 'X12', 'env_overrun_max'))
    components['env_max'] = min(1.0, x12 / 3.0)  # Cap at 3.0
    
    # env_violation_energy (X13)
    x13 = abs(_get_feature_value(features, 'X13', 'env_violation_energy'))
    components['env_energy'] = min(1.0, x13 / 5.0)  # Cap at 5.0
    
    # ==== In-envelope anomalies (internal shape anomalies) ====
    # gain_nonlinearity / jump_energy (X7)
    x7 = abs(_get_feature_value(features, 'X7', 'gain_nonlinearity', 'step_score'))
    components['jump_energy'] = min(1.0, x7 / 1.5)  # Cap at 1.5
    
    # inband_flatness / ripple (X2)
    x2 = abs(_get_feature_value(features, 'X2', 'inband_flatness', 'ripple_var'))
    components['flatness'] = min(1.0, x2 / 0.05)  # Cap at 0.05
    
    # kurtosis_detrended (X20) - in-envelope shape anomaly
    x20 = abs(_get_feature_value(features, 'X20', 'kurtosis_detrended'))
    components['kurtosis'] = min(1.0, max(0, x20 - 2.0) / 3.0)  # Excess from 2.0
    
    # ==== Frequency-specific evidence ====
    # corr_shift_bins (X16) - frequency shift
    x16 = abs(_get_feature_value(features, 'X16', 'corr_shift_bins'))
    components['freq_shift'] = min(1.0, x16 / 0.05)
    
    # warp_scale (X17) - frequency warping
    x17 = abs(_get_feature_value(features, 'X17', 'warp_scale'))
    components['warp_scale'] = min(1.0, x17 / 0.03)
    
    # warp_bias (X18) - frequency bias
    x18 = abs(_get_feature_value(features, 'X18', 'warp_bias'))
    components['warp_bias'] = min(1.0, x18 / 0.03)
    
    # ==== Reference-specific evidence ====
    # hf_attenuation_slope (X3) - ref level compression
    x3 = abs(_get_feature_value(features, 'X3', 'hf_attenuation_slope', 'res_slope'))
    components['compress_slope'] = min(1.0, x3 / 1e-9) if x3 > 1e-12 else 0.0
    
    # scale_consistency (X5)
    x5 = abs(_get_feature_value(features, 'X5', 'scale_consistency'))
    components['scale_consist'] = min(1.0, x5 / 0.25)
    
    # ==== Combine components with clipping to positive only ====
    # Only penalize anomalous direction (positive z-scores)
    component_values = [max(0.0, v) for v in components.values()]
    
    # Weighted combination (emphasize envelope + internal shape)
    weights = {
        'env_rate': 0.12,
        'env_max': 0.10,
        'env_energy': 0.10,
        'jump_energy': 0.15,
        'flatness': 0.10,
        'kurtosis': 0.08,
        'freq_shift': 0.10,
        'warp_scale': 0.08,
        'warp_bias': 0.05,
        'compress_slope': 0.06,
        'scale_consist': 0.06,
    }
    
    weighted_sum = sum(components.get(k, 0) * w for k, w in weights.items())
    total_weight = sum(weights.values())
    anchor_score = weighted_sum / (total_weight + 1e-12)
    
    return {
        'anchor_score': anchor_score,
        'components': components,
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
