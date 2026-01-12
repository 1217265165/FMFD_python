#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-0 Normal Anchor Module
============================

This module implements the first stage of system-level diagnosis:
determining whether a sample is Normal vs Abnormal before proceeding
to fault-type classification.

The normal anchor uses a robust combination of envelope-based features
to detect samples that fall within normal operating bounds.
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
        Features used for normal detection (from feature_injection_plan.yaml).
    T_normal : float
        Threshold for combined normal score.
    T_prob : float
        Probability threshold for uncertainty.
    use_calibration : bool
        Whether to load thresholds from calibration.json.
    """
    normal_features: List[str] = field(default_factory=lambda: [
        'X11', 'X12', 'X13', 'X7', 'X2'
    ])
    T_normal: float = 0.15
    T_prob: float = 0.30
    use_calibration: bool = True


def _get_feature_value(features: Dict[str, float], *names: str, default: float = 0.0) -> float:
    """Get feature value with fallback to aliases."""
    for name in names:
        if name in features:
            return float(features[name])
    return default


def _normalize_feature(value: float, lower: float, upper: float) -> float:
    """Normalize feature value to [0, 1] range."""
    value = max(lower, min(value, upper))
    return (value - lower) / (upper - lower + 1e-12)


def compute_normal_score(features: Dict[str, float], config: NormalAnchorConfig = None) -> Dict[str, float]:
    """Compute normal anchor score from features.
    
    This function computes a combined score indicating how "normal" a sample is.
    Lower scores indicate more normal behavior.
    
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
        - normal_score: Combined normal score (lower = more normal)
        - feature_scores: Individual normalized feature scores
        - thresholds: Applied thresholds
    """
    if config is None:
        config = NormalAnchorConfig()
    
    # Normalization parameters for each feature
    # These define the expected range for normal samples
    norm_params = {
        'X2': (0.0, 0.02, 0.15),    # inband_flatness: (min, normal_max, fault_max)
        'X7': (0.0, 0.3, 2.0),      # gain_nonlinearity
        'X11': (0.0, 0.05, 0.3),    # env_overrun_rate
        'X12': (0.0, 0.5, 5.0),     # env_overrun_max
        'X13': (0.0, 0.2, 10.0),    # env_violation_energy
    }
    
    feature_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0
    
    # Weights for combining features (higher = more important for normal detection)
    weights = {
        'X11': 0.25,  # env_overrun_rate is primary indicator
        'X12': 0.20,  # env_overrun_max
        'X13': 0.20,  # env_violation_energy
        'X7': 0.20,   # gain_nonlinearity (jumps)
        'X2': 0.15,   # inband_flatness
    }
    
    for feat_name in config.normal_features:
        raw_value = abs(_get_feature_value(features, feat_name))
        
        if feat_name in norm_params:
            min_val, normal_max, fault_max = norm_params[feat_name]
            # Score: 0 if within normal range, increasing toward 1 for faults
            if raw_value <= normal_max:
                score = 0.0
            else:
                score = (raw_value - normal_max) / (fault_max - normal_max + 1e-12)
                score = min(1.0, score)
        else:
            # Default normalization
            score = _normalize_feature(raw_value, 0.0, 1.0)
        
        feature_scores[feat_name] = score
        weight = weights.get(feat_name, 0.1)
        weighted_sum += score * weight
        total_weight += weight
    
    normal_score = weighted_sum / (total_weight + 1e-12)
    
    return {
        'normal_score': normal_score,
        'feature_scores': feature_scores,
        'thresholds': {
            'T_normal': config.T_normal,
            'T_prob': config.T_prob,
        }
    }


def infer_normal_anchor(
    features: Dict[str, float],
    calibration: Optional[Dict] = None,
    config: Optional[NormalAnchorConfig] = None
) -> Dict[str, float]:
    """Perform Stage-0 Normal Anchor inference.
    
    Determines whether a sample should be classified as Normal
    without proceeding to fault-type classification.
    
    Parameters
    ----------
    features : dict
        Feature dictionary containing X1-X22 or equivalent.
    calibration : dict, optional
        Calibration parameters from calibration.json.
        Expected keys: T_normal, T_prob
    config : NormalAnchorConfig, optional
        Configuration for normal detection.
        
    Returns
    -------
    dict
        is_normal : bool
            True if sample is determined to be normal.
        score : float
            Normal score (lower = more normal).
        uncertainty : float
            Uncertainty in the decision (0-1).
        confidence : float
            Confidence in the normal decision (0-1).
        bypass_classification : bool
            If True, skip fault classification and output Normal.
    """
    if config is None:
        config = NormalAnchorConfig()
    
    # Apply calibration if provided
    if calibration is not None and config.use_calibration:
        config.T_normal = calibration.get('T_normal', config.T_normal)
        config.T_prob = calibration.get('T_prob', config.T_prob)
    
    # Compute normal score
    result = compute_normal_score(features, config)
    normal_score = result['normal_score']
    
    # Decision logic
    # If normal_score is below threshold, sample is likely normal
    is_normal = normal_score < config.T_normal
    
    # Compute confidence and uncertainty
    if normal_score < config.T_normal:
        # Clear normal - high confidence
        confidence = 1.0 - (normal_score / config.T_normal)
        uncertainty = normal_score / config.T_normal
    elif normal_score < config.T_normal * 2:
        # Ambiguous zone - medium confidence
        confidence = (config.T_normal * 2 - normal_score) / config.T_normal
        uncertainty = 0.5
    else:
        # Clear fault - low normal confidence
        confidence = 0.0
        uncertainty = 1.0 - min(1.0, (normal_score - config.T_normal * 2) / config.T_normal)
    
    # Only bypass classification if we're very confident it's normal
    bypass_classification = is_normal and confidence > 0.7
    
    return {
        'is_normal': is_normal,
        'score': normal_score,
        'uncertainty': uncertainty,
        'confidence': confidence,
        'bypass_classification': bypass_classification,
        'feature_scores': result['feature_scores'],
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
