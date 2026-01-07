"""Amplitude sub-BRB for system-level inference."""
from __future__ import annotations

from typing import Dict, Tuple


def _get_raw_feature(features: Dict[str, float], *names) -> float:
    """Get raw feature value from dict."""
    for name in names:
        if name in features:
            return features[name]
    return 0.0


def infer_amp_brb(scores: Dict[str, float], match: Dict[str, Tuple[float, float, float]], rule_weight: float) -> Dict[str, float]:
    """Infer amplitude-related anomaly activation.

    Amplitude errors are characterized by:
    - band_consistency: amp~16.4 vs others<1 (STRONGEST discriminator)
    - X5/scale_consistency: amp~1101 vs others<61
    - switching_rate: amp~0.115 vs others~0.08
    
    Uses distinctive patterns with boosted weights when key thresholds are met.
    """
    # Get raw scores for threshold-based boosting
    x5_score = scores.get("X5", 0.0)
    x9_score = scores.get("X9", 0.0)  # band_consistency
    x13_score = scores.get("X13", 0.0)  # switching_rate
    
    # Primary features with base weights
    primary = [
        (x9_score, 0.35),   # band_consistency - most discriminative for amp
        (x5_score, 0.25),   # scale_consistency - high variance for amp
        (x13_score, 0.20),  # switching_rate - higher for amp
    ]
    
    secondary = [
        (scores.get("X6", 0.0), 0.08),   # ripple_std
        (scores.get("X11", 0.0), 0.06),  # amp_std
        (scores.get("X10", 0.0), 0.06),  # amp_iqr
    ]
    
    weighted_sum = sum(score * weight for score, weight in primary + secondary)
    
    # Boost if key amp indicators are present
    # X9 > 0.3 means band_consistency > 1.1 (threshold for amp_error)
    # X5 > 0.3 means scale_consistency > 90 (threshold for amp_error)
    if x9_score > 0.3 or x5_score > 0.3:
        weighted_sum *= 1.5  # Boost when amp patterns detected
    
    activation = rule_weight * min(1.0, weighted_sum)
    
    return {
        "name": "amp",
        "label": "幅度失准",
        "activation": float(activation),
        "fault": activation > 0.3,
    }
