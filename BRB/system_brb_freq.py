"""Frequency sub-BRB for system-level inference."""
from __future__ import annotations

from typing import Dict, Tuple


def infer_freq_brb(scores: Dict[str, float], match: Dict[str, Tuple[float, float, float]], rule_weight: float) -> Dict[str, float]:
    """Infer frequency-related anomaly activation.
    
    Frequency errors are characterized by:
    - X2/ripple_var: freq~10683 vs others<1 (STRONGEST discriminator)
    - X17/amp_range: freq~555 vs others~40
    - X11/amp_std: freq~35 vs others~1
    - X20/amp_kurtosis: freq~363 vs others<136
    
    These features make freq_error very distinctive.
    """
    # Get scores for threshold-based boosting
    x2_score = scores.get("X2", 0.0)   # ripple_var
    x17_score = scores.get("X17", 0.0)  # amp_range
    x11_score = scores.get("X11", 0.0)  # amp_std
    x20_score = scores.get("X20", 0.0)  # amp_kurtosis
    
    # Primary frequency features
    primary = [
        (x2_score, 0.35),   # ripple_var - strongest discriminator
        (x17_score, 0.25),  # amp_range - very distinctive
        (x11_score, 0.15),  # amp_std
    ]
    
    secondary = [
        (x20_score, 0.10),  # amp_kurtosis
        (scores.get("X12", 0.0), 0.08),  # noise_level
        (scores.get("X1", 0.0), 0.07),   # bias
    ]
    
    weighted_sum = sum(score * weight for score, weight in primary + secondary)
    
    # Boost if key freq indicators are present
    # Any of these being high is a strong indicator of freq_error
    if x2_score > 0.2 or x17_score > 0.2 or (x11_score > 0.3 and x20_score > 0.3):
        weighted_sum *= 1.5  # Boost when freq patterns detected
    
    activation = rule_weight * min(1.0, weighted_sum)
    
    return {
        "name": "freq",
        "label": "频率失准",
        "activation": float(activation),
        "fault": activation > 0.3,
    }
