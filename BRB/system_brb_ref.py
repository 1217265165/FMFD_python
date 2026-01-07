"""Reference-level sub-BRB for system-level inference."""
from __future__ import annotations

from typing import Dict, Tuple


def infer_ref_brb(scores: Dict[str, float], match: Dict[str, Tuple[float, float, float]], rule_weight: float) -> Dict[str, float]:
    """Infer reference-level anomaly activation.
    
    Reference errors are characterized by:
    - X1/bias: ref~1.64 vs normal~0.92, amp~0.82
    - amp_kurtosis: ref~63 (lower than normal~136)
    - amp_std: ref~1.38 vs normal~1.12
    - X5/scale_consistency: ref~26 (lower than normal~45)
    
    Reference errors show moderate bias increase without the extreme values of amp/freq errors.
    Must be careful not to over-predict since amp_error samples often overlap.
    """
    x1_score = scores.get("X1", 0.0)    # bias
    x11_score = scores.get("X11", 0.0)  # amp_std
    x5_score = scores.get("X5", 0.0)    # scale_consistency
    x9_score = scores.get("X9", 0.0)    # band_consistency
    x2_score = scores.get("X2", 0.0)    # ripple_var
    x17_score = scores.get("X17", 0.0)  # amp_range
    
    # Primary reference features
    primary = [
        (x1_score, 0.30),   # bias - ref~1.64 is distinctive
        (x11_score, 0.20),  # amp_std - ref~1.38
        (scores.get("X18", 0.0), 0.15),  # trend_intercept
    ]
    
    secondary = [
        (scores.get("X12", 0.0), 0.12),  # noise_level
        (scores.get("X10", 0.0), 0.10),  # amp_iqr
        (scores.get("X8", 0.0), 0.08),   # amp_skewness
        (scores.get("X15_high_band_residual_std", 0.0), 0.05),  # band4_std
    ]
    
    weighted_sum = sum(score * weight for score, weight in primary + secondary)
    
    # Reduce activation if amp or freq patterns are clearly present
    # This prevents ref from dominating when other patterns are clear
    if x9_score > 0.3 or x5_score > 0.5:  # Clear amp pattern
        weighted_sum *= 0.5
    if x2_score > 0.2 or x17_score > 0.3:  # Clear freq pattern
        weighted_sum *= 0.5
    
    activation = rule_weight * min(1.0, weighted_sum)
    
    return {
        "name": "ref",
        "label": "参考电平失准",
        "activation": float(activation),
        "fault": activation > 0.3,
    }
