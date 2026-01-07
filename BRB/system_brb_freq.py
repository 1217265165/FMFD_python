"""Frequency sub-BRB for system-level inference."""
from __future__ import annotations

from typing import Dict, Tuple


def infer_freq_brb(scores: Dict[str, float], match: Dict[str, Tuple[float, float, float]], rule_weight: float) -> Dict[str, float]:
    """Infer frequency-related anomaly activation."""
    activation = rule_weight * max(
        match.get("X4", (0, 0, 0))[2],
        match.get("X14", (0, 0, 0))[2],
        match.get("X15", (0, 0, 0))[2],
        match.get("X14_low_band_residual", (0, 0, 0))[2],
        match.get("X15_high_band_residual_std", (0, 0, 0))[2],
        match.get("X16", (0, 0, 0))[2],
        match.get("X17", (0, 0, 0))[2],
        match.get("X18", (0, 0, 0))[2],
    )
    return {
        "name": "freq",
        "label": "频率失准",
        "activation": float(activation),
        "fault": activation > 0.5,
    }
