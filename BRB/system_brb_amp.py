"""Amplitude sub-BRB for system-level inference."""
from __future__ import annotations

from typing import Dict, Tuple


def infer_amp_brb(scores: Dict[str, float], match: Dict[str, Tuple[float, float, float]], rule_weight: float) -> Dict[str, float]:
    """Infer amplitude-related anomaly activation.

    Uses amplitude-sensitive features: X1, X2, X5, envelope violations,
    gain distortion (X11) and amplitude-chain fine-grained features (X19-X22).
    """
    activation = rule_weight * max(
        match.get("X1", (0, 0, 0))[2],
        match.get("X2", (0, 0, 0))[2],
        match.get("X5", (0, 0, 0))[2],
        match.get("X11", (0, 0, 0))[2],
        match.get("X11_out_env_ratio", (0, 0, 0))[2],
        match.get("X12_max_env_violation", (0, 0, 0))[2],
        match.get("X13_env_violation_energy", (0, 0, 0))[2],
        match.get("X19", (0, 0, 0))[2],
        match.get("X20", (0, 0, 0))[2],
        match.get("X21", (0, 0, 0))[2],
        match.get("X22", (0, 0, 0))[2],
    )
    return {
        "name": "amp",
        "label": "幅度失准",
        "activation": float(activation),
        "fault": activation > 0.5,
    }
