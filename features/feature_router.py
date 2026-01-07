"""Feature routing for BRB modules by fault type."""
from __future__ import annotations

from typing import Dict


BASE_FEATURE_KEYS = {
    "step_score",
    "res_slope",
    "ripple_var",
    "df",
    "viol_rate",
    "bias",
    "gain",
    "module_id",
}


FAULT_FEATURE_MAP = {
    "amp": [
        "X1",
        "X6",
        "X11",
        "X12",
        "X13",
        "X19",
        "X20",
        "X21",
        "X22",
        "X11_out_env_ratio",
        "X12_max_env_violation",
        "X13_env_violation_energy",
        "X17",
        "X18",
    ],
    "freq": [
        "X2",
        "X4",
        "X7",
        "X14",
        "X15",
        "X16",
        "X17",
        "X18",
        "X14_low_band_residual",
        "X15_high_band_residual_std",
    ],
    "ref": [
        "X1",
        "X3",
        "X5",
        "X8",
        "X11",
        "X12",
        "X13",
        "X11_out_env_ratio",
        "X12_max_env_violation",
        "X13_env_violation_energy",
    ],
}


def feature_router(features: Dict[str, float], fault_type: str) -> Dict[str, float]:
    """Route features to BRB modules based on fault type.

    Parameters
    ----------
    features : dict
        Full feature dictionary produced by the extractor.
    fault_type : {'amp', 'freq', 'ref'}
        Fault category used by the system-level BRB.

    Returns
    -------
    dict
        Subset of features relevant to the requested fault type. Core
        module-level features (step_score, res_slope, etc.) are always
        passed through to avoid losing baseline cues.
    """
    if fault_type not in FAULT_FEATURE_MAP:
        raise ValueError("Unknown fault type")

    selected = {k: features[k] for k in BASE_FEATURE_KEYS if k in features}
    for key in FAULT_FEATURE_MAP[fault_type]:
        if key in features:
            selected[key] = features[key]
    return selected
