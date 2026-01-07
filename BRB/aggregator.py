"""System-level BRB aggregation utilities."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List


def softmax(values: Iterable[float], alpha: float = 2.0) -> List[float]:
    """Compute temperature-scaled softmax."""
    vals = list(values)
    if not vals:
        return []
    exps = [math.exp(alpha * v) for v in vals]
    total = sum(exps) + 1e-12
    return [v / total for v in exps]


def aggregate_system_results(
    results: List[Dict[str, float]],
    alpha: float = 2.0,
    overall_score: float | None = None,
    overall_threshold: float = 0.0,
    normal_prob_threshold: float = 0.3,
) -> Dict[str, float]:
    """Aggregate sub-BRB outputs with softmax calibration.

    Parameters
    ----------
    results : list of dict
        Each item should contain keys ``label`` and ``activation``.
    alpha : float
        Softmax temperature (default 2.0).
    overall_score : float, optional
        System-level aggregated anomaly score.
    overall_threshold : float
        Threshold for normal detection based on overall_score.
    normal_prob_threshold : float
        Threshold for the max probability to decide normal state.
    """
    activations = [float(r.get("activation", 0.0)) for r in results]
    probs = softmax(activations, alpha=alpha)
    fault_probs = {r["label"]: p for r, p in zip(results, probs)}

    max_prob = max(fault_probs.values()) if fault_probs else 0.0
    normal_weight = 0.0
    if overall_score is not None and overall_threshold > 0:
        if overall_score < overall_threshold:
            normal_weight = 1.0 - overall_score / (overall_threshold + 1e-12)
    if max_prob < normal_prob_threshold:
        normal_weight = max(normal_weight, normal_prob_threshold - max_prob)

    fault_scale = max(0.0, 1.0 - normal_weight)
    scaled_faults = {k: v * fault_scale for k, v in fault_probs.items()}
    total = normal_weight + sum(scaled_faults.values())

    if total <= 1e-12:
        normalized = {"正常": 1.0}
    else:
        normalized = {"正常": normal_weight / total}
        normalized.update({k: v / total for k, v in scaled_faults.items()})

    return {
        "probabilities": normalized,
        "max_prob": max(normalized.values()) if normalized else 0.0,
        "is_normal": normalized.get("正常", 0.0) >= 0.5 or max_prob < normal_prob_threshold,
        "uncertainty": 1.0 - (max(normalized.values()) if normalized else 0.0),
        "results": results,
    }
