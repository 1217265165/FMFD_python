#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System-Level Reference Level Branch Layers (分级注入)
=====================================================

This module implements the three-layer staged injection for the
reference level fault branch of the system-level BRB.

Layer structure:
- Layer-1: Uses stage1 features → outputs p1
- Layer-2: Uses p1 + stage2 features → outputs p2  
- Layer-3: Uses p2 + stage3 features → outputs p3 (final score)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default stage feature assignments (from feature_injection_plan.yaml)
REF_STAGE1_FEATURES = ['X3', 'X5']
REF_STAGE2_FEATURES = ['X7', 'X10']
REF_STAGE3_FEATURES = ['X12', 'X8']


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


def _triangular_membership(value: float, low: float = 0.15, center: float = 0.35, high: float = 0.7) -> Tuple[float, float, float]:
    """Return (low, normal, high) membership degrees."""
    if value <= low:
        return 1.0, 0.0, 0.0
    if low < value < center:
        low_mem = (center - value) / (center - low)
        normal_mem = 1.0 - low_mem
        return low_mem, normal_mem, 0.0
    if center <= value < high:
        high_mem = (value - center) / (high - center)
        normal_mem = 1.0 - high_mem
        return 0.0, normal_mem, high_mem
    return 0.0, 0.0, 1.0


def _compute_ref_scores(features: Dict[str, float]) -> Dict[str, float]:
    """Compute normalized scores for reference level-related features."""
    
    # Normalization ranges for reference level features
    norm_ranges = {
        'X3': (1e-12, 1e-9),    # hf_attenuation_slope
        'X5': (0.01, 0.35),     # scale_consistency
        'X7': (0.05, 2.0),      # gain_nonlinearity (step)
        'X8': (0.01, 1.0),      # lo_leakage
        'X10': (0.02, 0.5),     # band_amplitude_consistency
        'X12': (0.5, 5.0),      # env_overrun_max
    }
    
    scores = {}
    for feat_name, (lower, upper) in norm_ranges.items():
        raw_value = abs(_get_feature_value(features, feat_name))
        scores[feat_name] = _normalize_feature(raw_value, lower, upper)
    
    return scores


def ref_layer1_infer(features: Dict[str, float], alpha: float = 2.0) -> Dict[str, float]:
    """Layer-1 inference using stage1 features only.
    
    Parameters
    ----------
    features : dict
        Feature dictionary.
    alpha : float
        Softmax temperature.
        
    Returns
    -------
    dict
        p1: Layer-1 probability output
        activation: Rule activation level
        scores: Individual feature scores
    """
    scores = _compute_ref_scores(features)
    
    # Stage1 features: X3, X5
    stage1_scores = [
        scores.get('X3', 0.0),
        scores.get('X5', 0.0),
    ]
    
    # Weighted combination with membership-based activation
    weights = [0.50, 0.50]  # X3 (slope), X5 (scale consistency)
    activation = sum(s * w for s, w in zip(stage1_scores, weights))
    
    # Get membership degrees
    match_degrees = [_triangular_membership(s) for s in stage1_scores]
    
    # Compute rule activation using high membership
    rule_activation = max(m[2] for m in match_degrees)  # High membership
    
    p1 = min(1.0, activation + 0.30 * rule_activation)
    
    return {
        'p1': p1,
        'activation': activation,
        'rule_activation': rule_activation,
        'scores': {f: scores.get(f, 0.0) for f in REF_STAGE1_FEATURES},
    }


def ref_layer2_infer(features: Dict[str, float], p1: float, alpha: float = 2.0) -> Dict[str, float]:
    """Layer-2 inference using p1 + stage2 features.
    
    Parameters
    ----------
    features : dict
        Feature dictionary.
    p1 : float
        Output from layer-1.
    alpha : float
        Softmax temperature.
        
    Returns
    -------
    dict
        p2: Layer-2 probability output
        activation: Combined activation level
        scores: Individual feature scores
    """
    scores = _compute_ref_scores(features)
    
    # Stage2 features: X7, X10
    stage2_scores = [
        scores.get('X7', 0.0),
        scores.get('X10', 0.0),
    ]
    
    # Weighted combination
    weights = [0.55, 0.45]  # X7 (step), X10 (band consistency)
    stage2_activation = sum(s * w for s, w in zip(stage2_scores, weights))
    
    # Combine with p1 (layer injection)
    combined_activation = 0.60 * p1 + 0.40 * stage2_activation
    
    # Get membership degrees
    match_degrees = [_triangular_membership(s) for s in stage2_scores]
    rule_activation = max(m[2] for m in match_degrees)
    
    p2 = min(1.0, combined_activation + 0.25 * rule_activation)
    
    return {
        'p2': p2,
        'activation': combined_activation,
        'rule_activation': rule_activation,
        'scores': {f: scores.get(f, 0.0) for f in REF_STAGE2_FEATURES},
    }


def ref_layer3_infer(features: Dict[str, float], p2: float, alpha: float = 2.0) -> Dict[str, float]:
    """Layer-3 inference using p2 + stage3 features.
    
    Parameters
    ----------
    features : dict
        Feature dictionary.
    p2 : float
        Output from layer-2.
    alpha : float
        Softmax temperature.
        
    Returns
    -------
    dict
        p3: Final layer probability output (score_ref)
        activation: Final activation level
        scores: Individual feature scores
    """
    scores = _compute_ref_scores(features)
    
    # Stage3 features: X12, X8
    stage3_scores = [
        scores.get('X12', 0.0),
        scores.get('X8', 0.0),
    ]
    
    # Weighted combination
    weights = [0.55, 0.45]  # X12, X8
    stage3_activation = sum(s * w for s, w in zip(stage3_scores, weights))
    
    # Combine with p2 (final injection)
    combined_activation = 0.65 * p2 + 0.35 * stage3_activation
    
    # Get membership degrees
    match_degrees = [_triangular_membership(s) for s in stage3_scores]
    rule_activation = max(m[2] for m in match_degrees)
    
    p3 = min(1.0, combined_activation + 0.15 * rule_activation)
    
    return {
        'p3': p3,
        'activation': combined_activation,
        'rule_activation': rule_activation,
        'scores': {f: scores.get(f, 0.0) for f in REF_STAGE3_FEATURES},
    }


def infer_ref_layers(
    features: Dict[str, float],
    plan: Optional[Dict] = None,
    alpha: float = 2.0
) -> float:
    """Complete three-layer inference for reference level branch.
    
    Implements the staged injection pattern:
    Layer-1 (stage1) → p1
    Layer-2 (p1 + stage2) → p2
    Layer-3 (p2 + stage3) → p3 = score_ref
    
    Parameters
    ----------
    features : dict
        Feature dictionary containing X1-X22 or equivalent.
    plan : dict, optional
        Feature injection plan (unused in current implementation,
        reserved for dynamic configuration).
    alpha : float
        Softmax temperature for all layers.
        
    Returns
    -------
    float
        score_ref: Final reference level fault score.
    """
    # Layer 1
    layer1_result = ref_layer1_infer(features, alpha)
    p1 = layer1_result['p1']
    
    # Layer 2
    layer2_result = ref_layer2_infer(features, p1, alpha)
    p2 = layer2_result['p2']
    
    # Layer 3
    layer3_result = ref_layer3_infer(features, p2, alpha)
    p3 = layer3_result['p3']
    
    return p3


def infer_ref_layers_detailed(
    features: Dict[str, float],
    plan: Optional[Dict] = None,
    alpha: float = 2.0
) -> Dict[str, float]:
    """Complete three-layer inference with detailed output.
    
    Same as infer_ref_layers but returns full results from all layers.
    
    Returns
    -------
    dict
        score_ref: Final score
        layer1: Layer-1 results
        layer2: Layer-2 results
        layer3: Layer-3 results
    """
    # Layer 1
    layer1_result = ref_layer1_infer(features, alpha)
    p1 = layer1_result['p1']
    
    # Layer 2
    layer2_result = ref_layer2_infer(features, p1, alpha)
    p2 = layer2_result['p2']
    
    # Layer 3
    layer3_result = ref_layer3_infer(features, p2, alpha)
    p3 = layer3_result['p3']
    
    return {
        'score_ref': p3,
        'layer1': layer1_result,
        'layer2': layer2_result,
        'layer3': layer3_result,
    }


if __name__ == "__main__":
    # Self-test
    test_features = {
        'X3': 5e-10, 'X5': 0.2, 'X7': 1.0, 'X8': 0.5,
        'X10': 0.25, 'X12': 2.5,
    }
    
    score = infer_ref_layers(test_features)
    print(f"Reference level fault score: {score:.4f}")
    
    detailed = infer_ref_layers_detailed(test_features)
    print(f"Layer-1 p1: {detailed['layer1']['p1']:.4f}")
    print(f"Layer-2 p2: {detailed['layer2']['p2']:.4f}")
    print(f"Layer-3 p3: {detailed['layer3']['p3']:.4f}")
