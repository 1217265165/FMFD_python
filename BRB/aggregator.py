#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统级推理聚合器 (System-Level Aggregator)
==========================================
对应小论文系统级诊断的聚合逻辑。

本模块负责：
1. Stage-0 Normal Anchor (先判 Normal vs Abnormal)
2. 聚合三个子BRB（幅度、频率、参考电平）的推理结果
3. 应用门控+温度softmax进行概率校准
4. 执行正常状态识别
5. 输出最终的系统级诊断结果

Enhanced with:
- Evidence gating (beta_freq, beta_ref)
- Temperature-calibrated softmax
- Stage-0 normal anchor detection
- Calibration.json loading
- BRB-MU style reliability weighting (v5)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .system_brb_amp import amp_brb_infer
from .system_brb_freq import freq_brb_infer
from .system_brb_ref import ref_brb_infer


# v7 Constants for reliability mechanism
X16_AMP_DISTORTION_THRESHOLD = 100  # X16 (corr_shift_bins) > this indicates amp error distortion
GAMMA_MAX_EFFECT = 0.3  # Maximum effect of gamma on temperature adjustment
ALPHA_MAX_MULTIPLIER = 1.3  # Maximum multiplier for adaptive temperature (1 + GAMMA_MAX_EFFECT)
AMP_EVIDENCE_STRONG_THRESHOLD = 0.5  # v7: Above this, amp evidence is strong enough for soft suppression
SOFT_SUPPRESSION_LAMBDA = 1.0  # v7: Lambda for soft suppression in evidence gating

# v7 Suppression constants for Amp/Freq/Ref discrimination
FREQ_SUPPRESS_MULTIPLIER = 0.7  # How much freq is suppressed when amp is dominant
FREQ_MIN_RETAIN = 0.3  # Minimum retention for freq evidence (never fully suppress)
REF_SUPPRESS_MULTIPLIER = 0.5  # How much ref is suppressed when amp is dominant
REF_MIN_RETAIN = 0.1  # Minimum retention for ref evidence when amp is clearly dominant
X14_AMP_VS_REF_THRESHOLD = 0.5  # X14 > this suggests Amp error, not Ref error

# v8 Constants for multi-evidence Ref Gate (FIXED based on data analysis)
# Data analysis results (feature_separation_amp_ref.csv):
#   Amp→Ref errors: X14=0.25 (LOW), X30=0.45 (HIGH)
#   Amp_correct:    X14=2.32 (HIGH), X30=0.04 (LOW)
#   Ref_correct:    X14=0.20 (LOW), X30=0.06 (LOW)
#
# Key insight: HIGH X30 (compress_index > 0.2) = Amp error, NOT Ref!
# So we SUPPRESS ref when X30 is high (opposite to original v8 logic)
X30_AMP_INDICATOR_THRESHOLD = 0.2  # X30 > this suggests Amp error (suppress ref)
X14_REF_MAX_THRESHOLD = 0.5  # X14 < this may be Ref (combined with low X30)

# Default calibration values
DEFAULT_CALIBRATION = {
    'alpha': 2.0,           # Softmax temperature
    'beta_freq': 0.5,       # Frequency branch evidence boost
    'beta_ref': 0.5,        # Reference branch evidence boost
    'beta_amp': 0.3,        # Amplitude branch evidence boost (optional)
    'gamma': 0.5,           # Reliability adaptive temperature factor
    'T_normal': 0.15,       # Normal detection threshold
    'T_prob': 0.30,         # Probability threshold
    'overall_threshold': 0.15,
    'max_prob_threshold': 0.30,
    'T_rel': 0.6,           # Reliability threshold for module routing
}

_CALIBRATION_OVERRIDE: Optional[Dict] = None
_NORMAL_FEATURE_STATS: Optional[Dict[str, Dict[str, float]]] = None


def set_calibration_override(config: Optional[Dict]) -> None:
    """Set a process-level calibration override (used by tuning scripts)."""
    global _CALIBRATION_OVERRIDE
    _CALIBRATION_OVERRIDE = config


def _load_normal_feature_stats(stats_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Load robust feature stats from normal_feature_stats.csv (cached)."""
    global _NORMAL_FEATURE_STATS
    if _NORMAL_FEATURE_STATS is not None:
        return _NORMAL_FEATURE_STATS

    repo_root = Path(__file__).resolve().parents[1]
    stats_path = stats_path or (repo_root / "Output" / "normal_feature_stats.csv")
    stats: Dict[str, Dict[str, float]] = {}
    if stats_path.exists():
        try:
            with open(stats_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stat_name = row.get("stat") or row.get("") or row.get("Unnamed: 0")
                    if not stat_name:
                        continue
                    stats[stat_name] = {}
                    for key, value in row.items():
                        if key in {"stat", "", "Unnamed: 0"}:
                            continue
                        try:
                            stats[stat_name][key] = float(value) if value not in {None, ""} else 0.0
                        except ValueError:
                            stats[stat_name][key] = 0.0
        except Exception:
            stats = {}

    _NORMAL_FEATURE_STATS = stats
    return stats


def _stat_value(stats: Dict[str, Dict[str, float]], stat_name: str, key: str, default: float) -> float:
    return float(stats.get(stat_name, {}).get(key, default))


def load_calibration(calibration_path: Optional[Path] = None) -> Dict:
    """Load calibration parameters from JSON file.
    
    Parameters
    ----------
    calibration_path : Path, optional
        Path to calibration.json. If None, searches in Output directory.
        
    Returns
    -------
    dict
        Calibration parameters.
    """
    if _CALIBRATION_OVERRIDE is not None:
        result = DEFAULT_CALIBRATION.copy()
        result.update(_CALIBRATION_OVERRIDE)
        return result

    if calibration_path is None:
        # Try default location
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "Output" / "ours_best_config.json",
            repo_root / "Output" / "calibration.json",
        ]
        for path in candidates:
            if path.exists():
                calibration_path = path
                break
        else:
            calibration_path = repo_root / "Output" / "calibration.json"
    
    if calibration_path.exists():
        try:
            with open(calibration_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults
                result = DEFAULT_CALIBRATION.copy()
                result.update(loaded)
                return result
        except Exception:
            pass
    
    return DEFAULT_CALIBRATION.copy()


def compute_evidence_gating(
    features: Dict[str, float],
    calibration: Dict
) -> Tuple[float, float, float]:
    """Compute evidence gating boosts for amp, freq and ref branches.
    
    v7 FIX: Changed from hard MUTEX to SOFT SUPPRESSION.
    When amp evidence is strong, we now use a soft suppression factor
    that reduces (but doesn't zero out) freq/ref boosting.
    
    v7.1 FIX: Use X14 threshold to distinguish Amp from Ref.
    - Amp errors have X14 >> 0.5 (mean=1.33)
    - Ref errors have X14 ~= 0.3-0.5 (mean=0.38)
    When amp_evidence is high AND X14 > 0.5, suppress ref boosting.
    When amp_evidence is high AND X14 < 0.5, DON'T suppress ref (it's likely ref).
    
    v8 FIX: Multi-evidence Ref Gate
    - Ref Gate Score = I(X14>t14) + I(X29>t29) + I(X30>t30)
    - If ref_gate_score < 2: ref_boost = 0 (no ref boosting unless consistent evidence)
    - This prevents Amp errors with high X14 from being misclassified as Ref
    
    Parameters
    ----------
    features : dict
        Feature dictionary.
    calibration : dict
        Calibration parameters containing beta_freq, beta_ref, beta_amp.
        
    Returns
    -------
    Tuple[float, float, float]
        (amp_boost, freq_boost, ref_boost) - gating boost values.
    """
    beta_amp = calibration.get('beta_amp', 0.5)
    beta_freq = calibration.get('beta_freq', 0.5)
    beta_ref = calibration.get('beta_ref', 0.5)
    
    stats = calibration.get("normal_quantiles") or _load_normal_feature_stats()

    def _q95(key: str, default: float) -> float:
        if not stats:
            return default
        return _stat_value(stats, "p95", key, default)

    def _median(key: str, default: float) -> float:
        if not stats:
            return default
        return _stat_value(stats, "median", key, default)

    # Amplitude evidence thresholds
    amp_evidence = 0.0
    x11 = abs(float(features.get('X11', 0)))  # env_overrun_rate
    x12 = abs(float(features.get('X12', 0)))  # env_overrun_max
    x13 = abs(float(features.get('X13', 0)))  # env_violation_energy

    x11_thr = _q95("env_overrun_rate", 0.05)
    x12_thr = _q95("env_overrun_max", 1.0)
    x13_thr = _q95("env_overrun_mean", 50.0)

    if x11 > x11_thr or x12 > x12_thr or x13 > x13_thr:
        amp_evidence = max(x11 / (x11_thr + 1e-9), x12 / (x12_thr + 1e-9), x13 / (x13_thr + 1e-9))
        amp_evidence = min(1.0, amp_evidence)
    
    # Get X14, X29, X30 for v8 multi-evidence Ref Gate
    x14 = abs(float(features.get('X14', 0)))  # low_band_residual
    x29 = float(features.get('X29', 0))  # HF/LF energy ratio (can be negative)
    x30 = float(features.get('X30', 0))  # High-level compression index (can be negative)
    x31 = abs(float(features.get('X31', 0)))  # Piecewise offset consistency
    
    # v7 FIX: Use SOFT SUPPRESSION instead of hard MUTEX
    E_strong = AMP_EVIDENCE_STRONG_THRESHOLD
    amp_excess = max(0, amp_evidence - E_strong)
    suppression_factor = min(1.0, SOFT_SUPPRESSION_LAMBDA * amp_excess)

    freq_suppress = calibration.get("freq_suppress_multiplier", FREQ_SUPPRESS_MULTIPLIER)
    ref_suppress = calibration.get("ref_suppress_multiplier", REF_SUPPRESS_MULTIPLIER)

    # freq gets soft suppression when amp is strong (but never zero out completely)
    freq_retain = max(FREQ_MIN_RETAIN, 1.0 - suppression_factor * freq_suppress)
    
    # ===== v8 FIX: Corrected Ref Gate based on data analysis =====
    # Key finding from feature_separation_amp_ref.csv:
    #   Amp→Ref errors: X14=0.25 (low), X30=0.45 (HIGH) <- X30 indicates Amp NOT Ref!
    #   Amp_correct:    X14=2.32 (high), X30=0.04 (low)
    #   Ref_correct:    X14=0.20 (low), X30=0.06 (low)
    #
    # Rule: If X30 > 0.2, it's likely Amp (suppress ref); If X14 < 0.5 AND X30 < 0.2, allow ref
    
    # Check if this looks like an Amp error being wrongly classified as Ref
    is_amp_disguised_as_ref = abs(x30) > X30_AMP_INDICATOR_THRESHOLD  # High compress_index = Amp
    is_possible_ref = x14 < X14_REF_MAX_THRESHOLD and abs(x30) < X30_AMP_INDICATOR_THRESHOLD
    
    if is_amp_disguised_as_ref:
        # High X30 indicates Amp error - suppress ref completely
        ref_retain = 0.0
    elif amp_evidence > E_strong and x14 > X14_AMP_VS_REF_THRESHOLD:
        # High amp evidence with high X14 = definitely Amp, suppress ref
        ref_retain = REF_MIN_RETAIN
    elif is_possible_ref:
        # Low X14 AND low X30 = possibly Ref, allow boosting
        ref_retain = 1.0
    else:
        # Ambiguous - use partial suppression
        ref_retain = 0.5
    
    # Frequency evidence thresholds (normalized)
    freq_evidence = 0.0
    x16 = abs(float(features.get('X16', 0)))  # corr_shift_bins
    x17 = abs(float(features.get('X17', 0)))  # warp_scale
    x18 = abs(float(features.get('X18', 0)))  # warp_bias
    x24 = abs(float(features.get('X24', 0)))  # phase_slope_diff
    
    # v7 FIX: X16 can be falsely large for amp errors - use X24 (phase_slope) as primary
    x16_valid = x16 < X16_AMP_DISTORTION_THRESHOLD
    
    if x24 > 0.15:  # phase_slope_diff is the KEY freq feature
        freq_evidence = min(1.0, x24 / 0.5)
    elif x16_valid and (x16 > 10.0 or x17 > 0.005 or x18 > 0.005):
        freq_evidence = max((x16 / 50.0) if x16_valid else 0.0, x17 / 0.02, x18 / 0.02)
        freq_evidence = min(1.0, freq_evidence)
    
    # Reference evidence thresholds - primarily X14
    ref_evidence = 0.0
    x14_thr = _q95("band_offset_db_1", 0.05)
    if x14 > x14_thr:
        ref_evidence = min(1.0, x14 / (x14_thr + 1e-9))
    
    # Apply gating with soft suppression
    amp_boost = beta_amp * amp_evidence
    freq_boost = beta_freq * freq_evidence * freq_retain
    ref_boost = beta_ref * ref_evidence * ref_retain

    # === New robust gating based on envelope-insensitive features ===
    global_offset = float(features.get("global_offset_db", 0.0))
    shape_rmse = float(features.get("shape_rmse", 0.0))
    ripple_hp = float(features.get("ripple_hp", 0.0))
    freq_shift_score = float(features.get("freq_shift_score", 0.0))
    compress_ratio = float(features.get("compress_ratio", 0.0))
    compress_ratio_high = float(features.get("compress_ratio_high", 0.0))
    low_band_offset = float(features.get("band_offset_db_1", 0.0))

    offset_thr = _q95("global_offset_db", 0.1)
    offset_low_thr = _median("global_offset_db", 0.05)
    shape_thr = _q95("shape_rmse", 0.1)
    ripple_thr = _q95("ripple_hp", 0.05)
    shift_thr = _q95("freq_shift_score", 0.05)
    compress_thr = _q95("compress_ratio", 0.2)
    compress_high_thr = _q95("compress_ratio_high", 0.2)
    low_band_thr = _q95("band_offset_db_1", 0.05)

    w_offset = calibration.get("w_offset_to_ref", 0.5)
    w_ripple = calibration.get("w_ripple_to_amp", 0.5)
    w_shift = calibration.get("w_shift_to_freq", 0.5)

    # Ref gate: large global offset + low shape variation
    offset_norm = abs(global_offset) / (offset_thr + 1e-9) if offset_thr > 0 else 0.0
    shape_norm = shape_rmse / (shape_thr + 1e-9) if shape_thr > 0 else 0.0
    if offset_norm > 1.0 and shape_norm <= 1.0:
        ref_boost += w_offset * min(2.0, offset_norm)
        amp_boost *= max(0.0, 1.0 - ref_suppress * min(1.0, offset_norm))

    # Freq gate: strong correlation shift or slope drift
    shift_norm = freq_shift_score / (shift_thr + 1e-9) if shift_thr > 0 else 0.0
    offset_slope = float(features.get("offset_slope", 0.0))
    slope_thr = _q95("offset_slope", 0.05)
    slope_norm = abs(offset_slope) / (slope_thr + 1e-9) if slope_thr > 0 else 0.0
    if shift_norm > 1.0 or slope_norm > 1.0:
        freq_boost += w_shift * min(2.0, max(shift_norm, slope_norm))

    # Amp gate: ripple/Compression abnormal, but no large global offset
    ripple_norm = ripple_hp / (ripple_thr + 1e-9) if ripple_thr > 0 else 0.0
    compress_norm = compress_ratio_high / (compress_high_thr + 1e-9) if compress_high_thr > 0 else 0.0
    if (ripple_norm > 1.0 or compress_norm > 1.0) and abs(global_offset) <= offset_low_thr:
        amp_boost += w_ripple * min(2.0, max(ripple_norm, compress_norm))
        ref_boost *= max(0.0, 1.0 - ref_suppress * min(1.0, max(ripple_norm, compress_norm)))

    # Amp->Ref confusion guard: low-band residual needs offset + compression evidence
    if abs(low_band_offset) > low_band_thr:
        if abs(global_offset) > offset_thr and (compress_ratio > compress_thr or compress_ratio_high > compress_high_thr):
            ref_boost += w_offset * min(1.0, abs(low_band_offset) / (low_band_thr + 1e-9))
        else:
            ref_boost *= max(0.0, 1.0 - ref_suppress * 0.5)

    return amp_boost, freq_boost, ref_boost


def compute_reliability(
    features: Dict[str, float],
    calibration: Dict
) -> Dict[str, float]:
    """Compute BRB-MU style reliability scores based on evidence consistency.
    
    v6 FIX: Changed reliability calculation to use:
    - Robust z-scores (median/IQR) with clamping
    - Logistic sigmoid mapping for monotonic [0,1] output
    - Only penalize STRUCTURAL conflict (not "multiple strong evidence")
    
    Parameters
    ----------
    features : dict
        Feature dictionary.
    calibration : dict
        Calibration parameters.
        
    Returns
    -------
    dict
        - reliability: Overall reliability score [0, 1]
        - rel_amp: Amplitude branch reliability
        - rel_freq: Frequency branch reliability
        - rel_ref: Reference branch reliability
        - components: Detailed z-scores per feature
    """
    # Get feature values with defaults
    def _get_f(name, default=0.0):
        return float(features.get(name, default))
    
    def _robust_z(value, median, iqr, zcap=5.0):
        """Compute robust z-score, clamped to [-zcap, zcap]."""
        if iqr <= 1e-12:
            return 0.0
        z = (value - median) / (iqr + 1e-12)
        return max(-zcap, min(zcap, z))
    
    def _sigmoid(x, a0=2.0, a1=0.5):
        """Logistic mapping: high x -> low reliability (for evidence strength)."""
        return 1.0 / (1.0 + math.exp(-a0 + a1 * x))
    
    # === Compute robust z-scores for each evidence group ===
    # Using normal sample statistics (median, IQR) - approximate values from baseline
    
    # Amplitude group z-scores
    z_env_rate = _robust_z(_get_f('X11', 0), 0.005, 0.02)
    z_env_max = _robust_z(_get_f('X12', 0), 0.08, 0.2)
    z_env_energy = _robust_z(_get_f('X13', 0), 16.0, 50.0)
    z_jump = _robust_z(_get_f('X7', 0), 0.08, 0.15)
    
    amp_zscores = [max(0, z) for z in [z_env_rate, z_env_max, z_env_energy, z_jump]]  # Only positive (abnormal) direction
    
    # Frequency group z-scores
    z_shift = _robust_z(abs(_get_f('X16', 0)), 0.0, 10.0)
    z_warp_s = _robust_z(abs(_get_f('X17', 0)), 0.0, 0.005)
    z_warp_b = _robust_z(abs(_get_f('X18', 0)), 0.0, 0.005)
    z_phase = _robust_z(abs(_get_f('X24', 0)), 0.05, 0.1)
    
    freq_zscores = [max(0, z) for z in [z_shift, z_warp_s, z_warp_b, z_phase]]
    
    # Reference group z-scores
    z_low_band = _robust_z(abs(_get_f('X14', 0)), 0.027, 0.05)
    
    ref_zscores = [max(0, z_low_band)]
    
    # === Compute group-level reliability using sigmoid ===
    # Higher mean |z| in group -> lower reliability (more uncertain)
    # But CONSISTENT strong evidence (single group) is still reliable
    
    def group_reliability_v2(zscores: List[float]) -> float:
        """Compute reliability for a group.
        
        Logic: reliability is HIGH when evidence is:
        - Normal (all z near 0) -> rel = 1.0
        - Clearly abnormal (single strong evidence) -> rel = 0.9
        
        Reliability is LOW when:
        - Evidence is noisy/inconsistent within group
        """
        if not zscores:
            return 1.0
        
        max_z = max(zscores)
        mean_z = sum(zscores) / len(zscores)
        
        # If evidence is clearly normal or clearly abnormal, high reliability
        if max_z < 0.5:
            return 1.0  # Normal range - very reliable
        elif max_z > 2.0 and mean_z > 1.0:
            # Strong consistent evidence - reliable
            return 0.9
        elif max_z > 1.5:
            # Moderate evidence - slightly less reliable
            return 0.85
        else:
            # Gray zone - less reliable
            return 0.8
    
    rel_amp = group_reliability_v2(amp_zscores)
    rel_freq = group_reliability_v2(freq_zscores)
    rel_ref = group_reliability_v2(ref_zscores)
    
    # === v6 FIX: Only penalize STRUCTURAL conflict ===
    # Conflict: high coverage (normal-like) but jump_energy extreme (shape anomaly)
    # This is suspicious noise - NOT "multiple groups have strong evidence"
    
    coverage = _get_f('X11', 0)  # env_overrun_rate - low = high coverage
    jump_energy = _get_f('X7', 0)
    
    structural_conflict = False
    conflict_penalty = 0.0
    
    # Structural conflict: appears normal (low violation) but has shape anomaly
    if coverage < 0.02 and jump_energy > 0.5:
        structural_conflict = True
        conflict_penalty = 0.15
    
    # === Overall reliability ===
    overall_reliability = (rel_amp + rel_freq + rel_ref) / 3.0 - conflict_penalty
    overall_reliability = max(0.5, min(1.0, overall_reliability))  # Floor at 0.5
    
    return {
        'reliability': overall_reliability,
        'rel_amp': rel_amp,
        'rel_freq': rel_freq,
        'rel_ref': rel_ref,
        'conflict_penalty': conflict_penalty,
        'structural_conflict': structural_conflict,
        'components': {
            'amp_zscores': amp_zscores,
            'freq_zscores': freq_zscores,
            'ref_zscores': ref_zscores,
        }
    }


def softmax_with_temperature(values: list, alpha: float = 2.0) -> list:
    """带温度参数的softmax函数。
    
    对应小论文式(2)的softmax概率校准。
    
    Parameters
    ----------
    values : list
        输入值列表。
    alpha : float
        温度参数，越大分布越锐利，越小越平滑。
        
    Returns
    -------
    list
        归一化后的概率分布。
    """
    # 应用温度缩放
    scaled = [v * alpha for v in values]
    
    # 数值稳定性：减去最大值
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    
    # 归一化
    total = sum(exp_vals) + 1e-12
    return [v / total for v in exp_vals]


def compute_overall_score(features: Dict[str, float]) -> float:
    """计算整体异常度分数。
    
    用于正常状态识别，综合评估所有特征的异常程度。
    
    Parameters
    ----------
    features : dict
        输入特征字典。
        
    Returns
    -------
    float
        整体异常度分数 [0, 1]。
    """
    # 特征权重
    weights = {
        'X1': 0.15, 'X2': 0.12, 'X3': 0.08, 'X4': 0.10, 'X5': 0.10,
        'X6': 0.05, 'X7': 0.08, 'X8': 0.05, 'X9': 0.03, 'X10': 0.04,
        'X11': 0.08, 'X12': 0.05, 'X13': 0.05, 'X14': 0.02, 'X15': 0.02,
        'X16': 0.03, 'X17': 0.02, 'X18': 0.02,
        'X19': 0.02, 'X20': 0.02, 'X21': 0.01, 'X22': 0.01,
    }
    
    # 归一化参数
    norm_params = {
        'X1': (0.02, 0.5), 'X2': (0.002, 0.05), 'X3': (1e-12, 1e-9),
        'X4': (5e5, 3e7), 'X5': (0.01, 0.35), 'X6': (0.001, 0.03),
        'X7': (0.05, 2.0), 'X8': (0.01, 1.0), 'X9': (1e3, 1e5),
        'X10': (0.02, 0.5), 'X11': (0.01, 0.3), 'X12': (0.5, 5.0),
        'X13': (0.1, 10.0), 'X14': (0.01, 1.0), 'X15': (0.01, 0.5),
        'X16': (0.001, 0.1), 'X17': (0.001, 0.05), 'X18': (0.001, 0.05),
        'X19': (1e-12, 1e-10), 'X20': (0.5, 5.0), 'X21': (1, 20), 'X22': (0.1, 0.8),
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        if key in features:
            raw_value = abs(float(features[key]))
            lower, upper = norm_params.get(key, (0.0, 1.0))
            
            # 归一化到 [0, 1]
            norm_value = max(0.0, min(1.0, (raw_value - lower) / (upper - lower + 1e-12)))
            
            weighted_sum += weight * norm_value
            total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def aggregate_system_results(
    features: Dict[str, float],
    alpha: float = 2.0,
    overall_threshold: float = 0.15,
    max_prob_threshold: float = 0.3,
    calibration: Optional[Dict] = None
) -> Dict:
    """聚合三个子BRB的推理结果，输出系统级诊断。
    
    对应小论文系统级推理的聚合逻辑。
    Enhanced with evidence gating and temperature softmax.
    
    Parameters
    ----------
    features : dict
        完整的特征字典。
    alpha : float
        Softmax温度参数。
    overall_threshold : float
        整体异常度阈值，低于此值判定为正常。
    max_prob_threshold : float
        最大概率阈值，低于此值判定为正常。
    calibration : dict, optional
        Calibration parameters for evidence gating.
        
    Returns
    -------
    dict
        聚合后的诊断结果，包含：
        - probabilities: 四分类概率 {正常, 幅度失准, 频率失准, 参考电平失准}
        - max_prob: 最大概率值
        - predicted_class: 预测类别
        - is_normal: 是否为正常状态
        - uncertainty: 不确定度
        - overall_score: 整体异常度
        - sub_brb_results: 各子BRB的详细结果
    """
    # Load calibration if not provided
    if calibration is None:
        calibration = load_calibration()
    
    # Apply calibration to parameters
    alpha = calibration.get('alpha', alpha)
    overall_threshold = calibration.get('overall_threshold', overall_threshold)
    max_prob_threshold = calibration.get('max_prob_threshold', max_prob_threshold)
    
    # 执行三个子BRB推理
    amp_result = amp_brb_infer(features, alpha)
    freq_result = freq_brb_infer(features, alpha)
    ref_result = ref_brb_infer(features, alpha)
    
    # 获取各子BRB的激活度
    activations = [
        amp_result['activation'],
        freq_result['activation'],
        ref_result['activation'],
    ]

    branch_weights = calibration.get("branch_weights", {"amp": 1.0, "freq": 1.0, "ref": 1.0})
    activations = [
        activations[0] * float(branch_weights.get("amp", 1.0)),
        activations[1] * float(branch_weights.get("freq", 1.0)),
        activations[2] * float(branch_weights.get("ref", 1.0)),
    ]
    
    # Apply evidence gating to prevent amp from absorbing everything
    amp_boost, freq_boost, ref_boost = compute_evidence_gating(features, calibration)
    
    # Boost activations based on their specific evidence
    gated_activations = [
        activations[0] + amp_boost,        # amp: with boost
        activations[1] + freq_boost,       # freq: boost if freq-specific evidence
        activations[2] + ref_boost,        # ref: boost if ref-specific evidence
    ]
    
    # 计算整体异常度
    overall_score = compute_overall_score(features)
    
    # 计算故障概率分布 with temperature softmax
    fault_probs = softmax_with_temperature(gated_activations, alpha)
    
    # 正常状态识别
    normal_weight = 0.0
    
    # 策略1：整体异常度低于阈值
    if overall_score < overall_threshold:
        normal_weight = 1.0 - overall_score / (overall_threshold + 1e-12)
    
    # 策略2：最大故障概率低于阈值
    max_fault_prob = max(fault_probs)
    if max_fault_prob < max_prob_threshold:
        normal_weight = max(normal_weight, max_prob_threshold - max_fault_prob)
    
    # 调整故障概率（考虑正常权重）
    fault_scale = max(0.0, 1.0 - normal_weight)
    scaled_faults = [p * fault_scale for p in fault_probs]
    
    # 构建最终概率分布
    total = normal_weight + sum(scaled_faults)
    if total <= 1e-12:
        probabilities = {'正常': 1.0, '幅度失准': 0.0, '频率失准': 0.0, '参考电平失准': 0.0}
    else:
        probabilities = {
            '正常': normal_weight / total,
            '幅度失准': scaled_faults[0] / total,
            '频率失准': scaled_faults[1] / total,
            '参考电平失准': scaled_faults[2] / total,
        }
    
    # 确定预测类别
    max_prob = max(probabilities.values())
    predicted_class = max(probabilities, key=probabilities.get)
    
    # 判断是否为正常状态
    is_normal = (
        probabilities.get('正常', 0.0) >= 0.5 or 
        max_fault_prob < max_prob_threshold or
        overall_score < overall_threshold
    )
    
    # 计算不确定度
    uncertainty = 1.0 - max_prob
    
    return {
        'probabilities': probabilities,
        'max_prob': max_prob,
        'predicted_class': predicted_class,
        'is_normal': is_normal,
        'uncertainty': uncertainty,
        'overall_score': overall_score,
        'evidence_gating': {
            'amp_boost': amp_boost,
            'freq_boost': freq_boost,
            'ref_boost': ref_boost,
        },
        'sub_brb_results': {
            'amp': amp_result,
            'freq': freq_result,
            'ref': ref_result,
        },
    }


def system_level_infer_with_sub_brbs(
    features: Dict[str, float],
    alpha: float = 2.0,
    overall_threshold: float = 0.15,
    max_prob_threshold: float = 0.3,
    use_feature_routing: bool = True,
    calibration: Optional[Dict] = None
) -> Dict:
    """使用子BRB架构的系统级推理入口（v6：修复可靠度机制）。
    
    v6 FIX: Reliability mechanism now ONLY triggers in uncertain situations:
    - pmax < 0.55 OR margin < 0.15 OR anchor_score in gray zone
    - Clear samples use original logic (no reliability adjustment)
    - Adaptive temperature has an upper limit (no more than 1.3x alpha_base)
    
    这是优化后的系统级推理接口，支持：
    1. Stage-0 Normal Anchor with SOFT GATING (no bypass!)
    2. 特征分流到对应的子BRB
    3. Evidence gating + temperature softmax
    4. Normal logit competes with fault logits
    5. 四分类softmax输出
    6. Reliability weighting ONLY for uncertain samples (v6 FIX)
    
    Parameters
    ----------
    features : dict
        输入特征字典。
    alpha : float
        Softmax温度参数。
    overall_threshold : float
        整体异常度阈值（已被T_low/T_high替代，保留兼容）。
    max_prob_threshold : float
        最大概率阈值（已被软门控替代，保留兼容）。
    use_feature_routing : bool
        是否启用特征分流（默认True）。
    calibration : dict, optional
        Calibration parameters.
        
    Returns
    -------
    dict
        系统级诊断结果。
    """
    # Load calibration if not provided
    if calibration is None:
        calibration = load_calibration()
    
    # Apply calibration to parameters
    alpha_base = calibration.get('alpha_base', calibration.get('alpha', alpha))
    gamma = calibration.get('gamma', 0.5)  # Reliability temperature factor
    
    # Stage-0: Normal Anchor Detection (v2: SOFT GATING)
    anchor_result = None
    normal_logit = 0.0
    anchor_score = 0.0
    T_low = calibration.get('T_low', 0.10)
    T_high = calibration.get('T_high', 0.35)
    
    try:
        from .normal_anchor import infer_normal_anchor
        
        anchor_result = infer_normal_anchor(features, calibration)
        
        # v2: NO BYPASS - get Normal logit instead
        normal_logit = anchor_result.get('normal_logit', 0.0)
        anchor_score = anchor_result.get('anchor_score', 0.0)
        
    except ImportError:
        anchor_result = None
        normal_logit = 0.0
        anchor_score = 0.0
    
    # === v6: Compute reliability FIRST, but use it conditionally ===
    reliability_info = compute_reliability(features, calibration)
    reliability = reliability_info['reliability']
    rel_amp = reliability_info['rel_amp']
    rel_freq = reliability_info['rel_freq']
    rel_ref = reliability_info['rel_ref']
    
    # === STEP 1: Execute sub-BRBs with BASE alpha (no reliability adjustment yet) ===
    amp_result = amp_brb_infer(features, alpha_base)
    freq_result = freq_brb_infer(features, alpha_base)
    ref_result = ref_brb_infer(features, alpha_base)
    
    # 获取各子BRB的激活度
    activations = [
        amp_result['activation'],
        freq_result['activation'],
        ref_result['activation']
    ]
    
    # Apply evidence gating (base boosts)
    amp_boost, freq_boost, ref_boost = compute_evidence_gating(features, calibration)
    
    normal_prior_k = calibration.get("normal_prior_k", 1.0)
    normal_logit *= normal_prior_k

    # === STEP 2: Build initial logits WITHOUT reliability weighting ===
    logits_base = [
        normal_logit,                    # Normal: from anchor
        activations[0] + amp_boost,      # Amp: with evidence boost
        activations[1] + freq_boost,     # Freq: with evidence boost
        activations[2] + ref_boost,      # Ref: with evidence boost
    ]
    
    # === STEP 3: First-pass softmax to check if sample is "certain" ===
    probs_base = softmax_with_temperature(logits_base, alpha_base)
    pmax_base = max(probs_base)
    sorted_probs = sorted(probs_base, reverse=True)
    margin_base = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    
    # === v6 FIX: Only apply reliability mechanism when UNCERTAIN ===
    is_in_gray_zone = T_low < anchor_score < T_high
    pmax_threshold = calibration.get("pmax_threshold", 0.55)
    margin_threshold = calibration.get("margin_threshold", 0.15)
    is_low_confidence = pmax_base < pmax_threshold
    is_low_margin = margin_base < margin_threshold
    
    use_reliability = is_in_gray_zone or is_low_confidence or is_low_margin
    
    if use_reliability:
        # === UNCERTAIN SAMPLE: Apply reliability adjustments ===
        
        # v6 FIX: Adaptive temperature with CAPPED gamma effect
        # gamma_effective is limited so alpha_eff <= alpha_base * ALPHA_MAX_MULTIPLIER
        gamma_effective = min(gamma, GAMMA_MAX_EFFECT)
        alpha_eff = alpha_base * (1.0 + gamma_effective * (1.0 - reliability))
        alpha_eff = min(alpha_eff, alpha_base * ALPHA_MAX_MULTIPLIER)  # Hard cap
        
        # Apply reliability-weighted boosts
        freq_boost_weighted = freq_boost * rel_freq
        ref_boost_weighted = ref_boost * rel_ref
        amp_boost_weighted = amp_boost * rel_amp
        
        # Rebuild logits with reliability weighting
        logits = [
            normal_logit,
            activations[0] + amp_boost_weighted,
            activations[1] + freq_boost_weighted,
            activations[2] + ref_boost_weighted,
        ]
        
        # Final softmax with adaptive temperature
        fault_probs = softmax_with_temperature(logits, alpha_eff)
        reliability_applied = True
    else:
        # === CERTAIN SAMPLE: Use base logits and temperature ===
        logits = logits_base
        alpha_eff = alpha_base
        fault_probs = probs_base
        reliability_applied = False
    
    # Build probability dictionary
    probabilities = {
        '正常': fault_probs[0],
        '幅度失准': fault_probs[1],
        '频率失准': fault_probs[2],
        '参考电平失准': fault_probs[3],
    }
    
    # Determine prediction
    max_prob = max(probabilities.values())
    predicted_class = max(probabilities, key=probabilities.get)
    is_normal = predicted_class == '正常'
    
    # Compute overall score for debugging
    overall_score = anchor_result.get('anchor_score', 0.0) if anchor_result else compute_overall_score(features)
    
    # Uncertainty calculation
    base_uncertainty = 1.0 - max_prob
    if use_reliability:
        # Only add reliability factor for uncertain samples
        uncertainty = base_uncertainty + 0.1 * (1.0 - reliability)
    else:
        uncertainty = base_uncertainty
    uncertainty = min(1.0, uncertainty)
    
    result = {
        'probabilities': probabilities,
        'max_prob': max_prob,
        'predicted_class': predicted_class,
        'is_normal': is_normal,
        'uncertainty': uncertainty,
        'overall_score': overall_score,
        'logits': {
            'normal': logits[0],
            'amp': logits[1],
            'freq': logits[2],
            'ref': logits[3],
        },
        'evidence_gating': {
            'amp_boost': amp_boost if not use_reliability else amp_boost * rel_amp,
            'freq_boost': freq_boost if not use_reliability else freq_boost * rel_freq,
            'ref_boost': ref_boost if not use_reliability else ref_boost * rel_ref,
        },
        # v6: Reliability info with applied flag
        'reliability': {
            'overall': reliability,
            'rel_amp': rel_amp,
            'rel_freq': rel_freq,
            'rel_ref': rel_ref,
            'alpha_eff': alpha_eff,
            'reliability_applied': reliability_applied,
            'trigger_reason': 'gray_zone' if is_in_gray_zone else 'low_confidence' if is_low_confidence else 'low_margin' if is_low_margin else 'none',
        },
        'sub_brb_results': {
            'amp': amp_result,
            'freq': freq_result,
            'ref': ref_result,
        },
    }
    
    if anchor_result is not None:
        result['normal_anchor'] = anchor_result
    
    return result
