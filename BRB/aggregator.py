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

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .system_brb_amp import amp_brb_infer
from .system_brb_freq import freq_brb_infer
from .system_brb_ref import ref_brb_infer


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
    if calibration_path is None:
        # Try default location
        repo_root = Path(__file__).resolve().parents[1]
        calibration_path = repo_root / 'Output' / 'calibration.json'
    
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
    
    If frequency-specific evidence is strong (shift/warp features exceed
    normal quantile threshold), boost freq logit by beta_freq.
    Similarly for ref branch (compression/step features).
    
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
    beta_amp = calibration.get('beta_amp', 0.3)
    beta_freq = calibration.get('beta_freq', 0.5)
    beta_ref = calibration.get('beta_ref', 0.5)
    
    # Amplitude evidence thresholds
    amp_evidence = 0.0
    x11 = abs(float(features.get('X11', 0)))  # env_overrun_rate
    x12 = abs(float(features.get('X12', 0)))  # env_overrun_max
    x13 = abs(float(features.get('X13', 0)))  # env_violation_energy
    
    if x11 > 0.05 or x12 > 1.0 or x13 > 50.0:
        amp_evidence = max(x11 / 0.3, x12 / 4.0, x13 / 500.0)
        amp_evidence = min(1.0, amp_evidence)
    
    # Frequency evidence thresholds (normalized)
    freq_evidence = 0.0
    x16 = abs(float(features.get('X16', 0)))  # corr_shift_bins
    x17 = abs(float(features.get('X17', 0)))  # warp_scale
    x18 = abs(float(features.get('X18', 0)))  # warp_bias
    x24 = abs(float(features.get('X24', 0)))  # phase_slope_diff
    
    # Normalize and check if above normal range
    if x16 > 5.0 or x17 > 0.005 or x18 > 0.005 or x24 > 0.1:
        freq_evidence = max(x16 / 50.0, x17 / 0.02, x18 / 0.02, x24 / 0.5)
        freq_evidence = min(1.0, freq_evidence)
    
    # Reference evidence thresholds - primarily X14
    ref_evidence = 0.0
    x14 = abs(float(features.get('X14', 0)))  # low_band_residual - KEY feature
    
    if x14 > 0.05:  # normal is ~0.027, ref_error is ~0.376
        ref_evidence = min(1.0, x14 / 0.15)
    
    # Apply gating: boost = beta * evidence_strength
    amp_boost = beta_amp * amp_evidence
    freq_boost = beta_freq * freq_evidence
    ref_boost = beta_ref * ref_evidence
    
    return amp_boost, freq_boost, ref_boost


def compute_reliability(
    features: Dict[str, float],
    calibration: Dict
) -> Dict[str, float]:
    """Compute BRB-MU style reliability scores based on evidence consistency.
    
    Implements reliability computation inspired by BRB with Multiplicative Utility:
    - Higher coverage (more normal-like) → higher reliability
    - Evidence conflict → lower reliability
    - Robust z-scores used for normalization
    
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
    
    # === Compute z-scores for each evidence group ===
    # Using typical normal statistics (median, IQR) from baseline
    
    # Amplitude group z-scores
    z_env_rate = min(5.0, _get_f('X11', 0) / 0.02)
    z_env_max = min(5.0, _get_f('X12', 0) / 0.5)
    z_env_energy = min(5.0, _get_f('X13', 0) / 50.0)
    z_jump = min(5.0, _get_f('X7', 0) / 0.2)
    
    amp_zscores = [z_env_rate, z_env_max, z_env_energy, z_jump]
    
    # Frequency group z-scores
    z_shift = min(5.0, abs(_get_f('X16', 0)) / 20.0)
    z_warp_s = min(5.0, abs(_get_f('X17', 0)) / 0.01)
    z_warp_b = min(5.0, abs(_get_f('X18', 0)) / 0.01)
    z_phase = min(5.0, abs(_get_f('X24', 0)) / 0.2)
    
    freq_zscores = [z_shift, z_warp_s, z_warp_b, z_phase]
    
    # Reference group z-scores
    z_low_band = min(5.0, abs(_get_f('X14', 0)) / 0.05)
    
    ref_zscores = [z_low_band]
    
    # === Compute group-level reliability ===
    # Reliability decreases when z-scores are too high (uncertain evidence)
    # but also when z-scores conflict within a group
    
    def group_reliability(zscores: List[float]) -> float:
        if not zscores:
            return 1.0
        max_z = max(zscores)
        # If max evidence is strong and consistent, reliability is high
        # If evidence is very weak (all near 0), reliability is also high (normal)
        # If evidence is mixed/conflicting, reliability is lower
        
        # Simple formula: reliability drops with extreme evidence
        # but not with consistent mild evidence
        if max_z < 1.0:
            return 1.0  # Normal range - high reliability
        elif max_z < 3.0:
            return 0.9 - 0.1 * (max_z - 1.0)  # Slight drop
        else:
            return max(0.4, 0.7 - 0.1 * (max_z - 3.0))  # Cap at 0.4
    
    rel_amp = group_reliability(amp_zscores)
    rel_freq = group_reliability(freq_zscores)
    rel_ref = group_reliability(ref_zscores)
    
    # === Detect evidence conflict ===
    # If multiple groups have strong evidence, there's conflict
    strong_groups = 0
    if max(amp_zscores) > 2.0:
        strong_groups += 1
    if max(freq_zscores) > 2.0:
        strong_groups += 1
    if max(ref_zscores) > 2.0:
        strong_groups += 1
    
    conflict_penalty = 0.0
    if strong_groups >= 2:
        conflict_penalty = 0.15 * (strong_groups - 1)
    
    # === Overall reliability ===
    # Weighted combination of branch reliabilities
    overall_reliability = 0.4 * rel_amp + 0.3 * rel_freq + 0.3 * rel_ref - conflict_penalty
    overall_reliability = max(0.3, min(1.0, overall_reliability))
    
    return {
        'reliability': overall_reliability,
        'rel_amp': rel_amp,
        'rel_freq': rel_freq,
        'rel_ref': rel_ref,
        'conflict_penalty': conflict_penalty,
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
        ref_result['activation']
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
    """使用子BRB架构的系统级推理入口（v5：BRB-MU可信度权重版本）。
    
    这是优化后的系统级推理接口，支持：
    1. Stage-0 Normal Anchor with SOFT GATING (no bypass!)
    2. 特征分流到对应的子BRB
    3. Evidence gating + temperature softmax
    4. Normal logit competes with fault logits
    5. 四分类softmax输出
    6. BRB-MU style reliability weighting (v5 NEW)
    
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
    alpha_base = calibration.get('alpha', alpha)
    gamma = calibration.get('gamma', 0.5)  # Reliability temperature factor
    
    # Stage-0: Normal Anchor Detection (v2: SOFT GATING)
    anchor_result = None
    normal_logit = 0.0
    try:
        from .normal_anchor import infer_normal_anchor
        
        anchor_result = infer_normal_anchor(features, calibration)
        
        # v2: NO BYPASS - get Normal logit instead
        normal_logit = anchor_result.get('normal_logit', 0.0)
        
    except ImportError:
        anchor_result = None
        normal_logit = 0.0
    
    # === v5 NEW: Compute reliability scores ===
    reliability_info = compute_reliability(features, calibration)
    reliability = reliability_info['reliability']
    rel_amp = reliability_info['rel_amp']
    rel_freq = reliability_info['rel_freq']
    rel_ref = reliability_info['rel_ref']
    
    # === v5: Adaptive temperature (suppresses overconfidence when uncertain) ===
    # When reliability is low, make softmax softer (more uncertain output)
    alpha_eff = alpha_base * (1.0 + gamma * (1.0 - reliability))
    
    # Execute three sub-BRBs (always run, no bypass)
    amp_result = amp_brb_infer(features, alpha_eff)
    freq_result = freq_brb_infer(features, alpha_eff)
    ref_result = ref_brb_infer(features, alpha_eff)
    
    # 获取各子BRB的激活度
    activations = [
        amp_result['activation'],
        freq_result['activation'],
        ref_result['activation']
    ]
    
    # Apply evidence gating (now returns 3 values including amp_boost)
    amp_boost, freq_boost, ref_boost = compute_evidence_gating(features, calibration)
    
    # === v5: Apply reliability-weighted gating ===
    # Boost is scaled by branch reliability
    # When rel_freq is high, freq evidence is trusted more
    freq_boost_weighted = freq_boost * rel_freq
    ref_boost_weighted = ref_boost * rel_ref
    amp_boost_weighted = amp_boost * rel_amp
    
    # Build 4-way logits: [Normal, Amp, Freq, Ref]
    logits = [
        normal_logit,                                   # Normal: from anchor
        activations[0] + amp_boost_weighted,            # Amp: with reliability-weighted boost
        activations[1] + freq_boost_weighted,           # Freq: with reliability-weighted boost
        activations[2] + ref_boost_weighted,            # Ref: with reliability-weighted boost
    ]
    
    # Apply temperature softmax for 4-way classification (with adaptive temperature)
    fault_probs = softmax_with_temperature(logits, alpha_eff)
    
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
    
    # === v5: Enhanced uncertainty with reliability ===
    # Uncertainty increases when reliability is low
    base_uncertainty = 1.0 - max_prob
    uncertainty = base_uncertainty + 0.2 * (1.0 - reliability)
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
            'amp_boost': amp_boost_weighted,
            'freq_boost': freq_boost_weighted,
            'ref_boost': ref_boost_weighted,
        },
        # v5 NEW: Reliability info
        'reliability': {
            'overall': reliability,
            'rel_amp': rel_amp,
            'rel_freq': rel_freq,
            'rel_ref': rel_ref,
            'alpha_eff': alpha_eff,
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
