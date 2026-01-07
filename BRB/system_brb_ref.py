#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考电平异常子BRB推理模块 (Reference Level Sub-BRB)
====================================================
对应小论文系统级诊断中的参考电平失准推理子模块。

本模块负责：
1. 接收参考电平相关特征（X1, X3, X5, X10, X11, X12, X13）
2. 执行针对参考电平异常的BRB推理
3. 输出参考电平失准的概率和置信度
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


def _triangular_membership(value: float, low: float, center: float, high: float) -> Tuple[float, float, float]:
    """三角隶属度函数，返回 (Low, Normal, High) 隶属度。"""
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


def _normalize_feature(value: float, lower: float, upper: float) -> float:
    """归一化特征值到 [0, 1] 范围。"""
    value = max(lower, min(value, upper))
    return (value - lower) / (upper - lower + 1e-12)


# 参考电平相关特征及其归一化参数
REF_FEATURE_PARAMS = {
    'X1': (0.02, 0.5),      # 整体幅度偏移 - 与参考电平直接相关
    'X3': (1e-12, 1e-9),    # 高频段衰减斜率
    'X5': (0.01, 0.35),     # 幅度缩放一致性
    'X10': (0.02, 0.5),     # 频段幅度一致性
    'X11': (0.01, 0.3),     # 包络超出率
    'X12': (0.5, 5.0),      # 最大包络违规
    'X13': (0.1, 10.0),     # 包络违规能量
}

# 特征权重 - 表示各特征对参考电平异常识别的重要性
REF_FEATURE_WEIGHTS = {
    'X1': 0.25,   # 整体幅度偏移 - 最重要
    'X3': 0.15,   # 高频段衰减斜率
    'X5': 0.15,   # 幅度缩放一致性
    'X10': 0.10,  # 频段幅度一致性
    'X11': 0.15,  # 包络超出率
    'X12': 0.10,  # 最大包络违规
    'X13': 0.10,  # 包络违规能量
}


def _get_feature_value(features: Dict[str, float], key: str, default: float = 0.0) -> float:
    """安全获取特征值。"""
    if key in features:
        return float(features[key])
    # 尝试其他可能的键名
    alt_keys = {
        'X1': ['amplitude_offset', 'bias', 'overall_amplitude_offset'],
        'X3': ['hf_attenuation_slope', 'res_slope', 'high_freq_slope'],
        'X5': ['scale_consistency', 'amp_scale_consistency', 'gain_consistency'],
        'X10': ['band_amplitude_consistency'],
        'X11': ['env_overrun_rate', 'viol_rate', 'envelope_ratio'],
        'X12': ['env_overrun_max', 'max_env_violation'],
        'X13': ['env_violation_energy'],
    }
    for alt_key in alt_keys.get(key, []):
        if alt_key in features:
            return float(features[alt_key])
    return default


def compute_ref_scores(features: Dict[str, float]) -> Dict[str, float]:
    """计算参考电平相关特征的归一化分数。
    
    Parameters
    ----------
    features : dict
        输入特征字典。
        
    Returns
    -------
    dict
        归一化后的特征分数。
    """
    scores = {}
    for key, (lower, upper) in REF_FEATURE_PARAMS.items():
        raw_value = abs(_get_feature_value(features, key))
        scores[key] = _normalize_feature(raw_value, lower, upper)
    return scores


def compute_ref_match_degrees(scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    """计算参考电平特征的属性匹配度。
    
    Returns
    -------
    dict
        每个特征的 (Low, Normal, High) 隶属度。
    """
    return {name: _triangular_membership(value, 0.15, 0.35, 0.7) for name, value in scores.items()}


def ref_brb_infer(features: Dict[str, float], alpha: float = 2.0) -> Dict[str, float]:
    """执行参考电平异常的BRB推理。
    
    对应小论文系统级推理中参考电平失准的子BRB。
    
    Parameters
    ----------
    features : dict
        参考电平相关特征字典（可以是完整特征，会自动提取相关部分）。
    alpha : float
        Softmax温度参数，用于控制概率分布的锐度。
        
    Returns
    -------
    dict
        推理结果，包含：
        - probability: 参考电平失准概率
        - activation: 规则激活度
        - confidence: 置信度
        - feature_contributions: 各特征的贡献度
    """
    # 计算归一化分数
    scores = compute_ref_scores(features)
    
    # 计算属性匹配度
    match_degrees = compute_ref_match_degrees(scores)
    
    # 计算加权激活度
    weighted_activation = 0.0
    total_weight = 0.0
    feature_contributions = {}
    
    for key, weight in REF_FEATURE_WEIGHTS.items():
        if key in match_degrees:
            high_degree = match_degrees[key][2]  # High 隶属度
            weighted_activation += weight * high_degree
            total_weight += weight
            feature_contributions[key] = weight * high_degree
    
    # 归一化激活度
    if total_weight > 0:
        activation = weighted_activation / total_weight
    else:
        activation = 0.0
    
    # 应用 softmax 温度调整
    exp_activation = math.exp(alpha * activation)
    exp_normal = math.exp(alpha * (1.0 - activation))
    
    total_exp = exp_activation + exp_normal
    probability = exp_activation / total_exp if total_exp > 0 else 0.0
    
    # 计算置信度
    confidence = abs(probability - 0.5) * 2
    
    return {
        'probability': probability,
        'activation': activation,
        'confidence': confidence,
        'feature_contributions': feature_contributions,
        'scores': scores,
    }
