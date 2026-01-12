#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
频率异常子BRB推理模块 (Frequency Sub-BRB)
==========================================
对应小论文系统级诊断中的频率失准推理子模块。

本模块负责：
1. 接收频率相关特征（X4, X8, X9, X14, X15, X16, X17, X18）
2. 执行针对频率异常的BRB推理
3. 输出频率失准的概率和置信度
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


# 频率相关特征及其归一化参数
# **关键修正**: 频率子BRB应该主要依赖 X16/X17/X18（真正的频移/缩放特征）
# X4, X8, X9, X14, X15 对幅度故障也敏感，不能作为频率故障的主要指标
FREQ_FEATURE_PARAMS = {
    'X4': (0.0, 0.2),       # 频率标度非线性度 - 范围扩大以降低敏感度
    'X8': (0.0, 0.15),      # 本振泄漏 - 范围扩大
    'X9': (0.0, 0.05),      # 调谐线性度残差 - 范围扩大
    'X14': (-0.5, 0.5),     # 低频段残差均值 - 范围扩大
    'X15': (0.0, 0.15),     # 高频段残差标准差 - 范围扩大
    'X16': (-0.05, 0.05),   # 互相关滞后/频移 - 这是关键频率特征
    'X17': (-0.1, 0.1),     # 频轴缩放因子 - 关键频率特征
    'X18': (-0.1, 0.1),     # 频轴平移因子 - 关键频率特征
}

# 特征权重 - **大幅增加X16/X17/X18权重，降低其他特征权重**
# 因为X4/X8/X9/X14/X15对Amp故障也敏感
FREQ_FEATURE_WEIGHTS = {
    'X4': 0.05,   # 频率标度非线性 - 降低权重
    'X8': 0.05,   # 本振泄漏 - 降低权重
    'X9': 0.05,   # 调谐线性度残差 - 降低权重
    'X14': 0.05,  # 低频段残差 - 降低权重
    'X15': 0.05,  # 高频段残差 - 降低权重
    'X16': 0.30,  # 频移 - **关键指标，大幅增加**
    'X17': 0.25,  # 频率缩放 - **关键指标**
    'X18': 0.20,  # 频率平移 - **关键指标**
}


def _get_feature_value(features: Dict[str, float], key: str, default: float = 0.0) -> float:
    """安全获取特征值。"""
    if key in features:
        return float(features[key])
    # 尝试其他可能的键名
    alt_keys = {
        'X4': ['freq_scale_nonlinearity', 'df', 'frequency_nonlinearity'],
        'X8': ['lo_leakage', 'local_oscillator_leakage'],
        'X9': ['tuning_linearity_residual'],
        'X14': ['band_residual_low', 'low_band_residual'],
        'X15': ['band_residual_high_std', 'high_band_residual_std'],
        'X16': ['corr_shift_bins', 'frequency_shift'],
        'X17': ['warp_scale', 'frequency_scale'],
        'X18': ['warp_bias', 'frequency_bias'],
    }
    for alt_key in alt_keys.get(key, []):
        if alt_key in features:
            return float(features[alt_key])
    return default


def compute_freq_scores(features: Dict[str, float]) -> Dict[str, float]:
    """计算频率相关特征的归一化分数。
    
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
    for key, (lower, upper) in FREQ_FEATURE_PARAMS.items():
        raw_value = abs(_get_feature_value(features, key))
        scores[key] = _normalize_feature(raw_value, lower, upper)
    return scores


def compute_freq_match_degrees(scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    """计算频率特征的属性匹配度。
    
    Returns
    -------
    dict
        每个特征的 (Low, Normal, High) 隶属度。
    """
    return {name: _triangular_membership(value, 0.15, 0.35, 0.7) for name, value in scores.items()}


def freq_brb_infer(features: Dict[str, float], alpha: float = 2.0) -> Dict[str, float]:
    """执行频率异常的BRB推理。
    
    对应小论文系统级推理中频率失准的子BRB。
    
    Parameters
    ----------
    features : dict
        频率相关特征字典（可以是完整特征，会自动提取相关部分）。
    alpha : float
        Softmax温度参数，用于控制概率分布的锐度。
        
    Returns
    -------
    dict
        推理结果，包含：
        - probability: 频率失准概率
        - activation: 规则激活度
        - confidence: 置信度
        - feature_contributions: 各特征的贡献度
    """
    # 计算归一化分数
    scores = compute_freq_scores(features)
    
    # 计算属性匹配度
    match_degrees = compute_freq_match_degrees(scores)
    
    # 计算加权激活度
    weighted_activation = 0.0
    total_weight = 0.0
    feature_contributions = {}
    
    for key, weight in FREQ_FEATURE_WEIGHTS.items():
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
