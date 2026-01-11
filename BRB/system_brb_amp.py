#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幅度异常子BRB推理模块 (Amplitude Sub-BRB)
==========================================
对应小论文系统级诊断中的幅度失准推理子模块。

本模块负责：
1. 接收幅度相关特征（X1, X2, X5, X6, X7, X10, X11-X13, X19-X22）
2. 执行针对幅度异常的BRB推理
3. 输出幅度失准的概率和置信度
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


# 幅度相关特征及其归一化参数
# 特征现在是residual-based（相对baseline的偏差），正常样本围绕0分布
# 参数基于实际正常样本统计: (median, iqr) -> 归一化到 3σ范围
# 使用 [median - 3*iqr, median + 3*iqr] 作为正常范围
AMP_FEATURE_PARAMS = {
    'X1': (-0.2, 0.2),      # 整体幅度偏移（residual均值），正常~0
    'X2': (0.0, 0.001),     # 带内平坦度（residual方差），正常~0.0001
    'X5': (0.0, 0.6),       # 幅度缩放一致性，正常~0.25
    'X6': (0.0, 0.001),     # 纹波方差
    'X7': (0.0, 0.15),      # 增益非线性（最大步进）
    'X10': (0.0, 0.02),     # 频段幅度一致性
    'X11': (0.0, 0.02),     # 包络超出率，正常~0.004
    'X12': (0.0, 0.1),      # 最大包络违规，正常~0.03
    'X13': (0.0, 100.0),    # 包络违规能量
    'X19': (-0.001, 0.001), # 低频段斜率
    'X20': (-5.0, 5.0),     # 去趋势残差峰度
    'X21': (0, 60),         # 残差峰值数，正常~23
    'X22': (0.0, 0.5),      # 残差主频能量占比，正常~0.25
}

# 特征权重 - 增加X11、X12、X13的权重（关键幅度异常指标）
AMP_FEATURE_WEIGHTS = {
    'X1': 0.15,   # 整体幅度偏移
    'X2': 0.10,   # 带内平坦度
    'X5': 0.08,   # 幅度缩放一致性
    'X6': 0.06,   # 纹波
    'X7': 0.08,   # 增益非线性
    'X10': 0.05,  # 频段幅度一致性
    'X11': 0.15,  # 包络超出率 - 关键指标
    'X12': 0.12,  # 最大包络违规 - 关键指标
    'X13': 0.10,  # 包络违规能量 - 关键指标
    'X19': 0.03,  # 低频段斜率
    'X20': 0.03,  # 去趋势残差峰度
    'X21': 0.02,  # 残差峰值数
    'X22': 0.03,  # 残差主频能量占比
}


def _get_feature_value(features: Dict[str, float], key: str, default: float = 0.0) -> float:
    """安全获取特征值。"""
    if key in features:
        return float(features[key])
    # 尝试其他可能的键名
    alt_keys = {
        'X1': ['amplitude_offset', 'bias'],
        'X2': ['inband_flatness', 'ripple_var'],
        'X5': ['scale_consistency', 'amp_scale_consistency'],
        'X6': ['ripple_variance'],
        'X7': ['gain_nonlinearity', 'step_score'],
        'X10': ['band_amplitude_consistency'],
        'X11': ['env_overrun_rate', 'viol_rate'],
        'X12': ['env_overrun_max'],
        'X13': ['env_violation_energy'],
        'X19': ['slope_low'],
        'X20': ['kurtosis_detrended'],
        'X21': ['peak_count_residual'],
        'X22': ['ripple_dom_freq_energy'],
    }
    for alt_key in alt_keys.get(key, []):
        if alt_key in features:
            return float(features[alt_key])
    return default


def compute_amp_scores(features: Dict[str, float]) -> Dict[str, float]:
    """计算幅度相关特征的归一化分数。
    
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
    for key, (lower, upper) in AMP_FEATURE_PARAMS.items():
        raw_value = abs(_get_feature_value(features, key))
        scores[key] = _normalize_feature(raw_value, lower, upper)
    return scores


def compute_amp_match_degrees(scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    """计算幅度特征的属性匹配度。
    
    Returns
    -------
    dict
        每个特征的 (Low, Normal, High) 隶属度。
    """
    return {name: _triangular_membership(value, 0.15, 0.35, 0.7) for name, value in scores.items()}


def amp_brb_infer(features: Dict[str, float], alpha: float = 2.0) -> Dict[str, float]:
    """执行幅度异常的BRB推理。
    
    对应小论文系统级推理中幅度失准的子BRB。
    
    Parameters
    ----------
    features : dict
        幅度相关特征字典（可以是完整特征，会自动提取相关部分）。
    alpha : float
        Softmax温度参数，用于控制概率分布的锐度。
        
    Returns
    -------
    dict
        推理结果，包含：
        - probability: 幅度失准概率
        - activation: 规则激活度
        - confidence: 置信度
        - feature_contributions: 各特征的贡献度
    """
    # 计算归一化分数
    scores = compute_amp_scores(features)
    
    # 计算属性匹配度
    match_degrees = compute_amp_match_degrees(scores)
    
    # 计算加权激活度
    # 使用 High 隶属度表示异常程度
    weighted_activation = 0.0
    total_weight = 0.0
    feature_contributions = {}
    
    for key, weight in AMP_FEATURE_WEIGHTS.items():
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
    confidence = abs(probability - 0.5) * 2  # 0.5 表示最不确定，0或1表示最确定
    
    return {
        'probability': probability,
        'activation': activation,
        'confidence': confidence,
        'feature_contributions': feature_contributions,
        'scores': scores,
    }
