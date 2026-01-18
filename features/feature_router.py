#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征分流模块 (Feature Router)
==============================
对应小论文"基于知识驱动规则优化与分层推理的频谱分析仪故障诊断方法"的特征分流策略。

本模块定义特征到异常类型/模块组的映射规则，实现：
1. 系统级特征分流：将特征按幅度类、频率类、参考电平类分流
2. 模块层特征分流：根据不同模块的功能，仅激活相关的特征子集
3. 归一化参数计算与应用

使用方法：
    from features.feature_router import feature_router, robust_z_score_normalize
    
    # 获取特定故障类型的特征子集
    amp_features = feature_router(all_features, 'amp')
    
    # 对特征进行归一化
    normalized_features = robust_z_score_normalize(features, stats)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# ============================================================================
# 系统级特征分流规则
# ============================================================================

# 系统级特征定义 (X1-X10)
SYSTEM_FEATURES = {
    'X1': '整体幅度偏移',
    'X2': '带内平坦度指标',
    'X3': '高频段衰减斜率',
    'X4': '频率标度非线性度',
    'X5': '幅度缩放一致性',
    'X6': '纹波',
    'X7': '增益非线性',
    'X8': '本振泄漏',
    'X9': '调谐线性度残差',
    'X10': '不同频段幅度一致性',
}

# 模块级特征定义 (X11-X15)
MODULE_FEATURES = {
    'X11': '包络超出率 (envelope overrun rate)',
    'X12': '最大包络违规幅度 (max envelope violation)',
    'X13': '包络违规能量 (envelope violation energy)',
    'X14': '低频段残差均值',
    'X15': '高频段残差标准差',
}

# 频率对齐/形变特征 (X16-X18)
FREQUENCY_ALIGNMENT_FEATURES = {
    'X16': '互相关滞后/频移 (corr_shift_bins)',
    'X17': '频轴缩放因子 (warp_scale)',
    'X18': '频轴平移因子 (warp_bias)',
}

# 幅度链路细粒度特征 (X19-X22)
AMPLITUDE_DETAIL_FEATURES = {
    'X19': '低频段斜率 (slope_low)',
    'X20': '去趋势残差峰度 (kurtosis_detrended)',
    'X21': '残差峰值数量 (peak_count_residual)',
    'X22': '残差主频能量占比 (ripple_dom_freq_energy)',
}

# ============================================================================
# 系统级异常类型特征分流
# ============================================================================

SYSTEM_BRANCH_FEATURES = {
    # 幅度失准相关特征
    'amp': ['X1', 'X2', 'X5', 'X6', 'X7', 'X10', 'X11', 'X12', 'X13', 'X19', 'X20', 'X21', 'X22'],
    
    # 频率失准相关特征  
    'freq': ['X4', 'X8', 'X9', 'X14', 'X15', 'X16', 'X17', 'X18'],
    
    # 参考电平失准相关特征
    'ref': ['X1', 'X3', 'X5', 'X10', 'X11', 'X12', 'X13'],
}

# 中文键名映射
FAULT_TYPE_MAPPING = {
    '幅度失准': 'amp',
    '频率失准': 'freq', 
    '参考电平失准': 'ref',
    'amp': 'amp',
    'freq': 'freq',
    'ref': 'ref',
    'amplitude': 'amp',
    'frequency': 'freq',
    'reference': 'ref',
}

# ============================================================================
# 模块层特征分流规则
# ============================================================================

# 模块分组 - 按物理链路和功能相关性
MODULE_GROUPS = {
    # 幅度链路模块组
    'amp_group': [
        '衰减器', '前置放大器', '中频放大器', '数字放大器', 'ADC',
        '数字RBW', '数字检波器', 'VBW滤波器'
    ],
    
    # 频率链路模块组
    'freq_group': [
        '时钟振荡器', '时钟合成与同步网络', '本振源（谐波发生器）', '本振混频组件',
        '高频段YTF滤波器', '高频段混频器', '低频段前置低通滤波器', '低频段第一混频器'
    ],
    
    # 参考电平链路模块组
    'ref_group': [
        '校准源', '存储器', '校准信号开关', '衰减器'
    ],
    
    # 其他/通用模块
    'other_group': [
        '电源模块'
    ],
}

# 模块特征映射 - 每个模块组使用的特征
MODULE_FEATURES_BY_GROUP = {
    'amp_group': ['X1', 'X2', 'X5', 'X6', 'X7', 'X10', 'X11', 'X12', 'X13', 'X19', 'X20', 'X21', 'X22'],
    'freq_group': ['X4', 'X8', 'X9', 'X14', 'X15', 'X16', 'X17', 'X18'],
    'ref_group': ['X1', 'X3', 'X5', 'X10', 'X11', 'X12', 'X13'],
    'other_group': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'],  # 使用基础特征
}

# 异常类型到模块组的映射
FAULT_TYPE_TO_MODULE_GROUP = {
    'amp': 'amp_group',
    'freq': 'freq_group',
    'ref': 'ref_group',
    '幅度失准': 'amp_group',
    '频率失准': 'freq_group',
    '参考电平失准': 'ref_group',
}


# ============================================================================
# 特征分流函数
# ============================================================================

def feature_router(features: Dict[str, float], fault_type: str) -> Dict[str, float]:
    """根据故障类型分流特征，返回相关特征子集。
    
    对应小论文特征分流策略，确保每个BRB模块只接收与其相关的特征。
    
    Parameters
    ----------
    features : dict
        完整特征字典，包含 X1-X22 等特征。
    fault_type : str
        故障类型标识，支持：
        - 'amp' / '幅度失准' / 'amplitude': 幅度异常
        - 'freq' / '频率失准' / 'frequency': 频率异常
        - 'ref' / '参考电平失准' / 'reference': 参考电平异常
        
    Returns
    -------
    dict
        仅包含与该故障类型相关的特征子集。
        
    Raises
    ------
    ValueError
        如果 fault_type 不在预定义列表中。
        
    Examples
    --------
    >>> features = {'X1': 0.5, 'X2': 0.3, 'X4': 0.8, 'X16': 0.1}
    >>> amp_features = feature_router(features, 'amp')
    >>> print(amp_features)  # 只包含幅度相关特征
    {'X1': 0.5, 'X2': 0.3}
    """
    # 标准化故障类型名称
    normalized_type = FAULT_TYPE_MAPPING.get(fault_type)
    if normalized_type is None:
        raise ValueError(
            f"Unknown fault type: '{fault_type}'. "
            f"Expected one of: {list(FAULT_TYPE_MAPPING.keys())}"
        )
    
    # 获取该故障类型的相关特征列表
    relevant_features = SYSTEM_BRANCH_FEATURES.get(normalized_type, [])
    
    # 从完整特征中提取相关子集
    routed_features = {}
    for key in relevant_features:
        if key in features:
            routed_features[key] = features[key]
        else:
            # 尝试其他可能的键名
            alt_keys = _get_alternative_keys(key)
            for alt_key in alt_keys:
                if alt_key in features:
                    routed_features[key] = features[alt_key]
                    break
            else:
                # 如果找不到，使用默认值0
                routed_features[key] = 0.0
    
    return routed_features


def get_module_group_features(features: Dict[str, float], module_group: str) -> Dict[str, float]:
    """获取特定模块组的相关特征。
    
    Parameters
    ----------
    features : dict
        完整特征字典。
    module_group : str
        模块组名称: 'amp_group', 'freq_group', 'ref_group', 'other_group'
        
    Returns
    -------
    dict
        该模块组相关的特征子集。
    """
    relevant_features = MODULE_FEATURES_BY_GROUP.get(module_group, [])
    
    routed_features = {}
    for key in relevant_features:
        if key in features:
            routed_features[key] = features[key]
        else:
            alt_keys = _get_alternative_keys(key)
            for alt_key in alt_keys:
                if alt_key in features:
                    routed_features[key] = features[alt_key]
                    break
            else:
                routed_features[key] = 0.0
    
    return routed_features


def get_modules_for_fault_type(fault_type: str) -> List[str]:
    """获取与特定故障类型相关的模块列表。
    
    Parameters
    ----------
    fault_type : str
        故障类型标识。
        
    Returns
    -------
    list
        相关模块名称列表。
    """
    normalized_type = FAULT_TYPE_MAPPING.get(fault_type, fault_type)
    module_group = FAULT_TYPE_TO_MODULE_GROUP.get(normalized_type, 'other_group')
    return MODULE_GROUPS.get(module_group, [])


def _get_alternative_keys(key: str) -> List[str]:
    """获取特征键的替代名称。"""
    alternatives = {
        'X1': ['amplitude_offset', 'bias', 'overall_amplitude_offset'],
        'X2': ['inband_flatness', 'ripple_var', 'flatness_index'],
        'X3': ['hf_attenuation_slope', 'res_slope', 'high_freq_slope'],
        'X4': ['freq_scale_nonlinearity', 'df', 'frequency_nonlinearity'],
        'X5': ['scale_consistency', 'amp_scale_consistency', 'gain_consistency'],
        'X6': ['ripple_variance', 'ripple'],
        'X7': ['gain_nonlinearity', 'step_score'],
        'X8': ['lo_leakage', 'local_oscillator_leakage'],
        'X9': ['tuning_linearity_residual'],
        'X10': ['band_amplitude_consistency'],
        'X11': ['env_overrun_rate', 'viol_rate', 'envelope_ratio'],
        'X12': ['env_overrun_max', 'max_env_violation'],
        'X13': ['env_violation_energy'],
        'X14': ['band_residual_low', 'low_band_residual'],
        'X15': ['band_residual_high_std', 'high_band_residual_std'],
        'X16': ['corr_shift_bins', 'frequency_shift'],
        'X17': ['warp_scale', 'frequency_scale'],
        'X18': ['warp_bias', 'frequency_bias'],
        'X19': ['slope_low', 'low_freq_slope'],
        'X20': ['kurtosis_detrended', 'residual_kurtosis'],
        'X21': ['peak_count_residual', 'ripple_peak_count'],
        'X22': ['ripple_dom_freq_energy', 'dominant_freq_energy'],
    }
    return alternatives.get(key, [])


# ============================================================================
# 归一化函数
# ============================================================================

def robust_z_score_normalize(
    features: Dict[str, float],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """使用 robust z-score 归一化特征。
    
    对应小论文的归一化策略，使用中位数和四分位差进行标准化：
    normalized_value = (x - median) / (IQR + epsilon)
    
    这种方法对异常值不敏感，适用于噪声数据。
    
    Parameters
    ----------
    features : dict
        原始特征字典。
    stats : dict, optional
        特征统计量字典，格式为:
        {'X1': {'median': 0.1, 'q25': 0.05, 'q75': 0.15}, ...}
        如果为 None，使用默认的归一化参数。
    epsilon : float
        防止除零的小常数。
        
    Returns
    -------
    dict
        归一化后的特征字典。
    """
    # 默认统计参数 (基于典型正常数据的经验值)
    default_stats = _get_default_normalization_stats()
    
    normalized = {}
    for key, value in features.items():
        if stats and key in stats:
            feat_stats = stats[key]
            median = feat_stats.get('median', 0.0)
            q25 = feat_stats.get('q25', 0.0)
            q75 = feat_stats.get('q75', 1.0)
        elif key in default_stats:
            feat_stats = default_stats[key]
            median = feat_stats['median']
            q25 = feat_stats['q25']
            q75 = feat_stats['q75']
        else:
            # 对于未知特征，使用简单的 min-max 式归一化
            median = 0.0
            q25 = 0.0
            q75 = 1.0
        
        iqr = q75 - q25
        if iqr < epsilon:
            iqr = epsilon
        
        normalized[key] = (float(value) - median) / (iqr + epsilon)
    
    return normalized


def quantile_normalize(
    features: Dict[str, float],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """使用分位数归一化，将特征映射到 [0, 1] 范围。
    
    normalized_value = (x - min) / (max - min)
    
    Parameters
    ----------
    features : dict
        原始特征字典。
    stats : dict, optional
        特征统计量字典，格式为:
        {'X1': {'min': 0.0, 'max': 1.0}, ...}
    epsilon : float
        防止除零的小常数。
        
    Returns
    -------
    dict
        归一化后的特征字典，值域 [0, 1]。
    """
    default_stats = _get_default_normalization_stats()
    
    normalized = {}
    for key, value in features.items():
        if stats and key in stats:
            feat_stats = stats[key]
            min_val = feat_stats.get('min', 0.0)
            max_val = feat_stats.get('max', 1.0)
        elif key in default_stats:
            feat_stats = default_stats[key]
            min_val = feat_stats.get('min', feat_stats['q25'])
            max_val = feat_stats.get('max', feat_stats['q75'])
        else:
            min_val = 0.0
            max_val = 1.0
        
        range_val = max_val - min_val
        if range_val < epsilon:
            range_val = epsilon
        
        norm_val = (float(value) - min_val) / (range_val + epsilon)
        normalized[key] = max(0.0, min(1.0, norm_val))  # Clip to [0, 1]
    
    return normalized


def _get_default_normalization_stats() -> Dict[str, Dict[str, float]]:
    """获取默认的归一化统计参数。
    
    这些值基于典型正常频响数据的经验估计。
    """
    return {
        # 系统级基础特征 (X1-X5)
        'X1': {'median': 0.0, 'q25': -0.1, 'q75': 0.1, 'min': -0.5, 'max': 0.5},
        'X2': {'median': 0.005, 'q25': 0.002, 'q75': 0.01, 'min': 0.0, 'max': 0.05},
        'X3': {'median': 0.0, 'q25': -1e-10, 'q75': 1e-10, 'min': -1e-9, 'max': 1e-9},
        'X4': {'median': 0.05, 'q25': 0.02, 'q75': 0.1, 'min': 0.0, 'max': 0.5},
        'X5': {'median': 0.1, 'q25': 0.05, 'q75': 0.2, 'min': 0.0, 'max': 0.35},
        
        # 模块级症状特征 (X6-X10)
        'X6': {'median': 0.005, 'q25': 0.002, 'q75': 0.01, 'min': 0.0, 'max': 0.03},
        'X7': {'median': 0.2, 'q25': 0.1, 'q75': 0.5, 'min': 0.0, 'max': 2.0},
        'X8': {'median': 0.1, 'q25': 0.05, 'q75': 0.2, 'min': 0.0, 'max': 1.0},
        'X9': {'median': 1e4, 'q25': 5e3, 'q75': 2e4, 'min': 0.0, 'max': 1e5},
        'X10': {'median': 0.1, 'q25': 0.05, 'q75': 0.2, 'min': 0.0, 'max': 0.5},
        
        # 包络/残差特征 (X11-X15)
        'X11': {'median': 0.05, 'q25': 0.02, 'q75': 0.1, 'min': 0.0, 'max': 0.3},
        'X12': {'median': 1.0, 'q25': 0.5, 'q75': 2.0, 'min': 0.0, 'max': 5.0},
        'X13': {'median': 1.0, 'q25': 0.5, 'q75': 3.0, 'min': 0.0, 'max': 10.0},
        'X14': {'median': 0.1, 'q25': 0.05, 'q75': 0.3, 'min': 0.0, 'max': 1.0},
        'X15': {'median': 0.1, 'q25': 0.05, 'q75': 0.2, 'min': 0.0, 'max': 0.5},
        
        # 频率对齐特征 (X16-X18)
        'X16': {'median': 0.0, 'q25': -0.01, 'q75': 0.01, 'min': -0.1, 'max': 0.1},
        'X17': {'median': 0.0, 'q25': -0.01, 'q75': 0.01, 'min': -0.05, 'max': 0.05},
        'X18': {'median': 0.0, 'q25': -0.01, 'q75': 0.01, 'min': -0.05, 'max': 0.05},
        
        # 幅度细粒度特征 (X19-X22)
        'X19': {'median': 0.0, 'q25': -1e-11, 'q75': 1e-11, 'min': -1e-10, 'max': 1e-10},
        'X20': {'median': 1.0, 'q25': 0.5, 'q75': 2.0, 'min': 0.0, 'max': 5.0},
        'X21': {'median': 5, 'q25': 2, 'q75': 10, 'min': 0, 'max': 20},
        'X22': {'median': 0.3, 'q25': 0.15, 'q75': 0.5, 'min': 0.0, 'max': 0.8},
    }


def compute_normalization_stats(feature_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """从训练数据计算归一化统计量。
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        特征矩阵，shape = (n_samples, n_features)
    feature_names : list
        特征名称列表，与矩阵列对应。
        
    Returns
    -------
    dict
        每个特征的统计量字典。
    """
    stats = {}
    for i, name in enumerate(feature_names):
        values = feature_matrix[:, i]
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) > 0:
            stats[name] = {
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'median': float(np.median(valid_values)),
                'q25': float(np.percentile(valid_values, 25)),
                'q75': float(np.percentile(valid_values, 75)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'p95': float(np.percentile(valid_values, 95)),  # 用于阈值设定
            }
        else:
            stats[name] = {
                'mean': 0.0, 'std': 1.0, 'median': 0.0,
                'q25': 0.0, 'q75': 1.0, 'min': 0.0, 'max': 1.0, 'p95': 1.0
            }
    
    return stats


# ============================================================================
# 正常状态识别辅助函数
# ============================================================================

def compute_overall_anomaly_score(features: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """计算整体异常度分数 (overall_score)。
    
    用于正常状态识别，通过加权聚合多个特征的异常程度。
    
    Parameters
    ----------
    features : dict
        归一化后的特征字典。
    weights : dict, optional
        特征权重字典。如果为 None，使用默认权重。
        
    Returns
    -------
    float
        整体异常度分数，范围 [0, 1]，越高表示越异常。
    """
    # 默认权重 - 基于特征对异常检测的重要性
    default_weights = {
        'X1': 0.15, 'X2': 0.12, 'X3': 0.08, 'X4': 0.10, 'X5': 0.10,
        'X6': 0.05, 'X7': 0.08, 'X8': 0.05, 'X9': 0.03, 'X10': 0.04,
        'X11': 0.08, 'X12': 0.05, 'X13': 0.05, 'X14': 0.02, 'X15': 0.02,
        'X16': 0.03, 'X17': 0.02, 'X18': 0.02,
        'X19': 0.02, 'X20': 0.02, 'X21': 0.01, 'X22': 0.01,
    }
    
    weights = weights or default_weights
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, value in features.items():
        if key in weights:
            # 将归一化值转换为异常度 (取绝对值)
            anomaly_score = min(1.0, abs(float(value)))
            weighted_sum += weights[key] * anomaly_score
            total_weight += weights[key]
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def is_normal_state(
    overall_score: float,
    max_prob: float,
    overall_threshold: float = 0.15,
    max_prob_threshold: float = 0.3
) -> bool:
    """判断是否为正常状态。
    
    对应小论文的正常状态识别策略：
    1. 整体异常度低于阈值
    2. 最大概率低于阈值
    
    Parameters
    ----------
    overall_score : float
        整体异常度分数。
    max_prob : float
        推理结果的最大概率。
    overall_threshold : float
        整体异常度阈值，低于此值认为正常。
    max_prob_threshold : float
        最大概率阈值，低于此值认为正常。
        
    Returns
    -------
    bool
        True 表示正常状态。
    """
    return overall_score < overall_threshold or max_prob < max_prob_threshold
