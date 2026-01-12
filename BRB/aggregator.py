#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统级推理聚合器 (System-Level Aggregator)
==========================================
对应小论文系统级诊断的聚合逻辑。

本模块负责：
1. 聚合三个子BRB（幅度、频率、参考电平）的推理结果
2. 应用softmax进行概率校准
3. 执行正常状态识别
4. 输出最终的系统级诊断结果
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .system_brb_amp import amp_brb_infer
from .system_brb_freq import freq_brb_infer
from .system_brb_ref import ref_brb_infer


# 子BRB权重常量
# 频率子BRB的激活度对所有类别都偏高，需要降低其影响力
# 这些权重用于在聚合时平衡各子BRB的贡献
SUB_BRB_WEIGHT_AMP = 1.2   # 幅度子BRB权重
SUB_BRB_WEIGHT_FREQ = 0.7  # 频率子BRB权重（降低以避免过度分类为Freq）
SUB_BRB_WEIGHT_REF = 1.0   # 参考电平子BRB权重


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
    """计算整体异常度分数（使用z-score归一化）。
    
    用于正常状态识别，综合评估所有特征的异常程度。
    使用正常样本的统计数据进行z-score归一化，而不是硬编码的min/max区间。
    
    Parameters
    ----------
    features : dict
        输入特征字典。
        
    Returns
    -------
    float
        整体异常度分数 [0, 1]。
    """
    # 特征权重 - 重点使用envelope相关特征
    weights = {
        'X1': 0.10, 'X2': 0.08, 'X3': 0.05, 'X4': 0.08, 'X5': 0.05,
        'X6': 0.03, 'X7': 0.05, 'X8': 0.03, 'X9': 0.02, 'X10': 0.03,
        'X11': 0.18, 'X12': 0.15, 'X13': 0.10, 'X14': 0.02, 'X15': 0.02,
        'X16': 0.02, 'X17': 0.01, 'X18': 0.01,
        'X19': 0.01, 'X20': 0.01, 'X21': 0.01, 'X22': 0.01,
    }
    
    # 正常特征统计 (median, iqr) - 来自实际正常样本
    normal_stats = {
        'X1': (0.003152, 0.067323),
        'X2': (0.000094, 0.000115),
        'X3': (-0.000018, 0.000228),
        'X4': (0.010546, 0.005289),
        'X5': (0.255391, 0.101644),
        'X6': (0.000135, 0.000121),
        'X7': (0.040000, 0.025000),
        'X8': (0.008982, 0.004616),
        'X9': (0.000111, 0.000107),
        'X10': (0.005749, 0.003149),
        'X11': (0.003659, 0.001220),
        'X12': (0.030000, 0.010000),
        'X13': (6.394475, 23.550748),
        'X14': (0.023388, 0.039121),
        'X15': (0.008679, 0.008099),
        'X16': (0.000000, 0.001000),
        'X17': (0.000000, 0.001000),
        'X18': (0.000000, 0.001000),
        'X19': (-0.000036, 0.000172),
        'X20': (0.207247, 1.282695),
        'X21': (23.000000, 12.000000),
        'X22': (0.249534, 0.222095),
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        if key in features:
            raw_value = float(features[key])
            median, iqr = normal_stats.get(key, (0.0, 0.1))
            
            # 使用z-score归一化，|z|越大表示越异常
            if iqr < 1e-6:
                iqr = 1e-6
            z = (raw_value - median) / (iqr + 1e-6)
            
            # 将|z|映射到[0,1]的异常分数，|z|=3 -> score=1.0
            norm_value = min(1.0, abs(z) / 3.0)
            
            weighted_sum += weight * norm_value
            total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def aggregate_system_results(
    features: Dict[str, float],
    alpha: float = 3.0,  # Higher temperature for sharper distribution
    overall_threshold: float = 0.35,  # Higher threshold to capture more normal samples
    max_prob_threshold: float = 0.25  # Lower threshold for better fault confirmation
) -> Dict:
    """聚合三个子BRB的推理结果，输出系统级诊断。
    
    对应小论文系统级推理的聚合逻辑。
    
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
    # 执行三个子BRB推理
    amp_result = amp_brb_infer(features, alpha)
    freq_result = freq_brb_infer(features, alpha)
    ref_result = ref_brb_infer(features, alpha)
    
    # 获取各子BRB的激活度，并应用相对权重
    activations = [
        amp_result['activation'] * SUB_BRB_WEIGHT_AMP,
        freq_result['activation'] * SUB_BRB_WEIGHT_FREQ,
        ref_result['activation'] * SUB_BRB_WEIGHT_REF
    ]
    
    # 计算整体异常度
    overall_score = compute_overall_score(features)
    
    # 计算故障概率分布
    fault_probs = softmax_with_temperature(activations, alpha)
    
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
        'sub_brb_results': {
            'amp': amp_result,
            'freq': freq_result,
            'ref': ref_result,
        },
    }


def system_level_infer_with_sub_brbs(
    features: Dict[str, float],
    alpha: float = 3.0,  # Higher temperature for sharper distribution
    overall_threshold: float = 0.35,  # Higher threshold to capture more normal samples
    max_prob_threshold: float = 0.25,  # Lower threshold for better fault confirmation
    use_feature_routing: bool = True
) -> Dict:
    """使用子BRB架构的系统级推理入口。
    
    这是优化后的系统级推理接口，支持：
    1. 特征分流到对应的子BRB
    2. 聚合子BRB结果
    3. 正常状态识别
    
    Parameters
    ----------
    features : dict
        输入特征字典。
    alpha : float
        Softmax温度参数。
    overall_threshold : float
        整体异常度阈值。
    max_prob_threshold : float
        最大概率阈值。
    use_feature_routing : bool
        是否启用特征分流（默认True）。
        
    Returns
    -------
    dict
        系统级诊断结果。
    """
    if use_feature_routing:
        # 导入特征路由
        try:
            from features.feature_router import feature_router
            
            # 分流特征到各子BRB
            amp_features = feature_router(features, 'amp')
            freq_features = feature_router(features, 'freq')
            ref_features = feature_router(features, 'ref')
            
            # 执行各子BRB推理
            amp_result = amp_brb_infer(amp_features, alpha)
            freq_result = freq_brb_infer(freq_features, alpha)
            ref_result = ref_brb_infer(ref_features, alpha)
            
            # 聚合结果 - 应用子BRB权重
            activations = [
                amp_result['activation'] * SUB_BRB_WEIGHT_AMP,
                freq_result['activation'] * SUB_BRB_WEIGHT_FREQ,
                ref_result['activation'] * SUB_BRB_WEIGHT_REF
            ]
            
            overall_score = compute_overall_score(features)
            fault_probs = softmax_with_temperature(activations, alpha)
            
            # 后续处理与 aggregate_system_results 相同
            normal_weight = 0.0
            if overall_score < overall_threshold:
                normal_weight = 1.0 - overall_score / (overall_threshold + 1e-12)
            
            max_fault_prob = max(fault_probs)
            if max_fault_prob < max_prob_threshold:
                normal_weight = max(normal_weight, max_prob_threshold - max_fault_prob)
            
            fault_scale = max(0.0, 1.0 - normal_weight)
            scaled_faults = [p * fault_scale for p in fault_probs]
            
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
            
            max_prob = max(probabilities.values())
            predicted_class = max(probabilities, key=probabilities.get)
            is_normal = probabilities.get('正常', 0.0) >= 0.5 or max_fault_prob < max_prob_threshold
            
            return {
                'probabilities': probabilities,
                'max_prob': max_prob,
                'predicted_class': predicted_class,
                'is_normal': is_normal,
                'uncertainty': 1.0 - max_prob,
                'overall_score': overall_score,
                'sub_brb_results': {
                    'amp': amp_result,
                    'freq': freq_result,
                    'ref': ref_result,
                },
            }
            
        except ImportError:
            # 如果无法导入特征路由，回退到标准聚合
            pass
    
    # 不使用特征分流，直接使用完整特征
    return aggregate_system_results(features, alpha, overall_threshold, max_prob_threshold)
