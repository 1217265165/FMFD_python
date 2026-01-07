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
    max_prob_threshold: float = 0.3
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
    
    # 获取各子BRB的激活度
    activations = [
        amp_result['activation'],
        freq_result['activation'],
        ref_result['activation']
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
    alpha: float = 2.0,
    overall_threshold: float = 0.15,
    max_prob_threshold: float = 0.3,
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
            
            # 聚合结果
            activations = [
                amp_result['activation'],
                freq_result['activation'],
                ref_result['activation']
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
