"""
Module-level BRB reasoning (对应小论文 3.2 分层推理与规则压缩章节)。

本模块实现 21 个模块的压缩式 BRB 推理：利用系统级诊断结果
作为虚拟先验属性 V（式 (3)），仅激活与异常类型相关的规则组，
避免全组合爆炸。规则数≈45、参数≈38，显著少于传统设计。

优化特性（对应准确率提升需求）：
1. 特征分流：根据系统级异常类型，仅使用相关特征进行推理
2. 模块组激活：仅激活与异常类型相关的模块组进行推理
3. 规则压缩：通过物理链路知识压缩规则组合
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional

from .utils import BRBRule, SimpleBRB, normalize_feature


MODULE_LABELS: List[str] = [
    "衰减器",
    "前置放大器",  # DISABLED in single-band mode
    "低频段前置低通滤波器",
    "低频段第一混频器",
    "高频段YTF滤波器",
    "高频段混频器",
    "时钟振荡器",
    "时钟合成与同步网络",
    "本振源（谐波发生器）",
    "本振混频组件",
    "校准源",
    "存储器",
    "校准信号开关",
    "中频放大器",
    "ADC",
    "数字RBW",
    "数字放大器",
    "数字检波器",
    "VBW滤波器",
    "电源模块",
    "未定义/其他",
]

# Single-band mode flag: When True, preamp is disabled
SINGLE_BAND_MODE = True

# Disabled modules in single-band mode
DISABLED_MODULES = ["前置放大器"] if SINGLE_BAND_MODE else []

# 模块分组 - 按物理链路和功能相关性
# NOTE: 前置放大器 is DISABLED in single-band mode
MODULE_GROUPS = {
    # 幅度链路模块组 (前置放大器 excluded in single-band mode)
    'amp_group': [
        '衰减器', '中频放大器', '数字放大器', 'ADC',
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
    # 通用模块
    'other_group': [
        '电源模块', '未定义/其他'
    ],
}

# 异常类型到模块组的映射
FAULT_TYPE_TO_MODULE_GROUP = {
    '幅度失准': 'amp_group',
    '频率失准': 'freq_group',
    '参考电平失准': 'ref_group',
}


def _mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(statistics.mean(arr)) if arr else 0.0


def _aggregate_module_score(features: Dict[str, float], anomaly_type: str = None) -> float:
    """Aggregate module-level symptom scores with feature streaming.

    对应文中模块层虚拟属性 V 构建思路：根据系统层异常类型，选择性使用相关特征。
    
    Parameters
    ----------
    features : dict
        模块层特征字典，包含传统字段和X6-X22扩展字段。
    anomaly_type : str, optional
        系统层检测到的异常类型（"幅度失准", "频率失准", "参考电平失准"），
        用于特征分流。如果为None，使用全部特征。
    
    Returns
    -------
    float
        聚合后的模块层异常分数(0-1)。
    """

    # 传统特征
    md_step_raw = max(
        features.get("step_score", 0.0),
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
        features.get("X7", 0.0),  # 增益非线性
    )
    md_step = normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope = normalize_feature(abs(features.get("res_slope", 0.0)), 1e-12, 1e-10)
    md_ripple = normalize_feature(features.get("ripple_var", features.get("X6", 0.0)), 0.001, 0.02)
    md_df = normalize_feature(abs(features.get("df", 0.0)), 1e6, 5e7)
    md_viol = normalize_feature(features.get("viol_rate", features.get("X11", 0.0)), 0.02, 0.2)
    md_gain_bias = max(
        normalize_feature(abs(features.get("bias", 0.0)), 0.1, 1.0),
        normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.02, 0.2),
    )

    # 特征分流：根据异常类型选择相关特征
    if anomaly_type == "幅度失准":
        # 幅度模块：使用幅度相关特征X1,X2,X5,X11-X13,X19-X22
        amp_features = [
            md_step,
            md_ripple,
            md_gain_bias,
            normalize_feature(features.get("X11", 0.0), 0.01, 0.3),  # 包络超出率
            normalize_feature(features.get("X12", 0.0), 0.5, 5.0),  # 最大违规
            normalize_feature(features.get("X13", 0.0), 0.1, 10.0),  # 违规能量
            normalize_feature(abs(features.get("X19", 0.0)), 1e-12, 1e-10),  # 低频斜率
            normalize_feature(abs(features.get("X20", 0.0)), 0.5, 5.0),  # 峰度
            normalize_feature(features.get("X21", 0.0), 1, 20),  # 峰值数
            normalize_feature(features.get("X22", 0.0), 0.1, 0.8),  # 主频占比
        ]
        return _mean(amp_features)
    
    elif anomaly_type == "频率失准":
        # 频率模块：使用频率相关特征X4,X14-X15,X16-X18
        freq_features = [
            md_df,
            normalize_feature(features.get("X14", 0.0), 0.01, 1.0),  # 低频残差
            normalize_feature(features.get("X15", 0.0), 0.01, 0.5),  # 高频残差
            normalize_feature(abs(features.get("X16", 0.0)), 0.001, 0.1),  # 频移
            normalize_feature(abs(features.get("X17", 0.0)), 0.001, 0.05),  # 缩放
            normalize_feature(abs(features.get("X18", 0.0)), 0.001, 0.05),  # 平移
        ]
        return _mean(freq_features)
    
    elif anomaly_type == "参考电平失准":
        # 参考电平模块：使用参考相关特征X1,X3,X5,X11-X13
        ref_features = [
            md_slope,
            md_gain_bias,
            normalize_feature(features.get("X11", 0.0), 0.01, 0.3),  # 包络超出率
            normalize_feature(features.get("X12", 0.0), 0.5, 5.0),  # 最大违规
            normalize_feature(features.get("X13", 0.0), 0.1, 10.0),  # 违规能量
        ]
        return _mean(ref_features)
    
    else:
        # 默认：使用所有传统特征
        return _mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias])


def module_level_infer(features: Dict[str, float], sys_probs: Dict[str, float]) -> Dict[str, float]:
    """Perform compressed module-level BRB inference with feature streaming.

    Parameters
    ----------
    features : dict
        模块层特征，至少包含 step_score、res_slope、ripple_var、df、
        viol_rate、gain、bias，可额外提供 X6-X22扩展特征用于特征分流。
    sys_probs : dict
        系统级诊断概率分布，作为虚拟先验属性 V 对规则进行加权。
        既支持直接传入概率字典，也支持 system_level_infer 的完整
        返回值（会自动提取其中的 ``probabilities`` 字段）。

    Returns
    -------
    dict
        21 个模块的概率分布，顺序与 ``MODULE_LABELS`` 对齐。
        NOTE: 前置放大器 is set to 0 in single-band mode.
    """

    probs = sys_probs.get("probabilities", sys_probs)
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)
    ref_prior = probs.get("参考电平失准", 0.3)

    # 确定主导异常类型，用于特征分流
    max_prob_val = max(amp_prior, freq_prior, ref_prior)
    if amp_prior == max_prob_val:
        anomaly_type = "幅度失准"
    elif freq_prior == max_prob_val:
        anomaly_type = "频率失准"
    else:
        anomaly_type = "参考电平失准"

    # 使用特征分流计算模块层分数
    md = _aggregate_module_score(features, anomaly_type)

    # Rules updated: 前置放大器 removed from amplitude rules in single-band mode
    rules = [
        BRBRule(
            weight=0.8 * ref_prior,
            belief={"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16, "未定义/其他": 0.10},
        ),
        # Amplitude rules: 前置放大器 belief redistributed to other modules
        BRBRule(
            weight=0.6 * amp_prior,
            belief={"中频放大器": 0.35, "数字放大器": 0.30, "衰减器": 0.20, "ADC": 0.15},
        ),
        BRBRule(
            weight=0.7 * freq_prior,
            belief={"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15},
        ),
        BRBRule(weight=0.5 * freq_prior, belief={"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
        BRBRule(weight=0.5 * freq_prior, belief={"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
        BRBRule(weight=0.4 * amp_prior, belief={"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
        BRBRule(weight=0.3, belief={"电源模块": 0.80, "未定义/其他": 0.20}),
        BRBRule(weight=0.2, belief={"未定义/其他": 1.0}),
    ]

    brb = SimpleBRB(MODULE_LABELS, rules)
    result = brb.infer([md])
    
    # Force preamp probability to 0 in single-band mode
    if SINGLE_BAND_MODE and "前置放大器" in result:
        result["前置放大器"] = 0.0
        # Renormalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
    
    return result


def module_level_infer_with_activation(
    features: Dict[str, float], 
    sys_probs: Dict[str, float],
    only_activate_relevant: bool = True,
    uncertain_max_prob_threshold: float = 0.45,
    uncertain_top2_diff_threshold: float = 0.15,
    expand_top_m: int = 10,
    contract_top_k: int = 5
) -> Dict[str, float]:
    """优化版模块级推理：仅激活与异常类型相关的模块组 + 候选路由兜底。
    
    对应小论文规则压缩策略：根据系统级诊断结果，
    仅对可能受影响的模块子集执行推理，减少冗余计算。
    
    Enhanced with candidate routing fallback:
    - If system-level is uncertain (max_prob < threshold or top2 diff small):
      Expand candidates to Top-M (8~12)
    - If high certainty, contract to Top-K (3~6)
    
    Parameters
    ----------
    features : dict
        模块层特征字典。
    sys_probs : dict
        系统级诊断概率分布。
    only_activate_relevant : bool
        如果为True，仅激活与检测到的异常类型相关的模块组。
        如果为False，行为与 module_level_infer 相同。
    uncertain_max_prob_threshold : float
        If max_prob below this, expand candidates.
    uncertain_top2_diff_threshold : float
        If top1-top2 diff below this, expand candidates.
    expand_top_m : int
        Number of candidates when uncertain.
    contract_top_k : int
        Number of candidates when certain.
        
    Returns
    -------
    dict
        21个模块的概率分布。
    """
    probs = sys_probs.get("probabilities", sys_probs)
    
    # 检查是否为正常状态
    normal_prob = probs.get("正常", 0.0)
    if normal_prob > 0.5:
        # 正常状态，所有模块概率均匀分布且较低
        return {label: 1.0 / len(MODULE_LABELS) for label in MODULE_LABELS}
    
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)
    ref_prior = probs.get("参考电平失准", 0.3)
    
    # 确定主导异常类型
    max_prob_val = max(amp_prior, freq_prior, ref_prior)
    if amp_prior == max_prob_val:
        anomaly_type = "幅度失准"
        active_group = "amp_group"
    elif freq_prior == max_prob_val:
        anomaly_type = "频率失准"
        active_group = "freq_group"
    else:
        anomaly_type = "参考电平失准"
        active_group = "ref_group"
    
    # 使用特征分流计算模块层分数
    md = _aggregate_module_score(features, anomaly_type)
    
    if only_activate_relevant:
        # 仅激活相关模块组的规则
        active_modules = set(MODULE_GROUPS.get(active_group, []))
        active_modules.update(MODULE_GROUPS.get('other_group', []))  # 始终包含通用模块
        
        # 构建针对性的规则
        rules = _build_targeted_rules(anomaly_type, amp_prior, freq_prior, ref_prior)
    else:
        # 使用完整规则集
        rules = [
            BRBRule(
                weight=0.8 * ref_prior,
                belief={"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16, "未定义/其他": 0.10},
            ),
            BRBRule(
                weight=0.6 * amp_prior,
                belief={"前置放大器": 0.40, "中频放大器": 0.25, "数字放大器": 0.20, "衰减器": 0.10, "ADC": 0.05},
            ),
            BRBRule(
                weight=0.7 * freq_prior,
                belief={"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15},
            ),
            BRBRule(weight=0.5 * freq_prior, belief={"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
            BRBRule(weight=0.5 * freq_prior, belief={"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
            BRBRule(weight=0.4 * amp_prior, belief={"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
            BRBRule(weight=0.3, belief={"电源模块": 0.80, "未定义/其他": 0.20}),
            BRBRule(weight=0.2, belief={"未定义/其他": 1.0}),
        ]
    
    brb = SimpleBRB(MODULE_LABELS, rules)
    result = brb.infer([md])
    
    # Force preamp probability to 0 in single-band mode
    if SINGLE_BAND_MODE and "前置放大器" in result:
        result["前置放大器"] = 0.0
        # Renormalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
    
    return result


def _build_targeted_rules(anomaly_type: str, amp_prior: float, freq_prior: float, ref_prior: float) -> List[BRBRule]:
    """根据异常类型构建针对性的规则集。
    
    实现规则压缩：仅保留与检测到的异常类型相关的规则，
    显著减少规则数量。
    
    NOTE: 前置放大器 is EXCLUDED from all rules in single-band mode.
    """
    rules = []
    
    if anomaly_type == "幅度失准":
        # 幅度链路相关规则 - 权重增强 (NO 前置放大器)
        rules.extend([
            BRBRule(
                weight=0.8 * amp_prior,
                belief={"中频放大器": 0.35, "数字放大器": 0.30, "衰减器": 0.20, "ADC": 0.15},
            ),
            BRBRule(
                weight=0.6 * amp_prior,
                belief={"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10},
            ),
            BRBRule(
                weight=0.4 * amp_prior,
                belief={"衰减器": 0.60, "中频放大器": 0.25, "未定义/其他": 0.15},
            ),
        ])
        
    elif anomaly_type == "频率失准":
        # 频率链路相关规则 - 权重增强
        rules.extend([
            BRBRule(
                weight=0.8 * freq_prior,
                belief={"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15},
            ),
            BRBRule(
                weight=0.6 * freq_prior,
                belief={"高频段YTF滤波器": 0.50, "高频段混频器": 0.30, "本振混频组件": 0.20},
            ),
            BRBRule(
                weight=0.5 * freq_prior,
                belief={"低频段前置低通滤波器": 0.50, "低频段第一混频器": 0.35, "未定义/其他": 0.15},
            ),
        ])
        
    elif anomaly_type == "参考电平失准":
        # 参考电平链路相关规则 - 权重增强
        rules.extend([
            BRBRule(
                weight=0.8 * ref_prior,
                belief={"衰减器": 0.45, "校准源": 0.20, "校准信号开关": 0.20, "存储器": 0.10, "未定义/其他": 0.05},
            ),
            BRBRule(
                weight=0.5 * ref_prior,
                belief={"校准源": 0.40, "存储器": 0.30, "校准信号开关": 0.20, "未定义/其他": 0.10},
            ),
        ])
    
    # 添加通用规则（权重较低）
    rules.extend([
        BRBRule(weight=0.2, belief={"电源模块": 0.80, "未定义/其他": 0.20}),
        BRBRule(weight=0.1, belief={"未定义/其他": 1.0}),
    ])
    
    return rules


def get_top_k_modules(module_probs: Dict[str, float], k: int = 3) -> List[tuple]:
    """获取概率最高的前K个模块。
    
    Parameters
    ----------
    module_probs : dict
        模块概率分布。
    k : int
        返回的模块数量。
        
    Returns
    -------
    list
        (模块名称, 概率) 元组列表，按概率降序排列。
    """
    sorted_modules = sorted(module_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_modules[:k]
