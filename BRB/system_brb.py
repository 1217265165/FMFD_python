"""
System-level BRB inference module.

This file implements the system-level belief rule base (BRB) reasoning
for the spectrum analyzer fault diagnosis described in the short paper
“基于知识驱动规则优化与分层推理的频谱分析仪故障诊断方法”. 对应小论文系统级诊断
章节与式 (1)-(2)。

Core design principles:
- X1~X5 对应系统级特征：整体幅度偏移、带内平坦度、高频段衰减斜率、频率标度非线性、幅度缩放一致性。
- 属性匹配度使用三角隶属度 (High/Normal/Low)；属性权重与规则权重均显式保留。
- ER 合成前加入正常状态检测：利用整体异常度 overall_score 与 softmax 温度 α 控制置信度。
- 输出包括概率分布、最大概率、是否判定为正常以及不确定度 (1-max_prob)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass
class SystemBRBConfig:
    """Configuration of the system-level BRB.

    Attributes
    ----------
    alpha : float
        Softmax temperature used in ER fusion (对应式 (2) 的 α)。
    overall_threshold : float
        Overall anomaly score threshold; below this value samples are
        considered normal (ACCURACY_IMPROVEMENT.md 中的正常识别策略)。
    max_prob_threshold : float
        Maximum belief threshold; if the highest class probability is
        below the value, output will fall back to "正常" with explicit
        uncertainty.
    attribute_weights : Tuple[float, ...]
        Relative importance of features when computing the aggregated
        anomaly score. Extended to support X1-X22 (22 features).
        Grouped as: [X1-X5基础, X6-X10模块, X11-X15包络, X16-X18频率, X19-X22幅度]
    rule_weights : Tuple[float, float, float]
        Weights for amplitude/frequency/reference rule groups to mimic
        knowledge-driven rule compression.
    use_extended_features : bool
        If True, use X1-X22; if False, use only X1-X5 (backward compatibility).
    """

    alpha: float = 2.5  # 提高温度以增强区分度
    overall_threshold: float = 0.15  # 降低阈值使正常识别更严格
    max_prob_threshold: float = 0.28  # 降低阈值要求更高置信度
    attribute_weights: Tuple[float, ...] = (
        # X1-X5: 基础特征权重
        0.20, 0.18, 0.15, 0.14, 0.13,
        # X6-X10: 模块症状特征权重（较低，主要供模块层使用）
        0.05, 0.05, 0.04, 0.04, 0.04,
        # X11-X15: 包络/残差特征权重（系统层重要）
        0.12, 0.10, 0.08, 0.07, 0.07,
        # X16-X18: 频率特征权重（频率失准识别）
        0.10, 0.10, 0.10,
        # X19-X22: 幅度细粒度特征权重（幅度/参考识别）
        0.08, 0.07, 0.06, 0.05,
    )
    rule_weights: Tuple[float, float, float] = (1.2, 1.0, 1.1)  # amp/freq/ref权重调整
    use_extended_features: bool = True  # 默认启用扩展特征


def _triangular_membership(value: float, low: float, center: float, high: float) -> Tuple[float, float, float]:
    """Return (low, normal, high) membership using triangular functions."""
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


def _get_feature(features: Dict[str, float], *names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in features:
            return float(features[name])
    return default


def _normalize_feature_zscore(value: float, median: float, iqr: float, epsilon: float = 1e-6) -> float:
    """使用robust z-score归一化特征值。
    
    Step2要求：归一化只允许用 normal_feature_stats（robust z-score），
    禁止再改"手写区间"。
    
    归一化公式: z = (x - median) / (IQR + epsilon)
    然后clip到[-5, 5]范围。
    """
    if iqr < epsilon:
        iqr = epsilon
    z = (value - median) / (iqr + epsilon)
    return max(-5.0, min(5.0, z))


def _normalize_feature(value: float, lower: float, upper: float) -> float:
    """Legacy min-max normalization for backward compatibility."""
    value = max(lower, min(value, upper))
    return (value - lower) / (upper - lower + 1e-12)


# 正常特征统计（从run_baseline生成的normal_feature_stats.csv获取）
# 特征值现在是residual（相对baseline的偏差），正常样本应该围绕0分布
# 这些值从实际正常样本中计算得出，格式为 (median, iqr)
NORMAL_FEATURE_STATS = {
    # 格式: feature_name: (median, iqr)
    # residual-based特征：来自实际正常样本统计
    "X1": (0.003152, 0.067323),    # 整体幅度偏移（residual均值）
    "X2": (0.000094, 0.000115),    # 带内平坦度（residual方差）
    "X3": (-0.000018, 0.000228),   # 高频衰减斜率
    "X4": (0.010546, 0.005289),    # 频率标度非线性（std）
    "X5": (0.255391, 0.101644),    # 幅度缩放一致性
    "X6": (0.000135, 0.000121),    # 纹波方差
    "X7": (0.040000, 0.025000),    # 增益非线性
    "X8": (0.008982, 0.004616),    # 本振泄漏
    "X9": (0.000111, 0.000107),    # 调谐线性度残差
    "X10": (0.005749, 0.003149),   # 频段幅度一致性
    "X11": (0.003659, 0.001220),   # 包络越界率
    "X12": (0.030000, 0.010000),   # 最大包络违规
    "X13": (6.394475, 23.550748),  # 包络违规能量
    "X14": (0.023388, 0.039121),   # 低频段残差均值
    "X15": (0.008679, 0.008099),   # 高频段残差std
    "X16": (0.000000, 0.001000),   # 频移（互相关滞后），IQR=0时设为小正数
    "X17": (0.000000, 0.001000),   # 频率缩放
    "X18": (0.000000, 0.001000),   # 频率平移
    "X19": (-0.000036, 0.000172),  # 低频斜率
    "X20": (0.207247, 1.282695),   # 峰度
    "X21": (23.000000, 12.000000), # 峰值数
    "X22": (0.249534, 0.222095),   # 主频能量占比
}


def _compute_attribute_scores(features: Dict[str, float], feature_stats: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
    """Compute normalized attribute scores for X1~X22 using robust z-score.

    Step2要求：使用 normal_feature_stats 进行归一化，而不是硬编码的min/max区间。
    这样正常样本的z-score大多在[-2, 2]范围内。
    """
    
    stats = feature_stats or NORMAL_FEATURE_STATS

    # X1-X5: 基础系统级特征（现在是residual-based）
    x1_raw = _get_feature(features, "X1", "amplitude_offset", "bias")  # 不取abs，保留符号
    x2_raw = _get_feature(features, "X2", "inband_flatness", "ripple_var")
    x3_raw = _get_feature(features, "X3", "hf_attenuation_slope", "res_slope")
    x4_raw = _get_feature(features, "X4", "freq_scale_nonlinearity", "df")
    x5_raw = _get_feature(features, "X5", "scale_consistency", "amp_scale_consistency", "gain_consistency")

    # X6-X10: 模块级症状
    x6_raw = _get_feature(features, "X6", "ripple_variance")
    x7_raw = _get_feature(features, "X7", "gain_nonlinearity", "step_score")
    x8_raw = _get_feature(features, "X8", "lo_leakage")
    x9_raw = _get_feature(features, "X9", "tuning_linearity_residual")
    x10_raw = _get_feature(features, "X10", "band_amplitude_consistency")

    # X11-X15: 包络/残差特征
    x11_raw = _get_feature(features, "X11", "env_overrun_rate", "viol_rate")
    x12_raw = _get_feature(features, "X12", "env_overrun_max")
    x13_raw = _get_feature(features, "X13", "env_violation_energy")
    x14_raw = _get_feature(features, "X14", "band_residual_low")
    x15_raw = _get_feature(features, "X15", "band_residual_high_std")

    # X16-X18: 频率对齐/形变
    x16_raw = _get_feature(features, "X16", "corr_shift_bins")
    x17_raw = _get_feature(features, "X17", "warp_scale")
    x18_raw = _get_feature(features, "X18", "warp_bias")

    # X19-X22: 幅度链路细粒度
    x19_raw = _get_feature(features, "X19", "slope_low")
    x20_raw = _get_feature(features, "X20", "kurtosis_detrended")
    x21_raw = _get_feature(features, "X21", "peak_count_residual")
    x22_raw = _get_feature(features, "X22", "ripple_dom_freq_energy")

    # 使用z-score归一化，然后映射到[0,1]用于BRB
    # z-score绝对值越大表示越异常
    def zscore_to_score(z_value: float) -> float:
        """将z-score映射到[0,1]的异常分数，|z|越大分数越高"""
        return min(1.0, abs(z_value) / 3.0)  # |z|=3 对应 score=1.0

    scores = {
        "X1": zscore_to_score(_normalize_feature_zscore(x1_raw, *stats.get("X1", (0.0, 0.1)))),
        "X2": zscore_to_score(_normalize_feature_zscore(x2_raw, *stats.get("X2", (0.01, 0.02)))),
        "X3": zscore_to_score(_normalize_feature_zscore(x3_raw, *stats.get("X3", (0.0, 0.001)))),
        "X4": zscore_to_score(_normalize_feature_zscore(x4_raw, *stats.get("X4", (0.05, 0.03)))),
        "X5": zscore_to_score(_normalize_feature_zscore(x5_raw, *stats.get("X5", (0.3, 0.1)))),
        "X6": zscore_to_score(_normalize_feature_zscore(x6_raw, *stats.get("X6", (0.01, 0.02)))),
        "X7": zscore_to_score(_normalize_feature_zscore(x7_raw, *stats.get("X7", (0.05, 0.03)))),
        "X8": zscore_to_score(_normalize_feature_zscore(x8_raw, *stats.get("X8", (0.02, 0.02)))),
        "X9": zscore_to_score(_normalize_feature_zscore(x9_raw, *stats.get("X9", (0.01, 0.01)))),
        "X10": zscore_to_score(_normalize_feature_zscore(x10_raw, *stats.get("X10", (0.02, 0.02)))),
        "X11": zscore_to_score(_normalize_feature_zscore(x11_raw, *stats.get("X11", (0.0, 0.01)))),
        "X12": zscore_to_score(_normalize_feature_zscore(x12_raw, *stats.get("X12", (0.0, 0.2)))),
        "X13": zscore_to_score(_normalize_feature_zscore(x13_raw, *stats.get("X13", (0.0, 1.0)))),
        "X14": zscore_to_score(_normalize_feature_zscore(x14_raw, *stats.get("X14", (0.0, 0.05)))),
        "X15": zscore_to_score(_normalize_feature_zscore(x15_raw, *stats.get("X15", (0.05, 0.03)))),
        "X16": zscore_to_score(_normalize_feature_zscore(x16_raw, *stats.get("X16", (0.0, 0.001)))),
        "X17": zscore_to_score(_normalize_feature_zscore(x17_raw, *stats.get("X17", (0.0, 0.001)))),
        "X18": zscore_to_score(_normalize_feature_zscore(x18_raw, *stats.get("X18", (0.0, 0.001)))),
        "X19": zscore_to_score(_normalize_feature_zscore(x19_raw, *stats.get("X19", (0.0, 0.0005)))),
        "X20": zscore_to_score(_normalize_feature_zscore(x20_raw, *stats.get("X20", (0.0, 0.5)))),
        "X21": zscore_to_score(_normalize_feature_zscore(x21_raw, *stats.get("X21", (2.0, 3.0)))),
        "X22": zscore_to_score(_normalize_feature_zscore(x22_raw, *stats.get("X22", (0.02, 0.02)))),
    }
    return scores


def _attribute_match_degrees(scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    return {name: _triangular_membership(value, 0.15, 0.35, 0.7) for name, value in scores.items()}


def _aggregate_score(scores: Dict[str, float], weights: Tuple[float, ...], use_extended: bool = True) -> float:
    """Aggregate scores with weights, supporting both X1-X5 and X1-X22."""
    if use_extended and len(weights) >= 22:
        # 使用全部22个特征
        keys = [f"X{i}" for i in range(1, 23)]
        valid_keys = [k for k in keys if k in scores]
        weighted = [scores[k] * weights[i] for i, k in enumerate(valid_keys)]
        return sum(weighted) / (sum(weights[:len(valid_keys)]) + 1e-12)
    else:
        # 向后兼容：仅使用X1-X5
        keys = ["X1", "X2", "X3", "X4", "X5"]
        weighted = [scores.get(key, 0.0) * w for key, w in zip(keys, weights[:5])]
        return sum(weighted) / (sum(weights[:5]) + 1e-12)


def _system_level_infer_er(features: Dict[str, float], cfg: SystemBRBConfig) -> Dict[str, float]:
    """核心 ER 版系统级推理实现，支持扩展特征。"""
    scores = _compute_attribute_scores(features)
    match = _attribute_match_degrees(scores)

    # Rule compression with extended features support
    if cfg.use_extended_features:
        # 幅度失准：使用X1,X2,X5(基础)+X11,X12,X13(包络)+X19,X20,X21,X22(幅度细粒度)
        amp_activation = cfg.rule_weights[0] * max(
            match["X1"][2], match["X2"][2], match["X5"][2],
            match["X11"][2], match["X12"][2], match["X13"][2],
            match["X19"][2], match["X20"][2], match["X21"][2], match["X22"][2]
        )
        
        # 频率失准：使用X4(基础)+X14,X15(残差)+X16,X17,X18(频率对齐)
        freq_activation = cfg.rule_weights[1] * max(
            match["X4"][2],
            match["X14"][2], match["X15"][2],
            match["X16"][2], match["X17"][2], match["X18"][2]
        )
        
        # 参考电平失准：使用X1,X3,X5(基础)+X11,X12,X13(包络)
        ref_activation = cfg.rule_weights[2] * max(
            match["X1"][2], match["X3"][2], match["X5"][2],
            match["X11"][2], match["X12"][2], match["X13"][0]  # 低包络违规
        )
    else:
        # 向后兼容：仅使用X1-X5
        amp_activation = cfg.rule_weights[0] * max(match["X1"][2], match["X2"][2], match["X5"][2])
        freq_activation = cfg.rule_weights[1] * match["X4"][2]
        ref_activation = cfg.rule_weights[2] * max(match["X2"][0], match["X3"][2], match["X5"][0])

    overall_score = _aggregate_score(scores, cfg.attribute_weights, cfg.use_extended_features)
    activations = [amp_activation, freq_activation, ref_activation]

    es = [math.exp(cfg.alpha * a) for a in activations]
    s = sum(es) + 1e-12
    fault_probs = {"幅度失准": es[0] / s, "频率失准": es[1] / s, "参考电平失准": es[2] / s}

    # 正常状态检测：整体异常度低或最高概率低于阈值都视为正常
    normal_weight = 0.0
    if overall_score < cfg.overall_threshold:
        normal_weight = 1.0 - overall_score / (cfg.overall_threshold + 1e-12)

    max_fault_prob = max(fault_probs.values())
    if max_fault_prob < cfg.max_prob_threshold:
        normal_weight = max(normal_weight, cfg.max_prob_threshold - max_fault_prob)

    fault_scale = max(0.0, 1.0 - normal_weight)
    scaled_faults = {k: v * fault_scale for k, v in fault_probs.items()}

    total = normal_weight + sum(scaled_faults.values())
    if total <= 1e-12:
        normalized = {"正常": 1.0}
    else:
        normalized = {"正常": normal_weight / total}
        normalized.update({k: v / total for k, v in scaled_faults.items()})

    max_prob = max(normalized.values())
    is_normal = normalized.get("正常", 0.0) >= 0.5 or max_fault_prob < cfg.max_prob_threshold

    return {
        "probabilities": normalized,
        "max_prob": max_prob,
        "is_normal": is_normal,
        "uncertainty": 1 - max_prob,
        "overall_score": overall_score,
    }

def system_level_infer_er(features: Dict[str, float], config: SystemBRBConfig | None = None) -> Dict[str, float]:
    """显式 ER 版本入口，保持向后兼容。"""
    cfg = config or SystemBRBConfig()
    return _system_level_infer_er(features, cfg)


def system_level_infer_simple(features: Dict[str, float], config: SystemBRBConfig | None = None) -> Dict[str, float]:
    """无正常状态回退的简化版推理，兼容旧脚本的 `mode="simple"`。"""
    cfg = config or SystemBRBConfig()
    simple_cfg = SystemBRBConfig(
        alpha=cfg.alpha,
        overall_threshold=0.0,
        max_prob_threshold=0.0,
        attribute_weights=cfg.attribute_weights,
        rule_weights=cfg.rule_weights,
    )
    return _system_level_infer_er(features, simple_cfg)


def system_level_infer_sub_brb(features: Dict[str, float], config: SystemBRBConfig | None = None) -> Dict[str, float]:
    """使用子BRB架构的系统级推理（优化版本）。
    
    对应小论文分层推理架构：
    - 将系统级推理拆分为三个子BRB（幅度、频率、参考电平）
    - 使用特征分流，每个子BRB只接收相关特征
    - 聚合子BRB结果，输出最终诊断
    
    Parameters
    ----------
    features : dict
        输入特征字典。
    config : SystemBRBConfig, optional
        配置参数。
        
    Returns
    -------
    dict
        推理结果，格式与 system_level_infer_er 兼容。
    """
    cfg = config or SystemBRBConfig()
    
    try:
        from .aggregator import system_level_infer_with_sub_brbs
        
        result = system_level_infer_with_sub_brbs(
            features,
            alpha=cfg.alpha,
            overall_threshold=cfg.overall_threshold,
            max_prob_threshold=cfg.max_prob_threshold,
            use_feature_routing=True
        )
        
        # 转换为兼容格式
        return {
            "probabilities": result["probabilities"],
            "max_prob": result["max_prob"],
            "is_normal": result["is_normal"],
            "uncertainty": result["uncertainty"],
            "overall_score": result["overall_score"],
        }
    except ImportError:
        # 如果无法导入聚合器，回退到原始实现
        return system_level_infer_er(features, config=config)


def system_level_infer(
    features: Dict[str, float],
    config: SystemBRBConfig | None = None,
    mode: str | None = None,
) -> Dict[str, float]:
    """
    Perform system-level BRB inference.

    Parameters
    ----------
    features : dict
        Dictionary containing at least X1~X5 or the equivalent
        descriptive names. Values should be float magnitudes before
        normalization.
    config : SystemBRBConfig, optional
        Hyper-parameters controlling softmax temperature, thresholds and
        weights. Defaults to values suggested in ACCURACY_IMPROVEMENT.md.
    mode : {"er", "simple", "sub_brb"}, optional
        Compatibility flag:
        - ``mode="er"`` (default): 使用增强ER融合与正常状态回退
        - ``mode="simple"``: 禁用正常状态回退
        - ``mode="sub_brb"``: 使用子BRB架构（推荐，准确率更高）

    Returns
    -------
    dict
        Keys: `probabilities`, `max_prob`, `is_normal`, `uncertainty`,
        `overall_score`. Probabilities include four classes:
        正常 / 幅度失准 / 频率失准 / 参考电平失准.
    """
    selected = (mode or "er").lower()
    if selected == "er":
        return system_level_infer_er(features, config=config)
    if selected == "simple":
        return system_level_infer_simple(features, config=config)
    if selected == "sub_brb":
        return system_level_infer_sub_brb(features, config=config)
    raise ValueError(f"Unsupported mode '{mode}', expected 'er', 'simple', or 'sub_brb'.")

