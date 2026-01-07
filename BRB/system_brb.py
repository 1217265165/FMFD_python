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


def _normalize_feature(value: float, lower: float, upper: float) -> float:
    value = max(lower, min(value, upper))
    return (value - lower) / (upper - lower + 1e-12)


def _compute_attribute_scores(features: Dict[str, float]) -> Dict[str, float]:
    """Compute normalized attribute scores for X1~X22 (扩展版).

    The function is robust to different key names and accepts either
    X1~X22 or descriptive names used in the baseline scripts.
    """

    # X1-X5: 基础系统级特征
    x1_raw = abs(_get_feature(features, "X1", "amplitude_offset", "bias"))
    x2_raw = _get_feature(features, "X2", "inband_flatness", "ripple_var")
    x3_raw = abs(_get_feature(features, "X3", "hf_attenuation_slope", "res_slope"))
    x4_raw = abs(_get_feature(features, "X4", "freq_scale_nonlinearity", "df"))
    x5_raw = abs(_get_feature(features, "X5", "scale_consistency", "amp_scale_consistency", "gain_consistency"))

    # X6-X10: 模块级症状（系统层可选使用）
    x6_raw = _get_feature(features, "X6", "ripple_variance")
    x7_raw = _get_feature(features, "X7", "gain_nonlinearity", "step_score")
    x8_raw = _get_feature(features, "X8", "lo_leakage")
    x9_raw = _get_feature(features, "X9", "tuning_linearity_residual")
    x10_raw = _get_feature(features, "X10", "band_amplitude_consistency")

    # X11-X15: 包络/残差特征（系统层使用）
    x11_raw = _get_feature(features, "X11", "env_overrun_rate", "viol_rate")
    x12_raw = _get_feature(features, "X12", "env_overrun_max")
    x13_raw = _get_feature(features, "X13", "env_violation_energy")
    x14_raw = _get_feature(features, "X14", "band_residual_low")
    x15_raw = _get_feature(features, "X15", "band_residual_high_std")

    # X16-X18: 频率对齐/形变（频率失准识别）
    x16_raw = abs(_get_feature(features, "X16", "corr_shift_bins"))
    x17_raw = abs(_get_feature(features, "X17", "warp_scale"))
    x18_raw = abs(_get_feature(features, "X18", "warp_bias"))

    # X19-X22: 幅度链路细粒度（幅度/参考异常识别）
    x19_raw = abs(_get_feature(features, "X19", "slope_low"))
    x20_raw = abs(_get_feature(features, "X20", "kurtosis_detrended"))
    x21_raw = _get_feature(features, "X21", "peak_count_residual")
    x22_raw = _get_feature(features, "X22", "ripple_dom_freq_energy")

    scores = {
        # 基础特征归一化
        "X1": _normalize_feature(x1_raw, 0.02, 0.5),
        "X2": _normalize_feature(x2_raw, 0.002, 0.05),
        "X3": _normalize_feature(x3_raw, 1e-12, 1e-9),
        "X4": _normalize_feature(x4_raw, 5e5, 3e7),
        "X5": _normalize_feature(x5_raw, 0.01, 0.35),
        # 模块症状归一化
        "X6": _normalize_feature(x6_raw, 0.001, 0.03),
        "X7": _normalize_feature(x7_raw, 0.05, 2.0),
        "X8": _normalize_feature(x8_raw, 0.01, 1.0),
        "X9": _normalize_feature(x9_raw, 1e3, 1e5),
        "X10": _normalize_feature(x10_raw, 0.02, 0.5),
        # 包络/残差特征归一化
        "X11": _normalize_feature(x11_raw, 0.01, 0.3),
        "X12": _normalize_feature(x12_raw, 0.5, 5.0),
        "X13": _normalize_feature(x13_raw, 0.1, 10.0),
        "X14": _normalize_feature(x14_raw, 0.01, 1.0),
        "X15": _normalize_feature(x15_raw, 0.01, 0.5),
        # 频率对齐特征归一化
        "X16": _normalize_feature(x16_raw, 0.001, 0.1),
        "X17": _normalize_feature(x17_raw, 0.001, 0.05),
        "X18": _normalize_feature(x18_raw, 0.001, 0.05),
        # 幅度细粒度特征归一化
        "X19": _normalize_feature(x19_raw, 1e-12, 1e-10),
        "X20": _normalize_feature(x20_raw, 0.5, 5.0),
        "X21": _normalize_feature(x21_raw, 1, 20),
        "X22": _normalize_feature(x22_raw, 0.1, 0.8),
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

