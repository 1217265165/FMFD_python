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
    attribute_weights : Tuple[float, float, float, float, float]
        Relative importance of X1~X5 when computing the aggregated
        anomaly score and rule activation strength.
    rule_weights : Tuple[float, float, float]
        Weights for amplitude/frequency/reference rule groups to mimic
        knowledge-driven rule compression.
    """

    alpha: float = 2.0
    overall_threshold: float = 0.18
    max_prob_threshold: float = 0.3
    attribute_weights: Tuple[float, float, float, float, float] = (
        0.26,
        0.22,
        0.18,
        0.17,
        0.17,
    )
    rule_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)


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
    """Compute normalized attribute scores for X1~X5.

    The function is robust to different key names and accepts either
    X1~X5 or descriptive names used in the baseline scripts.
    """

    x1_raw = abs(_get_feature(features, "X1", "amplitude_offset", "bias"))
    x2_raw = _get_feature(features, "X2", "inband_flatness", "ripple_var")
    x3_raw = abs(_get_feature(features, "X3", "hf_attenuation_slope", "res_slope"))
    x4_raw = abs(_get_feature(features, "X4", "freq_scale_nonlinearity", "df"))
    x5_raw = abs(_get_feature(features, "X5", "scale_consistency", "amp_scale_consistency", "gain_consistency"))

    scores = {
        "X1": _normalize_feature(x1_raw, 0.02, 0.5),
        "X2": _normalize_feature(x2_raw, 0.002, 0.05),
        "X3": _normalize_feature(x3_raw, 1e-12, 1e-9),
        "X4": _normalize_feature(x4_raw, 5e5, 3e7),
        "X5": _normalize_feature(x5_raw, 0.01, 0.35),
    }
    return scores


def _attribute_match_degrees(scores: Dict[str, float]) -> Dict[str, Tuple[float, float, float]]:
    return {name: _triangular_membership(value, 0.15, 0.35, 0.7) for name, value in scores.items()}


def _aggregate_score(scores: Dict[str, float], weights: Tuple[float, float, float, float, float]) -> float:
    weighted = [scores[key] * w for key, w in zip(["X1", "X2", "X3", "X4", "X5"], weights)]
    return sum(weighted) / (sum(weights) + 1e-12)


def _system_level_infer_er(features: Dict[str, float], cfg: SystemBRBConfig) -> Dict[str, float]:
    """核心 ER 版系统级推理实现，供不同接口复用。"""
    scores = _compute_attribute_scores(features)
    match = _attribute_match_degrees(scores)

    # Rule compression: one dominant rule per fault family
    amp_activation = cfg.rule_weights[0] * max(match["X1"][2], match["X2"][2], match["X5"][2])
    freq_activation = cfg.rule_weights[1] * match["X4"][2]
    ref_activation = cfg.rule_weights[2] * max(match["X2"][0], match["X3"][2], match["X5"][0])

    overall_score = _aggregate_score(scores, cfg.attribute_weights)
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
    mode : {"er", "simple"}, optional
        Compatibility flag. ``mode="er"`` (default) uses the full
        enhanced ER fusion with正常回退；``mode="simple"`` disables the
        normal-state fallback to mimic旧脚本。其他值会抛出 ValueError。

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
    raise ValueError(f"Unsupported mode '{mode}', expected 'er' or 'simple'.")

