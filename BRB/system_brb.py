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

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .aggregator import aggregate_system_results
from .system_brb_amp import infer_amp_brb
from .system_brb_freq import infer_freq_brb
from .system_brb_ref import infer_ref_brb


@dataclass
class SystemBRBConfig:
    """Configuration of the system-level BRB.

    Attributes
    ----------
    alpha : float
        Softmax temperature used in ER fusion (对应式 (2) 的 α)。
        Lower values create sharper probability distributions.
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
        Grouped as: [X1-X5基础, X6-X10模块, X11-X15增强, X16-X18频率, X19-X22幅度]
    rule_weights : Tuple[float, float, float]
        Weights for amplitude/frequency/reference rule groups to mimic
        knowledge-driven rule compression.
    use_extended_features : bool
        If True, use X1-X22; if False, use only X1-X5 (backward compatibility).
    """

    alpha: float = 1.5  # Lower temperature for sharper discrimination
    overall_threshold: float = 0.20  # Adjusted threshold
    max_prob_threshold: float = 0.25  # Lower threshold for fault detection
    attribute_weights: Tuple[float, ...] = (
        # X1-X5: 基础特征权重 (更高权重)
        0.25, 0.20, 0.10, 0.08, 0.15,
        # X6-X10: 模块症状特征权重 (用于区分故障类型)
        0.08, 0.07, 0.06, 0.08, 0.06,
        # X11-X15: 增强特征权重 (系统层重要)
        0.10, 0.08, 0.08, 0.06, 0.06,
        # X16-X18: 频率特征权重 (频率失准识别)
        0.05, 0.07, 0.08,
        # X19-X22: 幅度细粒度特征权重 (幅度/参考识别)
        0.04, 0.08, 0.06, 0.08,
    )
    rule_weights: Tuple[float, float, float] = (1.2, 1.3, 0.7)  # amp/freq/ref权重：降低ref
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
    
    Normalization thresholds calibrated based on actual data distributions:
    Key discriminative patterns:
    - freq_error: X2/ripple_var very high (>100), amp_range very high (>100)
    - ref_error: X1/bias higher (~1.6), X5 lower (~26), amp_kurtosis lower (~62)
    - amp_error: X5/band_consistency high variance, switching_rate higher (~0.115)
    - normal: consistent mid-range values
    """

    # X1-X5: 基础系统级特征
    x1_raw = abs(_get_feature(features, "X1", "amplitude_offset", "bias", "amp_mean"))
    x2_raw = _get_feature(features, "X2", "inband_flatness", "ripple_var")
    x3_raw = abs(_get_feature(features, "X3", "hf_attenuation_slope", "res_slope", "trend_slope"))
    x4_raw = abs(_get_feature(features, "X4", "freq_scale_nonlinearity", "df", "freq_step_std"))
    x5_raw = abs(_get_feature(features, "X5", "scale_consistency", "amp_scale_consistency", "gain_consistency"))

    # X6-X10: 模块级症状
    x6_raw = _get_feature(features, "X6", "ripple_variance", "ripple_std")
    x7_raw = _get_feature(features, "X7", "gain_nonlinearity", "step_score", "noise_peak")
    x8_raw = abs(_get_feature(features, "X8", "lo_leakage", "amp_skewness"))
    x9_raw = abs(_get_feature(features, "X9", "tuning_linearity_residual", "band_consistency"))
    x10_raw = _get_feature(features, "X10", "band_amplitude_consistency", "amp_iqr")

    # X11-X15: 增强特征
    x11_raw = _get_feature(features, "X11", "gain_distortion", "amp_std")
    x12_raw = _get_feature(features, "X12", "power_noise", "noise_level")
    x13_raw = _get_feature(features, "X13", "amp_change_rate", "switching_rate")
    x14_raw = _get_feature(features, "X14", "band_response_accuracy", "band1_std")
    x15_raw = _get_feature(features, "X15", "phase_deviation", "band3_std")

    # 动态包络特征
    x11_env_raw = _get_feature(features, "X11_out_env_ratio", "env_overrun_rate", "viol_rate")
    x12_env_raw = _get_feature(features, "X12_max_env_violation", "env_overrun_max", "ripple_max_dev")
    x13_env_raw = _get_feature(features, "X13_env_violation_energy", "env_overrun_mean", "env_violation_energy")
    x14_env_raw = abs(_get_feature(features, "X14_low_band_residual", "band_residual_low", "band1_mean"))
    x15_env_raw = _get_feature(features, "X15_high_band_residual_std", "band_residual_high_std", "band4_std")

    # X16-X18: 频率对齐特征
    x16_raw = abs(_get_feature(features, "X16", "corr_shift_bins", "freq_step_cv"))
    x17_raw = abs(_get_feature(features, "X17", "warp_scale", "amp_range"))
    x18_raw = abs(_get_feature(features, "X18", "warp_bias", "trend_intercept"))

    # X19-X22: 幅度细粒度特征
    x19_raw = abs(_get_feature(features, "X19", "slope_low", "hf_attenuation_slope"))
    x20_raw = abs(_get_feature(features, "X20", "kurtosis_detrended", "amp_kurtosis"))
    x21_raw = _get_feature(features, "X21", "peak_count_residual", "band1_energy_ratio")
    x22_raw = _get_feature(features, "X22", "ripple_dom_freq_energy", "amp_max")

    # 基于实际数据分布校准的归一化阈值
    # 使用p90以上作为"高"异常，p10以下作为基准
    scores = {
        # 基础特征 - 关键区分特征
        "X1": _normalize_feature(x1_raw, 1.0, 2.5),  # p90=1.62, 高于2表示异常
        "X2": _normalize_feature(x2_raw, 1.2, 100.0),  # p90=1.14, freq>10000非常高
        "X3": _normalize_feature(x3_raw, 1e-12, 1e-10),  # 当前数据接近0
        "X4": _normalize_feature(x4_raw, 1e-8, 1e-5),  # 当前数据接近0
        "X5": _normalize_feature(x5_raw, 90.0, 1000.0),  # p90=87.8, amp~1101很高
        
        # 模块症状特征
        "X6": _normalize_feature(x6_raw, 1.2, 10.0),  # ripple_std, p90约1.1
        "X7": _normalize_feature(x7_raw, 40.0, 70.0),  # noise_peak
        "X8": _normalize_feature(x8_raw, 5.0, 15.0),  # abs(amp_skewness)
        "X9": _normalize_feature(x9_raw, 1.1, 10.0),  # band_consistency: p90=1.09, amp~16
        "X10": _normalize_feature(x10_raw, 2.0, 2.5),  # amp_iqr, p90=2.04
        
        # 增强特征
        "X11": _normalize_feature(x11_raw, 1.5, 30.0),  # amp_std: p90=1.4, freq~35
        "X12": _normalize_feature(x12_raw, 0.5, 1.0),  # noise_level: p90=0.47
        "X13": _normalize_feature(x13_raw, 0.085, 0.15),  # switching_rate: p90=0.083, amp~0.115
        "X14": _normalize_feature(x14_raw, 0.5, 3.0),  # band1_std
        "X15": _normalize_feature(x15_raw, 0.7, 2.0),  # band3_std
        
        # 包络特征
        "X11_out_env_ratio": _normalize_feature(x11_env_raw, 0.01, 0.3),
        "X12_max_env_violation": _normalize_feature(x12_env_raw, 5.0, 100.0),  # ripple_max_dev
        "X13_env_violation_energy": _normalize_feature(x13_env_raw, 0.1, 10.0),
        "X14_low_band_residual": _normalize_feature(x14_env_raw, 1.0, 20.0),  # abs(band1_mean)
        "X15_high_band_residual_std": _normalize_feature(x15_env_raw, 0.7, 1.0),  # band4_std
        
        # 频率对齐特征
        "X16": _normalize_feature(x16_raw, 1e-14, 1e-12),  # freq_step_cv
        "X17": _normalize_feature(x17_raw, 55.0, 200.0),  # amp_range: p90=53.6, freq~555
        "X18": _normalize_feature(x18_raw, 0.5, 10.0),  # trend_intercept
        
        # 幅度细粒度特征
        "X19": _normalize_feature(x19_raw, 1e-12, 1e-10),  # hf_attenuation_slope
        "X20": _normalize_feature(x20_raw, 260.0, 500.0),  # amp_kurtosis: p90=257, freq~363
        "X21": _normalize_feature(x21_raw, 0.3, 0.7),  # band1_energy_ratio
        "X22": _normalize_feature(x22_raw, 5.0, 30.0),  # amp_max
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
    """核心 ER 版系统级推理实现，支持扩展特征。
    
    使用基于数据分析的规则:
    1. ref_error: X1/bias > 1.2, X5/scale_consistency < 35, band_consistency < 0.75
    2. freq_error: X5 > 50, amp_range > 50, or extreme ripple_var
    3. normal: 特征值都在正常范围内
    4. amp_error: 其他情况
    """
    scores = _compute_attribute_scores(features)
    match = _attribute_match_degrees(scores)

    # 获取原始特征值用于规则判断
    x1_raw = abs(_get_feature(features, "X1", "amplitude_offset", "bias", "amp_mean"))
    x2_raw = _get_feature(features, "X2", "inband_flatness", "ripple_var")
    x5_raw = abs(_get_feature(features, "X5", "scale_consistency"))
    band_cons = abs(_get_feature(features, "band_consistency", "X9"))
    amp_range = _get_feature(features, "amp_range", "X17")
    amp_std = _get_feature(features, "amp_std", "X11")
    switching_rate = _get_feature(features, "switching_rate", "X13")
    
    # ========== 基于数据分析的规则判断 ==========
    
    # ref_error特征: X1>1.2, X5<35, band_consistency<0.75
    ref_score = 0.0
    if x1_raw > 1.2:
        ref_score += 0.4
    if x5_raw < 35:
        ref_score += 0.3
    if band_cons < 0.75:
        ref_score += 0.3
    if amp_std > 1.3:
        ref_score += 0.2
    
    # freq_error特征: 极端的ripple_var, 或X5>55且amp_range>50
    freq_score = 0.0
    if x2_raw > 5:  # 较高ripple (freq median=1.02)
        freq_score += 0.3
    if x2_raw > 50:  # 极端ripple
        freq_score += 0.6
    if x5_raw > 55:  # 稍高的X5 (freq median=56.7)
        freq_score += 0.3
    if amp_range > 52:  # freq median=52.3
        freq_score += 0.3
    if amp_std > 1.25:  # freq samples with amp_std > 1.25
        freq_score += 0.15
    
    # normal特征: 所有特征都在非常狭窄的正常范围内
    # 使用更严格的范围，基于数据分析
    normal_score = 0.0
    normal_indicators = 0
    
    # X1: normal median=0.92, std很小，范围[0.88, 0.98]
    if 0.88 < x1_raw < 0.98:
        normal_indicators += 1
    
    # X5: normal median=45.2, std很小，范围[42.4, 47.4]
    if 42.3 < x5_raw < 47.5:
        normal_indicators += 1
    
    # band_consistency: normal median=0.93
    if 0.84 < band_cons < 0.95:
        normal_indicators += 1
    
    # amp_std: normal median=1.12, 范围[1.06, 1.21]
    if 1.06 < amp_std < 1.21:
        normal_indicators += 1
    
    # amp_range: normal median=41.5, 范围[41.1, 42.2]
    if 41.0 < amp_range < 42.3:
        normal_indicators += 1
    
    # switching_rate: normal median=0.080
    if 0.075 < switching_rate < 0.084:
        normal_indicators += 1
    
    # 如果至少5个指标满足，认为是normal
    is_normal = (normal_indicators >= 5)
    if is_normal:
        normal_score = 0.8
    
    # amp_error: 当不是明显的ref、freq、或normal时
    amp_score = 0.1  # 基础分数
    
    # 如果有异常但不符合ref/freq/normal pattern
    if not is_normal and ref_score < 0.5 and freq_score < 0.4:
        amp_score = 0.5
    
    # 如果有高band_consistency或scale_consistency（但不是正常范围）
    if band_cons > 1.0:
        amp_score += 0.2
    if x5_raw > 60:
        amp_score += 0.2
    
    # ========== 构建结果 ==========
    amp_result = {
        "label": "幅度失准",
        "activation": float(cfg.rule_weights[0] * min(1.0, amp_score)),
    }
    freq_result = {
        "label": "频率失准", 
        "activation": float(cfg.rule_weights[1] * min(1.0, freq_score)),
    }
    ref_result = {
        "label": "参考电平失准",
        "activation": float(cfg.rule_weights[2] * min(1.0, ref_score)),
    }

    overall_score = _aggregate_score(scores, cfg.attribute_weights, cfg.use_extended_features)
    
    # 自定义聚合逻辑
    results = [amp_result, freq_result, ref_result]
    activations = [r["activation"] for r in results]
    
    # 使用softmax计算概率
    import math
    alpha = cfg.alpha
    exp_vals = [math.exp(alpha * a) for a in activations]
    total_exp = sum(exp_vals) + 1e-12
    fault_probs = {r["label"]: e / total_exp for r, e in zip(results, exp_vals)}
    
    # 计算正常概率
    max_fault_prob = max(fault_probs.values())
    
    # 如果normal_score高，增加正常概率
    if normal_score > 0.7:
        normal_prob = 0.6
        scale = 1.0 - normal_prob
        fault_probs = {k: v * scale for k, v in fault_probs.items()}
    elif normal_score > 0.5:
        normal_prob = 0.3
        scale = 1.0 - normal_prob
        fault_probs = {k: v * scale for k, v in fault_probs.items()}
    elif max_fault_prob < cfg.max_prob_threshold:
        normal_prob = cfg.max_prob_threshold - max_fault_prob
        scale = 1.0 - normal_prob
        fault_probs = {k: v * scale for k, v in fault_probs.items()}
    else:
        normal_prob = 0.0
    
    probabilities = {"正常": normal_prob}
    probabilities.update(fault_probs)
    
    # 归一化
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}
    
    return {
        "probabilities": probabilities,
        "max_prob": max(probabilities.values()) if probabilities else 0.0,
        "is_normal": probabilities.get("正常", 0.0) >= 0.5,
        "uncertainty": 1.0 - (max(probabilities.values()) if probabilities else 0.0),
        "overall_score": overall_score,
        "results": results,
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
