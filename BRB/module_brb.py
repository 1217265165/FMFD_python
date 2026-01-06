"""
Module-level BRB reasoning (对应小论文 3.2 分层推理与规则压缩章节)。

本模块实现 21 个模块的压缩式 BRB 推理：利用系统级诊断结果
作为虚拟先验属性 V（式 (3)），仅激活与异常类型相关的规则组，
避免全组合爆炸。规则数≈45、参数≈38，显著少于传统设计。
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List

from .utils import BRBRule, SimpleBRB, normalize_feature


MODULE_LABELS: List[str] = [
    "衰减器",
    "前置放大器",
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


def _mean(values: Iterable[float]) -> float:
    arr = list(values)
    return float(statistics.mean(arr)) if arr else 0.0


def _aggregate_module_score(features: Dict[str, float]) -> float:
    """Aggregate module-level symptom scores with normalization.

    对应文中模块层虚拟属性 V 构建思路：将步进误差、分辨率斜率、
    纹波、频标偏移、限制超差率以及增益/偏置统一归一化后求平均。
    """

    md_step_raw = max(
        features.get("step_score", 0.0),
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
    )
    md_step = normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope = normalize_feature(abs(features.get("res_slope", 0.0)), 1e-12, 1e-10)
    md_ripple = normalize_feature(features.get("ripple_var", 0.0), 0.001, 0.02)
    md_df = normalize_feature(abs(features.get("df", 0.0)), 1e6, 5e7)
    md_viol = normalize_feature(features.get("viol_rate", 0.0), 0.02, 0.2)
    md_gain_bias = max(
        normalize_feature(abs(features.get("bias", 0.0)), 0.1, 1.0),
        normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.02, 0.2),
    )
    return _mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias])


def module_level_infer(features: Dict[str, float], sys_probs: Dict[str, float]) -> Dict[str, float]:
    """Perform compressed module-level BRB inference.

    Parameters
    ----------
    features : dict
        模块层特征，至少包含 step_score、res_slope、ripple_var、df、
        viol_rate、gain、bias，可额外提供 switch_step_err_max、
        nonswitch_step_max（对应式 (3) 中的症状属性）。
    sys_probs : dict
        系统级诊断概率分布，作为虚拟先验属性 V 对规则进行加权。
        既支持直接传入概率字典，也支持 system_level_infer 的完整
        返回值（会自动提取其中的 ``probabilities`` 字段）。

    Returns
    -------
    dict
        21 个模块的概率分布，顺序与 ``MODULE_LABELS`` 对齐。
    """

    md = _aggregate_module_score(features)

    probs = sys_probs.get("probabilities", sys_probs)
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)
    ref_prior = probs.get("参考电平失准", 0.3)

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
    return brb.infer([md])
