"""A-IBRB: 自动化 interval-BRB（弱化专家依赖）。"""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import SystemBRBConfig, system_level_infer
from methods.base import BaseComparisonMethod


def _collapse_interval(features: Dict[str, float]) -> Dict[str, float]:
    collapsed = features.copy()
    for key, value in list(features.items()):
        if key.endswith("_interval") and isinstance(value, (list, tuple)) and len(value) == 2:
            collapsed[key[:-9]] = sum(value) / 2
        # 自动区间化：如果没有提供区间，按值的 20% 范围生成粗分段
        elif key in {"X1", "X2", "X3", "X4", "X5", "bias", "ripple_var", "res_slope", "df", "scale_consistency"}:
            span = abs(value) * 0.2
            if span:
                collapsed[f"{key}_interval"] = [value - span, value + span]
                collapsed[key] = value
    return collapsed


class AIBRBMethod(BaseComparisonMethod):
    name = "a_ibrb"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        collapsed = _collapse_interval(features)
        # Interval 可信度：区间越宽越保守，平坦度/参考类占比更高
        span_boost = {}
        for key in ["X1", "X2", "X3", "X4", "X5"]:
            interval = features.get(f"{key}_interval")
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                span = abs(interval[1] - interval[0])
                span_boost[key] = 1 + min(0.6, span)
        if span_boost:
            collapsed["ripple_var"] = collapsed.get("ripple_var", collapsed.get("X2", 0.0)) * span_boost.get("X2", 1.0)
            collapsed["res_slope"] = collapsed.get("res_slope", collapsed.get("X3", 0.0)) * span_boost.get("X3", 1.0)
            collapsed["scale_consistency"] = collapsed.get("scale_consistency", collapsed.get("X5", 0.0)) * span_boost.get("X5", 1.0)

        cfg = SystemBRBConfig(
            # Interval-BRB 弱化专家权重，采用更加均衡的属性重要度
            attribute_weights=(0.22, 0.21, 0.19, 0.19, 0.19),
            # 轻微抬高正常判定阈值，避免区间不确定性放大异常
            overall_threshold=0.2,
        )
        res = system_level_infer(collapsed, cfg)
        res["decision_threshold"] = 0.35
        probs = res.get("probabilities", {})
        fault_probs = {k: v for k, v in probs.items() if k != "正常"}
        max_label = max(fault_probs, key=fault_probs.get)
        max_prob = fault_probs[max_label]
        res["predicted_label"] = "正常" if max_prob < res["decision_threshold"] else max_label
        return res

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        collapsed = _collapse_interval(features)
        return module_level_infer(collapsed, sys_result.get("probabilities", {}))
