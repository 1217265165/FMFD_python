"""HCF (Zhang 2022): 分层认知框架实现版本。

实现要点：
- 系统层与模块层均使用完整特征集，不做显式特征筛选。
- 分层推理：先跑系统层，系统概率作为上下文传递到模块层，但不对模块特征做过滤，保持“认知层层递进”的思路。
- 规则构建较为完整，因此阈值更保守，体现高维/规则数偏大的特征。
"""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import SystemBRBConfig, system_level_infer
from methods.base import BaseComparisonMethod


class HCFMethod(BaseComparisonMethod):
    name = "hcf"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        # 使用完整特征集合，强调分层认知下的“全面观测”
        cfg = SystemBRBConfig(
            alpha=2.0,
            attribute_weights=(0.22, 0.21, 0.19, 0.2, 0.18),
            overall_threshold=0.18,
            max_prob_threshold=0.3,
        )
        result = system_level_infer(features, cfg)
        # 保守决策阈值：规则较多、组合更完整时，避免过早决策
        result["decision_threshold"] = cfg.max_prob_threshold
        probs = result.get("probabilities", {})
        fault_probs = {k: v for k, v in probs.items() if k != "正常"}
        if fault_probs:
            max_label = max(fault_probs, key=fault_probs.get)
            max_prob = fault_probs[max_label]
            result["predicted_label"] = "正常" if max_prob < cfg.max_prob_threshold else max_label
        return result

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        # 模块层同样使用完整特征，但将系统概率作为“上下文”柔性加权
        sys_probs = sys_result.get("probabilities", {})
        mod_probs = module_level_infer(features, sys_probs)
        # 如果系统层指向某些故障，适度放大对应模块概率，体现层级认知传递
        if sys_probs:
            factor = 1 + min(0.3, sys_probs.get("幅度失准", 0) * 0.4 + sys_probs.get("频率失准", 0) * 0.3 + sys_probs.get("参考电平失准", 0) * 0.3)
            mod_probs = {k: v * factor for k, v in mod_probs.items()}
            total = sum(mod_probs.values()) + 1e-12
            mod_probs = {k: v / total for k, v in mod_probs.items()}
        return mod_probs
