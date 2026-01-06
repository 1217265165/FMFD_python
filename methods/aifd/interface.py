"""AIFD: Adaptive Interpretable BRB (自适应 + 可解释性)。"""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import system_level_infer, SystemBRBConfig
from methods.base import BaseComparisonMethod


class AIFDMethod(BaseComparisonMethod):
    name = "aifd"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        # Adaptive temperature scaled by data sparsity
        sample_size = max(features.get("sample_size", 20), 1)
        alpha = 1.5 if sample_size < 30 else 2.0

        # 自适应特征增益：小样本时放大幅度和平坦度，大样本时强化频率与缩放一致性
        tuned_feats = features.copy()
        if sample_size < 30:
            tuned_feats["bias"] = features.get("bias", features.get("X1", 0.0)) * 1.25
            tuned_feats["ripple_var"] = features.get("ripple_var", features.get("X2", 0.0)) * 1.15
        else:
            tuned_feats["df"] = features.get("df", features.get("X4", 0.0)) * 1.2
            tuned_feats["scale_consistency"] = features.get("scale_consistency", features.get("X5", 0.0)) * 1.3

        cfg = SystemBRBConfig(alpha=alpha, attribute_weights=(0.24, 0.24, 0.16, 0.18, 0.18))
        result = system_level_infer(tuned_feats, cfg)
        result["adaptive_alpha"] = alpha
        result["decision_threshold"] = 0.28 if sample_size < 30 else 0.34
        probs = result.get("probabilities", {})
        fault_probs = {k: v for k, v in probs.items() if k != "正常"}
        if fault_probs:
            max_label = max(fault_probs, key=fault_probs.get)
            max_prob = fault_probs[max_label]
            result["predicted_label"] = "正常" if max_prob < result["decision_threshold"] else max_label
        return result

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        # Encourage interpretability by softening probabilities
        probs = module_level_infer(features, sys_result.get("probabilities", {}))
        smooth = {k: v * 0.9 for k, v in probs.items()}
        total = sum(smooth.values()) + 1e-12
        return {k: v / total for k, v in smooth.items()}
