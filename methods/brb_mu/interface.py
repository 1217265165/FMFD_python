"""BRB-MU / Trustworthy Fault Diagnosis (多源不确定信息)。"""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import SystemBRBConfig, system_level_infer
from methods.base import BaseComparisonMethod


class BRBMUMethod(BaseComparisonMethod):
    name = "brb_mu"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        # 多源不确定度：测量噪声 + 数据稀疏性
        measurement_u = features.get("measurement_uncertainty", 0.1)
        sparsity_u = 0.05 + 0.25 * (1.0 / max(features.get("sample_size", 30), 1))
        total_uncertainty = min(0.6, measurement_u + sparsity_u)

        # 不同来源特征赋予不同不确定度权重
        scaled_feats = features.copy()
        amp_shrink = 1 - min(0.5, 0.6 * total_uncertainty)
        scaled_feats["bias"] = features.get("bias", features.get("X1", 0.0)) * amp_shrink
        scaled_feats["ripple_var"] = features.get("ripple_var", features.get("X2", 0.0)) * (1 + 0.8 * total_uncertainty)
        scaled_feats["res_slope"] = features.get("res_slope", features.get("X3", 0.0)) * (1 + 0.5 * total_uncertainty)
        scaled_feats["df"] = features.get("df", features.get("X4", 0.0)) * (1 + total_uncertainty)
        scaled_feats["scale_consistency"] = features.get("scale_consistency", features.get("X5", 0.0)) * (1 + 0.4 * total_uncertainty)

        cfg = SystemBRBConfig(
            alpha=1.6,
            max_prob_threshold=0.25 + 0.1 * total_uncertainty,
            overall_threshold=0.16 + 0.05 * total_uncertainty,
            attribute_weights=(0.2, 0.18, 0.16, 0.22, 0.24),
        )
        base = system_level_infer(scaled_feats, cfg)
        probs = base["probabilities"].copy()
        # 将多源不确定度注入后件，降低极端概率
        for k in ["幅度失准", "频率失准", "参考电平失准"]:
            probs[k] = probs.get(k, 0.0) * (1 - total_uncertainty) + total_uncertainty / 3
        base["probabilities"] = probs
        base["uncertainty"] = max(base.get("uncertainty", 0.0), total_uncertainty)
        decision_th = cfg.max_prob_threshold
        base["decision_threshold"] = decision_th
        fault_probs = {k: v for k, v in probs.items() if k != "正常"}
        max_label = max(fault_probs, key=fault_probs.get)
        max_prob = fault_probs[max_label]
        base["predicted_label"] = "正常" if max_prob < decision_th else max_label
        return base

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        return module_level_infer(features, sys_result.get("probabilities", {}))
