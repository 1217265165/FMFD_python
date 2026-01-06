"""BRB-P: Probability-constrained BRB baseline (概率表初始化 + 解释性约束)."""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import SystemBRBConfig, system_level_infer
from methods.base import BaseComparisonMethod


class BRBPMethod(BaseComparisonMethod):
    name = "brb_p"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        cfg = SystemBRBConfig(
            # 概率表约束：增大 α，减少 overall 阈值，强化最大类概率
            alpha=2.6,
            overall_threshold=0.12,
            max_prob_threshold=0.32,
            rule_weights=(1.1, 0.9, 1.0),
        )

        # 基于样本统计生成“概率表”先验：幅度/频率/参考对应三个类别
        amp = features.get("bias", features.get("X1", 0.0))
        freq_dev = features.get("df", features.get("X4", 0.0))
        ref = features.get("scale_consistency", features.get("X5", 0.0))
        total = abs(amp) + abs(freq_dev) + abs(ref) + 1e-12
        prob_table = {
            "幅度失准": abs(amp) / total,
            "频率失准": abs(freq_dev) / total,
            "参考电平失准": abs(ref) / total,
        }

        # 依据概率表对输入特征做初始化重标定
        tuned_feats = features.copy()
        tuned_feats["bias"] = amp * (1.0 + 0.4 * prob_table["幅度失准"])
        tuned_feats["df"] = freq_dev * (1.0 + 0.4 * prob_table["频率失准"])
        tuned_feats["scale_consistency"] = ref * (1.0 + 0.4 * prob_table["参考电平失准"])

        result = system_level_infer(tuned_feats, cfg)
        probs = result.get("probabilities", {})

        # 解释性约束：限制规则后件概率偏移幅度，避免与先验严重冲突
        constrained = {}
        for label in ["幅度失准", "频率失准", "参考电平失准"]:
            after = probs.get(label, 0.0)
            prior = prob_table.get(label, 1 / 3)
            constrained[label] = 0.7 * after + 0.3 * prior

        norm = sum(constrained.values()) + 1e-12
        constrained = {k: v / norm for k, v in constrained.items()}
        result["probabilities"] = {"正常": probs.get("正常", 0.0), **constrained}
        result["decision_threshold"] = cfg.max_prob_threshold
        if constrained:
            max_label = max(constrained, key=constrained.get)
            max_prob = constrained[max_label]
            result["predicted_label"] = "正常" if max_prob < cfg.max_prob_threshold else max_label
        return result

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        return module_level_infer(features, sys_result.get("probabilities", {}))
