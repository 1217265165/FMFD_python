"""Deep BRB (DBRB) simplified hierarchical stack."""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import SystemBRBConfig, system_level_infer
from methods.base import BaseComparisonMethod


class DBRBMethod(BaseComparisonMethod):
    name = "dbrb"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        # XGBoost 思想：按特征重要性排序并拆成多个子集（此处用启发式替代）
        importance = {
            "bias": abs(features.get("bias", features.get("X1", 0.0))),
            "ripple_var": abs(features.get("ripple_var", features.get("X2", 0.0))) * 0.8,
            "res_slope": abs(features.get("res_slope", features.get("X3", 0.0))) * 0.9,
            "df": abs(features.get("df", features.get("X4", 0.0))) * 1.1,
            "scale_consistency": abs(features.get("scale_consistency", features.get("X5", 0.0))) * 1.0,
        }
        # 拆分成两个子集：幅度/平坦度优先 vs 高频/参考优先
        first_subset = {k: features.get(k, features.get(k.upper(), 0.0)) for k in ["bias", "ripple_var", "res_slope"]}
        second_subset = {k: features.get(k, features.get(k.upper(), 0.0)) for k in ["df", "scale_consistency", "ripple_var"]}

        # Stage-1：浅层偏重幅度与平坦度，规则更稀疏以缓解规则爆炸
        shallow_cfg = SystemBRBConfig(
            alpha=1.8,
            overall_threshold=0.23,
            attribute_weights=(0.32, 0.25, 0.21, 0.12, 0.1),
        )
        shallow = system_level_infer(first_subset, shallow_cfg)

        # Stage-2：深层强化频率/参考，模拟“深层”特征抽取
        boosted_second = second_subset.copy()
        boosted_second["df"] = second_subset.get("df", 0.0) * 1.35
        boosted_second["scale_consistency"] = second_subset.get("scale_consistency", 0.0) * 1.2
        mid_cfg = SystemBRBConfig(alpha=1.5, attribute_weights=(0.18, 0.2, 0.2, 0.27, 0.15))
        deep = system_level_infer(boosted_second, config=mid_cfg)

        probs = {}
        for k in set(shallow["probabilities"]).union(deep["probabilities"]):
            shallow_weight = 0.55 + 0.05 * (importance.get("bias", 0) > importance.get("df", 0))
            deep_weight = 1 - shallow_weight
            probs[k] = shallow_weight * shallow["probabilities"].get(k, 0.0) + deep_weight * deep["probabilities"].get(k, 0.0)

        decision_th = 0.26
        max_prob = max(probs.values())
        max_label = max(probs, key=probs.get)
        return {
            "probabilities": probs,
            "max_prob": max_prob,
            "uncertainty": 1 - max_prob,
            "is_normal": shallow.get("is_normal", False) and deep.get("is_normal", False),
            "decision_threshold": decision_th,
            "predicted_label": "正常" if max_prob < decision_th else max_label,
        }

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        return module_level_infer(features, sys_result.get("probabilities", {}))
