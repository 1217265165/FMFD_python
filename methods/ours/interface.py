"""Ours: 知识驱动规则压缩 + 分层 BRB（本文方法）。"""
from __future__ import annotations

from typing import Dict

from BRB.module_brb import module_level_infer
from BRB.system_brb import system_level_infer
from methods.base import BaseComparisonMethod


class OursMethod(BaseComparisonMethod):
    name = "ours"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        result = system_level_infer(features)
        result.setdefault("decision_threshold", 0.3)
        return result

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        return module_level_infer(features, sys_result.get("probabilities", {}))
