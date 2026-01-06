"""Common utilities for comparison methods."""
from __future__ import annotations

from typing import Dict


class BaseComparisonMethod:
    """Base class ensuring a unified interface for comparison pipelines."""

    name: str = "base"

    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def infer_module(self, features: Dict[str, float], sys_result: Dict[str, float]) -> Dict[str, float]:
        return {}

    def run_method(self, features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        system = self.infer_system(features)
        module = self.infer_module(features, system)
        return {"system": system, "module": module}
