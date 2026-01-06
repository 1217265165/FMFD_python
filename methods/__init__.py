"""Unified interface for comparison methods (ours + five baselines).

Each submodule exposes a class with `infer_system` and `infer_module`
methods so `pipelines/compare_methods.py` can call them consistently.
"""

from .ours.interface import OursMethod
from .hcf.interface import HCFMethod
from .brb_mu.interface import BRBMUMethod
from .dbrb.interface import DBRBMethod
from .aifd.interface import AIFDMethod
from .a_ibrb.interface import AIBRBMethod
from .brb_p.interface import BRBPMethod

__all__ = [
    "OursMethod",
    "HCFMethod",
    "BRBMUMethod",
    "DBRBMethod",
    "AIFDMethod",
    "AIBRBMethod",
    "BRBPMethod",
]
