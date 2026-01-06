"""Common utilities for comparison methods."""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np


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


class MethodAdapter:
    """Unified adapter interface for training-based method comparison.
    
    All comparison methods must implement this interface to ensure
    consistent evaluation in compare_methods.py pipeline.
    """
    
    name: str = "base"
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray, 
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Train the method on training data.
        
        Args:
            X_train: Training features, shape (N, D)
            y_sys_train: System-level labels, shape (N,)
            y_mod_train: Module-level labels, shape (N,) or None
            meta: Optional metadata (feature names, etc.)
        """
        raise NotImplementedError
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict on test data.
        
        Args:
            X_test: Test features, shape (N, D)
            meta: Optional metadata
            
        Returns:
            dict with keys:
                - system_proba: np.ndarray shape (N, C_sys) - system-level probabilities
                - system_pred: np.ndarray shape (N,) - system-level predictions
                - module_proba: np.ndarray shape (N, 21) or None - module probabilities
                - module_pred: np.ndarray shape (N,) or None - module predictions
                - meta: dict with fit_time_sec, infer_time_ms_per_sample, n_rules, 
                        n_params, n_features_used, features_used
        """
        raise NotImplementedError
    
    def complexity(self) -> Dict:
        """Return complexity metrics.
        
        Returns:
            dict with keys:
                - n_rules: int
                - n_params: int
                - n_features_used: int
        """
        raise NotImplementedError
