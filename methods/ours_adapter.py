"""Adapter for Ours method (knowledge-driven rule compression + hierarchical BRB)."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from methods.base import MethodAdapter
from BRB.system_brb import system_level_infer, SystemBRBConfig
from BRB.module_brb import module_level_infer


class OursAdapter(MethodAdapter):
    """Our proposed method: Knowledge-driven rule compression + hierarchical BRB.
    
    MUST-HAVE mechanisms:
    - Two-layer inference: System BRB -> Module BRB
    - System result gating: only activate physically-related module subset
    - Knowledge mapping: modules use relevant frequency bands/features only
    
    Complexity:
    - Rules: system layer + module layer (configured rules only)
    - Params: attribute weights + rule weights + belief degrees
    """
    
    name = "ours"
    
    def __init__(self):
        self.config = SystemBRBConfig()
        self.feature_names = None
        self.n_system_rules = 12  # Configured in brb_rules.yaml
        self.n_module_rules = 33  # Configured per-module rules
        self.n_params = 38  # Attribute weights + rule weights + belief degrees
        self.kd_features = ['X1', 'X2', 'X3', 'X4', 'X5', 
                           'bias', 'ripple_var', 'res_slope', 'df', 'scale_consistency']
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit method (rule-based, minimal training).
        
        For BRB, we don't do intensive training, but we can:
        - Store feature statistics for normalization
        - Optionally tune attribute/rule weights (simplified)
        """
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        
        # For this implementation, rules are pre-configured
        # Could add lightweight parameter tuning here if needed
        pass
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict on test data using hierarchical BRB."""
        n_test = len(X_test)
        n_sys_classes = 4  # Normal, Amp, Freq, Ref
        
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        
        # Initialize outputs
        sys_proba = np.zeros((n_test, n_sys_classes))
        sys_pred = np.zeros(n_test, dtype=int)
        mod_proba = np.zeros((n_test, 21))  # 21 modules
        mod_pred = np.zeros(n_test, dtype=int)
        
        start_time = time.time()
        
        for i in range(n_test):
            # Convert sample to feature dict
            features = self._array_to_dict(X_test[i])
            
            # System-level inference
            sys_result = system_level_infer(features, self.config)
            probs = sys_result.get('probabilities', {})
            
            # Map to probability array
            sys_proba[i, 0] = probs.get('正常', 0.25)
            sys_proba[i, 1] = probs.get('幅度失准', 0.25)
            sys_proba[i, 2] = probs.get('频率失准', 0.25)
            sys_proba[i, 3] = probs.get('参考电平失准', 0.25)
            
            sys_pred[i] = np.argmax(sys_proba[i])
            
            # Module-level inference (knowledge-driven gating)
            mod_probs_dict = module_level_infer(features, probs)
            
            # Convert to array (assume module IDs 1-21)
            for mod_id_str, prob in mod_probs_dict.items():
                try:
                    # Extract module ID from string like "模块1" or "1"
                    if isinstance(mod_id_str, str):
                        mod_id = int(''.join(filter(str.isdigit, mod_id_str)))
                    else:
                        mod_id = int(mod_id_str)
                    
                    if 1 <= mod_id <= 21:
                        mod_proba[i, mod_id - 1] = prob
                except (ValueError, IndexError):
                    continue
            
            if np.sum(mod_proba[i]) > 0:
                mod_pred[i] = np.argmax(mod_proba[i])
            else:
                mod_pred[i] = 0
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000 if n_test > 0 else 0.0
        
        return {
            'system_proba': sys_proba,
            'system_pred': sys_pred,
            'module_proba': mod_proba,
            'module_pred': mod_pred + 1,  # Convert to 1-based
            'meta': {
                'fit_time_sec': 0.0,  # Rule-based, no training
                'infer_time_ms_per_sample': infer_time_ms,
                'n_rules': self.n_system_rules + self.n_module_rules,
                'n_params': self.n_params,
                'n_features_used': len(self.kd_features),
                'features_used': self.kd_features,
            }
        }
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        return {
            'n_rules': self.n_system_rules + self.n_module_rules,
            'n_params': self.n_params,
            'n_features_used': len(self.kd_features),
        }
    
    def _array_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to feature dict."""
        if self.feature_names is None:
            # Default mapping
            feature_dict = {
                'X1': float(x[0]) if len(x) > 0 else 0.0,
                'X2': float(x[1]) if len(x) > 1 else 0.0,
                'X3': float(x[2]) if len(x) > 2 else 0.0,
                'X4': float(x[3]) if len(x) > 3 else 0.0,
                'X5': float(x[4]) if len(x) > 4 else 0.0,
            }
        else:
            feature_dict = {}
            for i, name in enumerate(self.feature_names):
                if i < len(x):
                    feature_dict[name] = float(x[i])
        
        # Ensure required aliases
        feature_dict.setdefault('bias', feature_dict.get('X1', 0.0))
        feature_dict.setdefault('ripple_var', feature_dict.get('X2', 0.0))
        feature_dict.setdefault('res_slope', feature_dict.get('X3', 0.0))
        feature_dict.setdefault('df', feature_dict.get('X4', 0.0))
        feature_dict.setdefault('scale_consistency', feature_dict.get('X5', 0.0))
        
        return feature_dict
