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
    
    Enhanced with X1-X22 features for improved accuracy while maintaining framework.
    
    Complexity:
    - Rules: system layer + module layer (configured rules only)
    - Params: attribute weights + rule weights + belief degrees (增加扩展特征权重)
    """
    
    name = "ours"
    
    def __init__(self):
        self.config = SystemBRBConfig()  # 默认启用扩展特征
        self.feature_names = None
        self.n_system_rules = 12  # Configured in brb_rules.yaml
        self.n_module_rules = 33  # Configured per-module rules
        self.n_params = 58  # 增加：22个特征权重 + 3个规则权重 + 33个belief参数
        self.kd_features = [f'X{i}' for i in range(1, 23)]  # X1-X22
        # 添加别名映射
        self.kd_features_aliases = {
            'bias': 'X1', 'ripple_var': 'X2', 'res_slope': 'X3', 
            'df': 'X4', 'scale_consistency': 'X5',
            'ripple_variance': 'X6', 'gain_nonlinearity': 'X7', 
            'lo_leakage': 'X8', 'tuning_linearity_residual': 'X9',
            'band_amplitude_consistency': 'X10',
            'env_overrun_rate': 'X11', 'env_overrun_max': 'X12',
            'env_violation_energy': 'X13', 'band_residual_low': 'X14',
            'band_residual_high_std': 'X15',
            'corr_shift_bins': 'X16', 'warp_scale': 'X17', 'warp_bias': 'X18',
            'slope_low': 'X19', 'kurtosis_detrended': 'X20',
            'peak_count_residual': 'X21', 'ripple_dom_freq_energy': 'X22',
        }
    
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
        """Predict on test data using hierarchical BRB with extended features."""
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
            # Convert sample to feature dict (支持X1-X22)
            features = self._array_to_dict(X_test[i])
            
            # System-level inference
            sys_result = system_level_infer(features, self.config)
            probs = sys_result.get('probabilities', {})
            
            # Map to probability array with better fallback handling
            # Order: 正常(0), 幅度失准(1), 频率失准(2), 参考电平失准(3)
            total_prob = sum(probs.values()) if probs else 0.0
            
            if total_prob > 0.01:  # Valid probabilities
                sys_proba[i, 0] = probs.get('正常', 0.0)
                sys_proba[i, 1] = probs.get('幅度失准', 0.0)
                sys_proba[i, 2] = probs.get('频率失准', 0.0)
                sys_proba[i, 3] = probs.get('参考电平失准', 0.0)
                
                # Normalize if needed
                row_sum = np.sum(sys_proba[i])
                if row_sum > 0:
                    sys_proba[i] /= row_sum
            else:
                # Fallback: use uniform distribution
                sys_proba[i] = np.ones(n_sys_classes) / n_sys_classes
            
            sys_pred[i] = np.argmax(sys_proba[i])
            
            # Module-level inference (knowledge-driven gating with feature streaming)
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
        """Convert numpy array to feature dict, supporting X1-X22."""
        if self.feature_names is None:
            # Default mapping: 假设按X1-X22顺序排列
            feature_dict = {}
            for i in range(min(len(x), 22)):
                feature_dict[f'X{i+1}'] = float(x[i])
            # 填充缺失特征
            for i in range(len(x), 22):
                feature_dict[f'X{i+1}'] = 0.0
        else:
            feature_dict = {}
            for i, name in enumerate(self.feature_names):
                if i < len(x):
                    # 支持别名映射
                    canonical_name = self.kd_features_aliases.get(name, name)
                    feature_dict[canonical_name] = float(x[i])
                    # 同时保留原名
                    feature_dict[name] = float(x[i])
        
        # 确保所有X1-X22都存在
        for i in range(1, 23):
            key = f'X{i}'
            if key not in feature_dict:
                feature_dict[key] = 0.0
        
        # Ensure required aliases for backward compatibility
        feature_dict.setdefault('bias', feature_dict.get('X1', 0.0))
        feature_dict.setdefault('ripple_var', feature_dict.get('X2', 0.0))
        feature_dict.setdefault('res_slope', feature_dict.get('X3', 0.0))
        feature_dict.setdefault('df', feature_dict.get('X4', 0.0))
        feature_dict.setdefault('scale_consistency', feature_dict.get('X5', 0.0))
        
        return feature_dict
