"""Adapter for Ours method (knowledge-driven rule compression + hierarchical BRB)."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from methods.base import MethodAdapter
from BRB.system_brb import system_level_infer, SystemBRBConfig
from BRB.module_brb import module_level_infer, module_level_infer_with_activation


class OursAdapter(MethodAdapter):
    """Our proposed method: Knowledge-driven rule compression + hierarchical BRB.
    
    MUST-HAVE mechanisms:
    - Two-layer inference: System BRB -> Module BRB
    - System result gating: only activate physically-related module subset
    - Knowledge mapping: modules use relevant frequency bands/features only
    - Sub-BRB architecture: separate BRBs for amp/freq/ref faults
    - Normal anchor: Two-stage classification (Normal vs Fault first, then fault type)
    
    Enhanced with X1-X22 features for improved accuracy while maintaining framework.
    
    Complexity:
    - Rules: system layer (3 sub-BRBs) + module layer (configured rules only)
    - Params: attribute weights + rule weights + belief degrees (增加扩展特征权重)
    """
    
    name = "ours"
    
    # Fallback probability distributions when BRB output is invalid
    # These are used when sum of probabilities is too low to normalize
    FALLBACK_NORMAL_PROBS = np.array([0.7, 0.1, 0.1, 0.1])  # High normal, low fault
    FALLBACK_FAULT_PROBS = np.array([0.1, 0.3, 0.3, 0.3])   # Low normal, uniform fault
    
    # Normal anchor thresholds - calibrated for better class separation
    NORMAL_ANCHOR_SCORE_THRESHOLD = 0.12  # Below this = definitely normal
    FAULT_CONFIRMATION_THRESHOLD = 0.25   # Above this = definitely fault
    
    def __init__(self):
        self.config = SystemBRBConfig(
            alpha=3.0,  # Higher temperature for sharper distribution
            overall_threshold=0.12,  # Lower threshold for better normal detection
            max_prob_threshold=0.25,  # Lower threshold for better fault confirmation
        )
        self.feature_names = None
        self.n_system_rules = 15  # 3 sub-BRBs with 5 rules each
        self.n_module_rules = 33  # Configured per-module rules
        self.n_params = 68  # 增加：22个特征权重 + 3个规则权重 + 33个belief参数 + 10个子BRB参数
        self.kd_features = [f'X{i}' for i in range(1, 23)]  # X1-X22
        self.use_sub_brb = True  # 启用子BRB架构以提高准确率
        self.use_normal_anchor = True  # 启用Normal锚点两阶段判定
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
        """Predict on test data using hierarchical BRB with extended features and sub-BRB architecture.
        
        Uses a two-stage Normal Anchor approach:
        1. First determine if sample is Normal vs Fault based on overall anomaly score
        2. If Fault, use sub-BRB architecture to classify among Amp/Freq/Ref
        """
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
        
        # 选择推理模式：使用sub_brb架构以提高准确率
        inference_mode = 'sub_brb' if self.use_sub_brb else 'er'
        
        # B2 FIX: Unified probability key mapping (Chinese -> English index)
        # Order must match: 正常(0), 幅度失准(1), 频率失准(2), 参考电平失准(3)
        prob_key_map = {
            '正常': 0, 'Normal': 0, 'normal': 0,
            '幅度失准': 1, 'Amp': 1, 'amp': 1, 'amp_error': 1,
            '频率失准': 2, 'Freq': 2, 'freq': 2, 'freq_error': 2,
            '参考电平失准': 3, 'Ref': 3, 'ref': 3, 'ref_error': 3,
        }
        
        for i in range(n_test):
            # Convert sample to feature dict (支持X1-X22)
            features = self._array_to_dict(X_test[i])
            
            # ==================== Normal Anchor Stage ====================
            # Two-stage classification: First determine Normal vs Fault
            if self.use_normal_anchor:
                # Compute overall anomaly score for normal anchor
                overall_score = self._compute_overall_anomaly_score(features)
                
                # Stage 1: Normal anchor check
                if overall_score < self.NORMAL_ANCHOR_SCORE_THRESHOLD:
                    # Low anomaly score -> classify as Normal
                    sys_proba[i] = np.array([0.8, 0.067, 0.067, 0.066])
                    sys_pred[i] = 0  # Normal
                    continue
            
            # ==================== Fault Classification Stage ====================
            # System-level inference - 使用sub_brb模式
            sys_result = system_level_infer(features, self.config, mode=inference_mode)
            probs = sys_result.get('probabilities', {})
            
            # B2 FIX: Map probability keys to indices with unified mapping
            # Ensure all 4 classes are present, default to 0.0 if missing
            for key, value in probs.items():
                if key in prob_key_map:
                    idx = prob_key_map[key]
                    sys_proba[i, idx] = float(value)
            
            # ==================== Fault Type Balancing ====================
            # Apply class-specific calibration to prevent Amp dominance
            # Reduce Amp probability if it's dominating
            if sys_proba[i, 1] > 0.5 and sys_proba[i, 1] > sys_proba[i, 2] + sys_proba[i, 3]:
                # Check if Freq or Ref features are significant
                freq_score = self._compute_freq_specific_score(features)
                ref_score = self._compute_ref_specific_score(features)
                
                # Redistribute probability if other classes have significant features
                if freq_score > 0.3:
                    # Boost Freq probability
                    boost = min(0.2, freq_score * 0.3)
                    sys_proba[i, 2] += boost
                    sys_proba[i, 1] -= boost * 0.7
                    
                if ref_score > 0.3:
                    # Boost Ref probability
                    boost = min(0.2, ref_score * 0.3)
                    sys_proba[i, 3] += boost
                    sys_proba[i, 1] -= boost * 0.7
            
            # Normalize if sum > 0, otherwise use fallback priors
            row_sum = np.sum(sys_proba[i])
            if row_sum > 0.01:
                sys_proba[i] /= row_sum
            else:
                # Fallback: use prior based on overall_score
                overall_score = sys_result.get('overall_score', 0.5)
                if overall_score < 0.15:
                    # Low anomaly: likely normal
                    sys_proba[i] = self.FALLBACK_NORMAL_PROBS.copy()
                else:
                    # High anomaly: uniform over fault classes
                    sys_proba[i] = self.FALLBACK_FAULT_PROBS.copy()
            
            sys_pred[i] = np.argmax(sys_proba[i])
            
            # Module-level inference - 使用module_level_infer_with_activation以激活相关模块组
            mod_probs_dict = module_level_infer_with_activation(features, sys_result, only_activate_relevant=True)
            
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
    
    def _normalize_feature(self, value: float, lower: float, upper: float) -> float:
        """Normalize feature value to [0, 1] range."""
        value = max(lower, min(value, upper))
        return (value - lower) / (upper - lower + 1e-12)
    
    def _compute_overall_anomaly_score(self, features: Dict[str, float]) -> float:
        """Compute overall anomaly score for Normal anchor detection.
        
        Uses weighted combination of key anomaly indicators.
        Low score (< 0.12) indicates normal state.
        """
        # Feature normalization parameters - 调整范围以匹配实际数据分布
        norm_params = {
            'X1': (-15, -5),      # amplitude offset (raw mean dB)
            'X2': (0.0, 0.2),     # inband flatness
            'X4': (0.0, 0.5),     # frequency nonlinearity (std)
            'X5': (0.1, 0.7),     # scale consistency
            'X11': (0.0, 0.1),    # envelope overrun rate
            'X12': (0.0, 2.0),    # max envelope violation
            'X13': (0.0, 20.0),   # envelope violation energy
        }
        
        # Weights for overall anomaly score
        weights = {
            'X1': 0.15, 'X2': 0.15, 'X4': 0.10, 'X5': 0.10,
            'X11': 0.20, 'X12': 0.15, 'X13': 0.15,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = float(features.get(key, 0.0))
                lower, upper = norm_params.get(key, (0.0, 1.0))
                norm_value = self._normalize_feature(raw_value, lower, upper)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _compute_freq_specific_score(self, features: Dict[str, float]) -> float:
        """Compute frequency-specific anomaly score.
        
        Features unique to frequency faults: X4, X16, X17, X18.
        """
        norm_params = {
            'X4': (0.0, 0.5),     # frequency nonlinearity (std)
            'X16': (0.0, 0.01),   # correlation shift
            'X17': (0.0, 0.01),   # warp scale
            'X18': (0.0, 0.01),   # warp bias
        }
        
        weights = {'X4': 0.50, 'X16': 0.20, 'X17': 0.15, 'X18': 0.15}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = abs(float(features.get(key, 0.0)))
                lower, upper = norm_params.get(key, (0.0, 1.0))
                norm_value = self._normalize_feature(raw_value, lower, upper)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _compute_ref_specific_score(self, features: Dict[str, float]) -> float:
        """Compute reference-level-specific anomaly score.
        
        Features unique to reference level faults: X3, X10.
        X3: High-frequency attenuation slope (reference level affects this)
        X10: Band amplitude consistency (reference level mismatch between bands)
        """
        norm_params = {
            'X3': (-0.005, 0.005),  # HF attenuation slope
            'X10': (0.0, 0.2),      # band amplitude consistency
        }
        
        weights = {'X3': 0.50, 'X10': 0.50}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = abs(float(features.get(key, 0.0)))
                lower, upper = norm_params.get(key, (0.0, 1.0))
                norm_value = self._normalize_feature(raw_value, lower, upper)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
