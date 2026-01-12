"""Adapter for Ours method (knowledge-driven rule compression + hierarchical BRB)."""
from __future__ import annotations

import json
import time
from pathlib import Path
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
    - Branch evidence gating: freq/ref specific gates to boost logits
    
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
    
    # Normal anchor probability distribution (used when sample is classified as Normal)
    NORMAL_ANCHOR_PROBS = np.array([0.8, 0.067, 0.067, 0.066])
    
    # Default thresholds - will be overridden by calibration file if available
    # Step1 requirement: Normal anchor must use auto-calibrated thresholds from normal stats
    DEFAULT_NORMAL_ANCHOR_THRESHOLD = 0.12  # Below this = definitely normal
    DEFAULT_FAULT_CONFIRMATION_THRESHOLD = 0.25   # Above this = definitely fault
    
    # Branch evidence gating defaults - Step2 requirement
    # Boosts for freq/ref branches when their specific evidence exceeds gate
    DEFAULT_FREQ_GATE = 0.3   # Threshold for X16/X17/X18 z-score to trigger boost
    DEFAULT_REF_GATE = 0.3    # Threshold for X3/X10 z-score to trigger boost
    DEFAULT_FREQ_BOOST = 1.0  # Logit boost amount for freq branch
    DEFAULT_REF_BOOST = 1.0   # Logit boost amount for ref branch
    
    # Class balancing parameters
    AMP_DOMINANCE_THRESHOLD = 0.5  # Threshold to detect Amp dominance
    MAX_BOOST_AMOUNT = 0.2         # Maximum probability boost
    BOOST_SCALE_FACTOR = 0.3      # Scale factor for boost calculation
    AMP_REDUCTION_RATIO = 0.7     # Ratio to reduce Amp probability during boost
    
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
    # 配置开关：Normal锚点和证据门控默认禁用，等待校准文件启用
        # 这些可以通过calibration.json配置：{"use_normal_anchor": true, "use_evidence_gating": true}
        self.use_sub_brb = True  # 启用子BRB架构以提高准确率
        self.use_normal_anchor = False  # 默认禁用，通过calibration.json启用
        self.use_evidence_gating = False  # 默认禁用，通过calibration.json启用
        
        # Calibrated thresholds - will be loaded from calibration.json if available
        self.normal_anchor_threshold = self.DEFAULT_NORMAL_ANCHOR_THRESHOLD
        self.fault_confirmation_threshold = self.DEFAULT_FAULT_CONFIRMATION_THRESHOLD
        self.freq_gate = self.DEFAULT_FREQ_GATE
        self.ref_gate = self.DEFAULT_REF_GATE
        self.freq_boost = self.DEFAULT_FREQ_BOOST
        self.ref_boost = self.DEFAULT_REF_BOOST
        
        # Normal feature statistics - loaded from baseline or calibration
        self.normal_feature_stats = None
        
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
        
        # Try to load calibration from file
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibrated thresholds from calibration.json if available.
        
        Step1 requirement: T_normal must read from statistics file (auto-calibrated),
        not hard-coded constants.
        """
        # Try multiple possible locations
        calibration_paths = [
            Path('Output/calibration.json'),
            Path('Output/sim_spectrum/calibration.json'),
            Path(__file__).parent.parent / 'Output' / 'calibration.json',
        ]
        
        for calib_path in calibration_paths:
            if calib_path.exists():
                try:
                    with open(calib_path, 'r', encoding='utf-8') as f:
                        calib = json.load(f)
                    
                    # Load calibrated thresholds
                    if 'use_normal_anchor' in calib:
                        self.use_normal_anchor = bool(calib['use_normal_anchor'])
                    if 'use_evidence_gating' in calib:
                        self.use_evidence_gating = bool(calib['use_evidence_gating'])
                    if 'normal_anchor_threshold' in calib:
                        self.normal_anchor_threshold = float(calib['normal_anchor_threshold'])
                    if 'fault_confirmation_threshold' in calib:
                        self.fault_confirmation_threshold = float(calib['fault_confirmation_threshold'])
                    if 'freq_gate' in calib:
                        self.freq_gate = float(calib['freq_gate'])
                    if 'ref_gate' in calib:
                        self.ref_gate = float(calib['ref_gate'])
                    if 'freq_boost' in calib:
                        self.freq_boost = float(calib['freq_boost'])
                    if 'ref_boost' in calib:
                        self.ref_boost = float(calib['ref_boost'])
                    if 'alpha' in calib:
                        self.config.alpha = float(calib['alpha'])
                    if 'overall_threshold' in calib:
                        val = float(calib['overall_threshold'])
                        if 0.0 <= val <= 1.0:
                            self.config.overall_threshold = val
                    if 'max_prob_threshold' in calib:
                        val = float(calib['max_prob_threshold'])
                        if 0.0 <= val <= 1.0:
                            self.config.max_prob_threshold = val
                    
                    print(f"[OursAdapter] Loaded calibration from {calib_path}")
                    break
                except Exception as e:
                    print(f"[OursAdapter] Warning: Failed to load calibration from {calib_path}: {e}")
        
        # Try to load normal feature stats for percentile-based thresholds
        stats_paths = [
            Path('Output/normal_feature_stats.csv'),
            Path('Output/sim_spectrum/normal_feature_stats.csv'),
            Path(__file__).parent.parent / 'Output' / 'normal_feature_stats.csv',
        ]
        
        for stats_path in stats_paths:
            if stats_path.exists():
                try:
                    import pandas as pd
                    stats_df = pd.read_csv(stats_path, index_col=0)
                    self.normal_feature_stats = {}
                    
                    # Extract median and IQR for each feature
                    for col in stats_df.columns:
                        if col.startswith('X') and col[1:].isdigit():
                            median = stats_df.loc['50%', col] if '50%' in stats_df.index else 0.0
                            q25 = stats_df.loc['25%', col] if '25%' in stats_df.index else 0.0
                            q75 = stats_df.loc['75%', col] if '75%' in stats_df.index else 0.0
                            iqr = q75 - q25
                            # Use std as fallback when IQR is 0
                            if iqr < 1e-9:
                                std = stats_df.loc['std', col] if 'std' in stats_df.index else 0.0
                                iqr = float(std) * 1.35  # Approx IQR for normal distribution
                            # Ensure IQR is never zero to avoid division by zero
                            if iqr < 1e-9:
                                iqr = 0.1  # Use default small value
                            p95 = stats_df.loc['95%', col] if '95%' in stats_df.index else 0.0
                            p97 = stats_df.loc['97%', col] if '97%' in stats_df.index else p95
                            self.normal_feature_stats[col] = {
                                'median': float(median),
                                'iqr': float(iqr),
                                'p95': float(p95),
                                'p97': float(p97),
                            }
                    
                    print(f"[OursAdapter] Loaded normal feature stats from {stats_path}")
                    break
                except Exception as e:
                    print(f"[OursAdapter] Warning: Failed to load normal feature stats from {stats_path}: {e}")
    
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
        
        Step2: Branch evidence gating - boost freq/ref logits when specific evidence is present
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
            
            # ==================== Normal Anchor Stage (Step1) ====================
            # Two-stage classification: First determine Normal vs Fault
            if self.use_normal_anchor:
                # Compute overall anomaly score for normal anchor
                overall_score = self._compute_overall_anomaly_score(features)
                
                # Stage 1: Normal anchor check - use calibrated threshold
                if overall_score < self.normal_anchor_threshold:
                    # Low anomaly score -> classify as Normal
                    sys_proba[i] = self.NORMAL_ANCHOR_PROBS.copy()
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
            
            # ==================== Branch Evidence Gating (Step2) ====================
            # Boost freq/ref branches when their specific evidence exceeds gate
            if self.use_evidence_gating:
                freq_z_score = self._compute_freq_zscore(features)
                ref_z_score = self._compute_ref_zscore(features)
                
                # Apply logit-space boost for freq branch
                if freq_z_score > self.freq_gate:
                    # Convert probs to logits, boost, convert back
                    logits = np.log(sys_proba[i] + 1e-12)
                    logits[2] += self.freq_boost * (freq_z_score - self.freq_gate)
                    # Convert back to probs
                    exp_logits = np.exp(logits - np.max(logits))
                    sys_proba[i] = exp_logits / (np.sum(exp_logits) + 1e-12)
                
                # Apply logit-space boost for ref branch
                if ref_z_score > self.ref_gate:
                    logits = np.log(sys_proba[i] + 1e-12)
                    logits[3] += self.ref_boost * (ref_z_score - self.ref_gate)
                    exp_logits = np.exp(logits - np.max(logits))
                    sys_proba[i] = exp_logits / (np.sum(exp_logits) + 1e-12)
            
            # ==================== Fault Type Balancing ====================
            # Apply class-specific calibration to prevent Amp dominance
            # Reduce Amp probability if it's dominating
            if sys_proba[i, 1] > self.AMP_DOMINANCE_THRESHOLD and sys_proba[i, 1] > sys_proba[i, 2] + sys_proba[i, 3]:
                # Check if Freq or Ref features are significant
                freq_score = self._compute_freq_specific_score(features)
                ref_score = self._compute_ref_specific_score(features)
                
                # Redistribute probability if other classes have significant features
                if freq_score > 0.3:
                    # Boost Freq probability
                    boost = min(self.MAX_BOOST_AMOUNT, freq_score * self.BOOST_SCALE_FACTOR)
                    sys_proba[i, 2] += boost
                    sys_proba[i, 1] -= boost * self.AMP_REDUCTION_RATIO
                    
                if ref_score > 0.3:
                    # Boost Ref probability
                    boost = min(self.MAX_BOOST_AMOUNT, ref_score * self.BOOST_SCALE_FACTOR)
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
    
    def _zscore_normalize(self, value: float, median: float, iqr: float) -> float:
        """使用robust z-score归一化特征值，返回绝对值用于异常检测。
        
        Step2要求：归一化只允许用 normal_feature_stats（robust z-score）。
        z = (x - median) / (IQR + eps)
        """
        if iqr < 1e-6:
            iqr = 1e-6
        z = (value - median) / (iqr + 1e-6)
        return min(1.0, abs(z) / 3.0)  # |z|=3 对应 score=1.0
    
    # 默认正常特征统计：(median, iqr)
    # 如果加载了calibration文件则会被覆盖
    DEFAULT_NORMAL_FEATURE_STATS = {
        "X1": (0.003152, 0.067323),    # 整体幅度偏移（residual均值）
        "X2": (0.000094, 0.000115),    # 带内平坦度（residual方差）
        "X3": (-0.000018, 0.000228),   # 高频衰减斜率
        "X4": (0.010546, 0.005289),    # 频率标度非线性
        "X5": (0.255391, 0.101644),    # 幅度缩放一致性
        "X11": (0.003659, 0.001220),   # 包络越界率
        "X12": (0.030000, 0.010000),   # 最大包络违规
        "X13": (6.394475, 23.550748),  # 包络违规能量
        "X16": (0.000000, 0.001000),   # 频移
        "X17": (0.000000, 0.001000),   # 频率缩放
        "X18": (0.000000, 0.001000),   # 频率平移
    }
    
    def _get_feature_stats(self, feature_name: str) -> tuple:
        """Get median and IQR for a feature, using loaded stats or defaults."""
        if self.normal_feature_stats and feature_name in self.normal_feature_stats:
            stats = self.normal_feature_stats[feature_name]
            return (stats['median'], stats['iqr'])
        return self.DEFAULT_NORMAL_FEATURE_STATS.get(feature_name, (0.0, 0.1))
    
    def _compute_overall_anomaly_score(self, features: Dict[str, float]) -> float:
        """Compute overall anomaly score for Normal anchor detection.
        
        Step3要求：Normal两阶段判决必须用"证据阈值"而不是"softmax最大概率"。
        使用z-score归一化，低分（< normal_anchor_threshold）表示正常状态。
        """
        # Weights for overall anomaly score
        # 重点使用envelope相关特征和residual-based特征
        weights = {
            'X1': 0.15,  # 幅度偏移（residual）
            'X2': 0.10,  # 平坦度（residual方差）
            'X4': 0.10,  # 频率非线性
            'X5': 0.05,  # 缩放一致性
            'X11': 0.25, # 包络越界率（关键）
            'X12': 0.20, # 最大包络违规（关键）
            'X13': 0.15, # 包络违规能量
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = float(features.get(key, 0.0))
                median, iqr = self._get_feature_stats(key)
                norm_value = self._zscore_normalize(raw_value, median, iqr)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _compute_freq_zscore(self, features: Dict[str, float]) -> float:
        """Compute aggregate z-score for frequency-specific features.
        
        Step2 requirement: freq gate based on X16/X17/X18.
        """
        z_scores = []
        for key in ['X16', 'X17', 'X18']:
            if key in features:
                raw_value = float(features.get(key, 0.0))
                median, iqr = self._get_feature_stats(key)
                if iqr < 1e-6:
                    iqr = 1e-6
                z = abs(raw_value - median) / (iqr + 1e-6)
                z_scores.append(z)
        
        return max(z_scores) if z_scores else 0.0
    
    def _compute_ref_zscore(self, features: Dict[str, float]) -> float:
        """Compute aggregate z-score for reference-level-specific features.
        
        Step2 requirement: ref gate based on X3/X10.
        """
        z_scores = []
        for key in ['X3', 'X10']:
            if key in features:
                raw_value = float(features.get(key, 0.0))
                median, iqr = self._get_feature_stats(key)
                if iqr < 1e-6:
                    iqr = 1e-6
                z = abs(raw_value - median) / (iqr + 1e-6)
                z_scores.append(z)
        
        return max(z_scores) if z_scores else 0.0
    
    def _compute_freq_specific_score(self, features: Dict[str, float]) -> float:
        """Compute frequency-specific anomaly score.
        
        Features unique to frequency faults: X4, X16, X17, X18.
        """
        weights = {'X4': 0.40, 'X16': 0.25, 'X17': 0.20, 'X18': 0.15}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = float(features.get(key, 0.0))
                median, iqr = self._get_feature_stats(key)
                norm_value = self._zscore_normalize(raw_value, median, iqr)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _compute_ref_specific_score(self, features: Dict[str, float]) -> float:
        """Compute reference-level-specific anomaly score.
        
        Features unique to reference level faults: X3, X10.
        """
        # X3: High-frequency attenuation slope
        # X10: Band amplitude consistency
        stats = {
            'X3': (0.0, 0.001),
            'X10': (0.02, 0.02),
        }
        
        weights = {'X3': 0.50, 'X10': 0.50}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in features:
                raw_value = float(features.get(key, 0.0))
                median, iqr = self._get_feature_stats(key)
                norm_value = self._zscore_normalize(raw_value, median, iqr)
                weighted_sum += weight * norm_value
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
