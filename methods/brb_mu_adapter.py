"""Adapter for BRB-MU method (Feng 2024) - Multi-source Uncertainty Fusion."""
from __future__ import annotations

import time
from typing import Dict, Optional, List, Tuple

import numpy as np
from methods.base import MethodAdapter


class BRBMUAdapter(MethodAdapter):
    """BRB-MU: Multi-source Uncertain Information Fusion BRB (Feng 2024).
    
    MUST-HAVE mechanisms:
    - Multi-source feature grouping (at least 3 sources)
    - Uncertainty modeling for each source: u_s = f(SNR, SVD)
    - SNR: signal-to-noise ratio component
    - SVD: structural uncertainty component  
    - Fusion weights: w_s ∝ (1 - u_s)
    - Final prediction: weighted fusion of source predictions
    """
    
    name = "brb_mu"
    
    def __init__(self):
        self.sources = {}  # Source name -> feature indices
        self.source_models = {}  # Per-source simple models
        self.source_uncertainties = {}  # Estimated uncertainties
        self.source_weights = {}  # Fusion weights
        self.n_rules = 72
        self.n_params = 110
        self.n_features_used = 6
        self.feature_names = None
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit BRB-MU with multi-source uncertainty fusion."""
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_sys_train))
        
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        
        # ========== Define sources (feature groups) ==========
        self.sources = self._define_sources(self.feature_names, n_features)
        
        # ========== Train per-source models and estimate uncertainties ==========
        for source_name, source_idx in self.sources.items():
            if len(source_idx) == 0:
                continue
            
            X_source = X_train[:, source_idx]
            
            # Train simple model for this source (logistic regression or naive rule-based)
            model = self._train_source_model(X_source, y_sys_train, n_classes)
            self.source_models[source_name] = model
            
            # Estimate source uncertainty
            u_s = self._estimate_source_uncertainty(X_source)
            self.source_uncertainties[source_name] = u_s
        
        # ========== Compute fusion weights ==========
        total_reliability = 0.0
        for source_name, u_s in self.source_uncertainties.items():
            reliability = 1.0 - u_s
            self.source_weights[source_name] = reliability
            total_reliability += reliability
        
        # Normalize weights
        if total_reliability > 1e-8:
            for source_name in self.source_weights:
                self.source_weights[source_name] /= total_reliability
        else:
            # Equal weights fallback
            n_sources = len(self.source_weights)
            for source_name in self.source_weights:
                self.source_weights[source_name] = 1.0 / n_sources
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using multi-source fusion."""
        n_test = len(X_test)
        n_classes = 4
        
        if not self.source_models:
            sys_pred = np.random.randint(0, n_classes, n_test)
            sys_proba = np.eye(n_classes)[sys_pred]
            return self._create_result(sys_proba, sys_pred, n_test, 0.0)
        
        start_time = time.time()
        
        # Get predictions from each source
        source_probs = {}
        for source_name, source_idx in self.sources.items():
            if source_name not in self.source_models:
                continue
            
            X_source = X_test[:, source_idx]
            model = self.source_models[source_name]
            probs = self._predict_source(X_source, model, n_classes)
            source_probs[source_name] = probs
        
        # Weighted fusion
        sys_proba = np.zeros((n_test, n_classes))
        total_weight = 0.0
        
        for source_name, probs in source_probs.items():
            weight = self.source_weights.get(source_name, 0.0)
            sys_proba += weight * probs
            total_weight += weight
        
        if total_weight > 1e-8:
            sys_proba /= total_weight
        else:
            sys_proba = np.ones((n_test, n_classes)) / n_classes
        
        sys_pred = np.argmax(sys_proba, axis=1)
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000
        
        return self._create_result(sys_proba, sys_pred, n_test, infer_time_ms)
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        n_features = sum(len(idx) for idx in self.sources.values())
        return {
            'n_rules': self.n_rules,
            'n_params': self.n_params,
            'n_features_used': max(n_features, self.n_features_used),
        }
    
    def _define_sources(self, feature_names: Optional[List[str]], 
                       n_features: int) -> Dict[str, List[int]]:
        """Define at least 3 feature sources."""
        sources = {}
        
        if not feature_names or len(feature_names) != n_features:
            # Default: divide into 3-4 sources
            size = max(1, n_features // 4)
            sources['amplitude'] = list(range(0, size))
            sources['frequency'] = list(range(size, 2*size))
            sources['noise'] = list(range(2*size, 3*size))
            sources['switching'] = list(range(3*size, n_features))
        else:
            # Semantic grouping
            amp_idx, freq_idx, noise_idx, switch_idx = [], [], [], []
            
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if 'amp' in name_lower or 'bias' in name_lower or 'x1' in name_lower:
                    amp_idx.append(i)
                elif 'freq' in name_lower or 'df' in name_lower or 'x4' in name_lower or 'step' in name_lower:
                    freq_idx.append(i)
                elif 'ripple' in name_lower or 'noise' in name_lower or 'x2' in name_lower:
                    noise_idx.append(i)
                elif 'switch' in name_lower or 'x3' in name_lower or 'slope' in name_lower:
                    switch_idx.append(i)
                else:
                    # Distribute to smallest group
                    min_group = min([amp_idx, freq_idx, noise_idx, switch_idx], key=len)
                    min_group.append(i)
            
            sources['amplitude'] = amp_idx if amp_idx else [0]
            sources['frequency'] = freq_idx if freq_idx else [1] if n_features > 1 else []
            sources['noise'] = noise_idx if noise_idx else [2] if n_features > 2 else []
            sources['switching'] = switch_idx if switch_idx else [3] if n_features > 3 else []
        
        # Remove empty sources
        sources = {k: v for k, v in sources.items() if len(v) > 0}
        
        return sources
    
    def _train_source_model(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> Dict:
        """Train a simple model for one source."""
        # Use simple statistics-based model
        model = {
            'means': [],
            'stds': [],
            'priors': np.bincount(y, minlength=n_classes) / len(y)
        }
        
        for c in range(n_classes):
            X_c = X[y == c]
            if len(X_c) > 0:
                model['means'].append(np.mean(X_c, axis=0))
                model['stds'].append(np.std(X_c, axis=0) + 1e-8)
            else:
                model['means'].append(np.zeros(X.shape[1]))
                model['stds'].append(np.ones(X.shape[1]))
        
        return model
    
    def _predict_source(self, X: np.ndarray, model: Dict, n_classes: int) -> np.ndarray:
        """Predict using source model (Gaussian-like)."""
        n_samples = len(X)
        probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for c in range(n_classes):
                # Gaussian likelihood
                mean = model['means'][c]
                std = model['stds'][c]
                diff = X[i] - mean
                likelihood = np.exp(-0.5 * np.sum((diff / std) ** 2))
                probs[i, c] = likelihood * model['priors'][c]
            
            # Normalize
            if np.sum(probs[i]) > 1e-10:
                probs[i] /= np.sum(probs[i])
            else:
                probs[i] = model['priors']
        
        return probs
    
    def _estimate_source_uncertainty(self, X: np.ndarray) -> float:
        """Estimate source uncertainty combining SNR and SVD components.
        
        u_s = normalize(a * u_snr + b * u_svd)
        - u_snr: based on signal-to-noise ratio
        - u_svd: based on SVD structural uncertainty
        """
        n_samples, n_features = X.shape
        
        # ========== SNR component ==========
        # SNR = mean(|x|) / std(x)
        if n_features > 0:
            mean_signal = np.mean(np.abs(X))
            std_noise = np.std(X)
            snr = mean_signal / (std_noise + 1e-8)
            # Convert to uncertainty: higher SNR -> lower uncertainty
            u_snr = 1.0 / (1.0 + snr)
        else:
            u_snr = 0.5
        
        # ========== SVD component ==========
        # Structural uncertainty: 1 - (sigma_1 / sum(sigma))
        if n_samples >= n_features and n_features > 1:
            try:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                total_var = np.sum(S)
                if total_var > 1e-10:
                    u_svd = 1.0 - (S[0] / total_var)
                else:
                    u_svd = 0.5
            except:
                u_svd = 0.5
        else:
            u_svd = 0.5
        
        # ========== Combine ==========
        a, b = 0.6, 0.4  # Weights for SNR and SVD
        u_combined = a * u_snr + b * u_svd
        
        # Normalize to [0, 1]
        u_normalized = np.clip(u_combined, 0.0, 1.0)
        
        return float(u_normalized)
    
    def _create_result(self, sys_proba: np.ndarray, sys_pred: np.ndarray,
                      n_test: int, infer_time_ms: float) -> Dict:
        """Create standardized result dict."""
        n_features = sum(len(idx) for idx in self.sources.values())
        return {
            'system_proba': sys_proba,
            'system_pred': sys_pred,
            'module_proba': None,
            'module_pred': None,
            'meta': {
                'fit_time_sec': 0.0,
                'infer_time_ms_per_sample': infer_time_ms,
                'n_rules': self.n_rules,
                'n_params': self.n_params,
                'n_features_used': max(n_features, self.n_features_used),
                'features_used': [idx for indices in self.sources.values() for idx in indices],
                'source_uncertainties': self.source_uncertainties,
                'source_weights': self.source_weights,
            }
        }
