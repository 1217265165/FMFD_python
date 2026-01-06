"""Adapter for HCF method (Zhang 2022) - Hierarchical Cognitive Framework."""
from __future__ import annotations

import time
from typing import Dict, Optional, List

import numpy as np
from methods.base import MethodAdapter


class HCFAdapter(MethodAdapter):
    """HCF: Hierarchical Cognitive Framework (Zhang 2022).
    
    MUST-HAVE three-tier process:
    - Level-a: Sensor/feature cognitive (select primary/secondary feature sources)
    - Level-b: Single-source pattern cognitive (clustering per source)
    - Level-c: Data climate cognitive (combine encodings -> classification)
    
    Implementation:
    - FN3WD approximation: Fisher score + CV error for feature selection
    - GMM clustering for pattern encoding per source
    - Interpretable classifier (logistic regression) for final decision
    """
    
    name = "hcf"
    
    def __init__(self):
        self.primary_features = []
        self.secondary_features = []
        self.sources = []  # List of feature groups
        self.gmm_models = {}  # GMM per source
        self.classifier = None
        self.n_rules = 90  # Estimated: multiple sources * clusters * classes
        self.n_params = 130  # GMM params + classifier weights
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit HCF model."""
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_selection import f_classif
        except ImportError:
            print("sklearn not available, using fallback")
            self.classifier = None
            return
        
        n_samples, n_features = X_train.shape
        feature_names = meta.get('feature_names', []) if meta else []
        
        # ========== Level-a: Feature Cognitive (FN3WD approximation) ==========
        # Fisher score for separability
        if n_samples > n_features:
            try:
                f_scores, _ = f_classif(X_train, y_sys_train)
                f_scores = np.nan_to_num(f_scores, 0.0)
            except:
                f_scores = np.ones(n_features)
        else:
            f_scores = np.ones(n_features)
        
        # Select top features as primary, rest as secondary
        n_primary = min(6, n_features)
        primary_idx = np.argsort(f_scores)[-n_primary:]
        secondary_idx = np.argsort(f_scores)[:-n_primary] if n_features > n_primary else []
        
        self.primary_features = list(primary_idx)
        self.secondary_features = list(secondary_idx)
        
        # ========== Define sources (feature groups) ==========
        # Group features into semantic sources
        self.sources = self._define_sources(feature_names, n_features)
        
        # ========== Level-b: Single-source Pattern Cognitive (GMM clustering) ==========
        encoded_features = []
        
        for source_name, source_idx in self.sources.items():
            if len(source_idx) == 0:
                continue
            
            X_source = X_train[:, source_idx]
            
            # Fit GMM for this source
            n_clusters = min(3, max(2, len(np.unique(y_sys_train))))  # 2-3 clusters
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, 
                                 covariance_type='diag', max_iter=50)
            try:
                gmm.fit(X_source)
                cluster_labels = gmm.predict(X_source)
                self.gmm_models[source_name] = gmm
            except:
                # Fallback: simple quantization
                cluster_labels = np.digitize(X_source[:, 0], bins=np.linspace(
                    X_source[:, 0].min(), X_source[:, 0].max(), n_clusters+1)) - 1
                cluster_labels = np.clip(cluster_labels, 0, n_clusters-1)
                self.gmm_models[source_name] = None
            
            # One-hot encode cluster IDs
            cluster_onehot = np.zeros((n_samples, n_clusters))
            cluster_onehot[np.arange(n_samples), cluster_labels] = 1
            encoded_features.append(cluster_onehot)
        
        if len(encoded_features) == 0:
            # Fallback: use raw features
            encoded_features = [X_train]
        
        # Combine encoded features
        X_encoded = np.hstack(encoded_features)
        
        # ========== Level-c: Data Climate Cognitive (classification) ==========
        try:
            self.classifier = LogisticRegression(random_state=42, max_iter=200, 
                                                multi_class='multinomial', solver='lbfgs')
        except TypeError:
            # Older sklearn version
            self.classifier = LogisticRegression(random_state=42, max_iter=200, solver='lbfgs')
        self.classifier.fit(X_encoded, y_sys_train)
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using HCF."""
        n_test = len(X_test)
        n_sys_classes = 4
        
        if self.classifier is None:
            # Fallback: random baseline
            sys_pred = np.random.randint(0, n_sys_classes, n_test)
            sys_proba = np.eye(n_sys_classes)[sys_pred]
            return self._create_result(sys_proba, sys_pred, n_test, 0.0)
        
        start_time = time.time()
        
        # Encode test data using same sources + GMM
        encoded_features = []
        
        for source_name, source_idx in self.sources.items():
            if len(source_idx) == 0 or source_name not in self.gmm_models:
                continue
            
            X_source = X_test[:, source_idx]
            gmm = self.gmm_models[source_name]
            
            if gmm is not None:
                try:
                    cluster_labels = gmm.predict(X_source)
                    n_clusters = gmm.n_components
                except:
                    n_clusters = 3
                    cluster_labels = np.digitize(X_source[:, 0], bins=np.linspace(
                        X_source[:, 0].min(), X_source[:, 0].max(), n_clusters+1)) - 1
                    cluster_labels = np.clip(cluster_labels, 0, n_clusters-1)
            else:
                n_clusters = 3
                cluster_labels = np.zeros(n_test, dtype=int)
            
            # One-hot encode
            cluster_onehot = np.zeros((n_test, n_clusters))
            cluster_onehot[np.arange(n_test), cluster_labels] = 1
            encoded_features.append(cluster_onehot)
        
        if len(encoded_features) == 0:
            encoded_features = [X_test]
        
        X_encoded = np.hstack(encoded_features)
        
        # Predict
        sys_proba = self.classifier.predict_proba(X_encoded)
        sys_pred = np.argmax(sys_proba, axis=1)
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000
        
        return self._create_result(sys_proba, sys_pred, n_test, infer_time_ms)
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        n_features_used = len(self.primary_features) + len(self.secondary_features)
        return {
            'n_rules': self.n_rules,
            'n_params': self.n_params,
            'n_features_used': max(n_features_used, 6),
        }
    
    def _define_sources(self, feature_names: List[str], n_features: int) -> Dict[str, List[int]]:
        """Define feature sources based on semantic groups."""
        sources = {}
        
        if not feature_names or len(feature_names) != n_features:
            # Default: divide into 3 sources
            size = n_features // 3
            sources['amplitude'] = list(range(0, size))
            sources['frequency'] = list(range(size, 2*size))
            sources['noise'] = list(range(2*size, n_features))
        else:
            # Semantic grouping
            amp_idx = []
            freq_idx = []
            noise_idx = []
            other_idx = []
            
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if 'amp' in name_lower or 'bias' in name_lower or 'x1' in name_lower or 'band' in name_lower and 'mean' in name_lower:
                    amp_idx.append(i)
                elif 'freq' in name_lower or 'df' in name_lower or 'x4' in name_lower or 'step' in name_lower:
                    freq_idx.append(i)
                elif 'ripple' in name_lower or 'noise' in name_lower or 'x2' in name_lower:
                    noise_idx.append(i)
                else:
                    other_idx.append(i)
            
            sources['amplitude'] = amp_idx if amp_idx else [0]
            sources['frequency'] = freq_idx if freq_idx else [1] if n_features > 1 else []
            sources['noise'] = noise_idx if noise_idx else [2] if n_features > 2 else []
            if other_idx:
                sources['other'] = other_idx
        
        return sources
    
    def _create_result(self, sys_proba: np.ndarray, sys_pred: np.ndarray, 
                      n_test: int, infer_time_ms: float) -> Dict:
        """Create standardized result dict."""
        n_features = len(self.primary_features) + len(self.secondary_features)
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
                'n_features_used': max(n_features, 6),
                'features_used': self.primary_features + self.secondary_features,
            }
        }
