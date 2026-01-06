"""Adapter for A-IBRB method (Wan 2025) - Automatic Interval-BRB."""
from __future__ import annotations

import time
from typing import Dict, Optional, List, Tuple

import numpy as np
from methods.base import MethodAdapter


class AIBRBAdapter(MethodAdapter):
    """A-IBRB: Automatic Interval-BRB with automated structure construction (Wan 2025).
    
    MUST-HAVE mechanisms (4 steps):
    - (1) Interval construction: error-constrained k-means++ 1D clustering
    - (2) Interval rule generation: only generate rules for observed combinations
    - (3) Belief initialization (GIBM): use interval sample distribution
    - (4) Optimization (P-CMA-ES): constrained parameter tuning
    
    Implementation creates interval-based BRB with automatic partitioning.
    """
    
    name = "a_ibrb"
    
    def __init__(self):
        self.intervals = {}  # Feature -> list of interval boundaries
        self.rules = []  # List of interval rule combinations
        self.rule_beliefs = None  # Belief matrix
        self.n_rules = 50
        self.n_params = 65
        self.n_features_used = 5
        self.feature_indices = None
        self.means = None
        self.stds = None
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit A-IBRB with automatic interval construction."""
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_sys_train))
        
        # Select features
        self.n_features_used = min(5, n_features)
        feature_scores = np.var(X_train, axis=0)
        self.feature_indices = np.argsort(feature_scores)[-self.n_features_used:]
        X_selected = X_train[:, self.feature_indices]
        
        # Normalize for stability
        self.means = np.mean(X_selected, axis=0)
        self.stds = np.std(X_selected, axis=0) + 1e-8
        X_norm = (X_selected - self.means) / self.stds
        
        # ========== (1) Interval Construction ==========
        # For each feature, create intervals using error-constrained k-means++
        epsilon = 0.3  # Reconstruction error threshold (as fraction of std)
        
        for feat_idx in range(self.n_features_used):
            intervals = self._construct_intervals_1d(X_norm[:, feat_idx], epsilon)
            self.intervals[feat_idx] = intervals
        
        # ========== (2) Interval Rule Generation ==========
        # Generate rules only for observed combinations
        self.rules = self._generate_interval_rules(X_norm, y_sys_train)
        
        # Update n_rules to actual generated count
        self.n_rules = len(self.rules)
        
        # ========== (3) Belief Initialization (GIBM) ==========
        self.rule_beliefs = self._initialize_beliefs_gibm(X_norm, y_sys_train, n_classes)
        
        # ========== (4) Optimization (simplified P-CMA-ES) ==========
        self.rule_beliefs = self._optimize_beliefs(X_norm, y_sys_train, 
                                                   self.rule_beliefs, n_classes,
                                                   n_iter=20)
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using A-IBRB."""
        n_test = len(X_test)
        n_classes = self.rule_beliefs.shape[1] if self.rule_beliefs is not None else 4
        
        if self.rule_beliefs is None:
            sys_pred = np.random.randint(0, n_classes, n_test)
            sys_proba = np.eye(n_classes)[sys_pred]
            return self._create_result(sys_proba, sys_pred, n_test, 0.0)
        
        start_time = time.time()
        
        # Select and normalize
        X_selected = X_test[:, self.feature_indices]
        X_norm = (X_selected - self.means) / self.stds
        
        # Interval-based inference
        sys_proba = self._interval_inference(X_norm, n_classes)
        sys_pred = np.argmax(sys_proba, axis=1)
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000
        
        return self._create_result(sys_proba, sys_pred, n_test, infer_time_ms)
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        # Count interval boundaries
        n_intervals = sum(len(bounds) for bounds in self.intervals.values())
        
        return {
            'n_rules': self.n_rules,
            'n_params': self.n_params,
            'n_features_used': self.n_features_used,
        }
    
    def _construct_intervals_1d(self, x: np.ndarray, epsilon: float) -> List[float]:
        """Construct intervals for 1D feature using error-constrained k-means++.
        
        Returns list of interval boundaries.
        """
        # Start with k=2, increase if reconstruction error too high
        max_k = 5
        best_boundaries = None
        
        for k in range(2, max_k + 1):
            # k-means++ initialization
            centers = self._kmeans_1d(x, k)
            
            # Create interval boundaries (midpoints between centers)
            sorted_centers = np.sort(centers)
            boundaries = [x.min()]
            for i in range(len(sorted_centers) - 1):
                mid = (sorted_centers[i] + sorted_centers[i+1]) / 2
                boundaries.append(mid)
            boundaries.append(x.max())
            
            # Check reconstruction error
            reconstructed = np.zeros_like(x)
            for i in range(len(x)):
                # Find interval
                interval_idx = np.digitize(x[i], boundaries) - 1
                interval_idx = np.clip(interval_idx, 0, len(sorted_centers) - 1)
                reconstructed[i] = sorted_centers[interval_idx]
            
            error = np.std(x - reconstructed)
            
            if error <= epsilon or k == max_k:
                best_boundaries = boundaries
                break
        
        return best_boundaries if best_boundaries else [x.min(), x.max()]
    
    def _kmeans_1d(self, x: np.ndarray, k: int) -> np.ndarray:
        """Simple 1D k-means clustering."""
        # k-means++ initialization
        centers = [np.random.choice(x)]
        
        for _ in range(k - 1):
            # Distance to nearest center
            distances = np.min([np.abs(x - c) for c in centers], axis=0)
            # Sample new center proportional to distance squared
            probs = distances ** 2
            probs /= np.sum(probs)
            new_center = np.random.choice(x, p=probs)
            centers.append(new_center)
        
        centers = np.array(centers)
        
        # Lloyd's algorithm (simplified, few iterations)
        for _ in range(10):
            # Assign points to nearest center
            assignments = np.argmin([np.abs(x - c) for c in centers], axis=0)
            
            # Update centers
            new_centers = []
            for i in range(k):
                cluster_points = x[assignments == i]
                if len(cluster_points) > 0:
                    new_centers.append(np.mean(cluster_points))
                else:
                    new_centers.append(centers[i])
            
            centers = np.array(new_centers)
        
        return centers
    
    def _generate_interval_rules(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """Generate interval rules only for observed combinations."""
        rules = []
        
        # For each sample, identify its interval combination
        interval_combos = set()
        
        for i in range(len(X)):
            combo = []
            for feat_idx in range(X.shape[1]):
                boundaries = self.intervals.get(feat_idx, [X[:, feat_idx].min(), X[:, feat_idx].max()])
                interval_idx = np.digitize(X[i, feat_idx], boundaries) - 1
                interval_idx = np.clip(interval_idx, 0, len(boundaries) - 2)
                combo.append(interval_idx)
            
            interval_combos.add(tuple(combo))
        
        rules = list(interval_combos)
        
        # Limit to reasonable number
        if len(rules) > 100:
            rules = rules[:100]
        
        return rules
    
    def _initialize_beliefs_gibm(self, X: np.ndarray, y: np.ndarray, 
                                 n_classes: int) -> np.ndarray:
        """Initialize beliefs using GIBM (Gaussian-weighted interval-based method)."""
        n_rules = len(self.rules)
        beliefs = np.zeros((n_rules, n_classes))
        
        for r, rule_combo in enumerate(self.rules):
            # Find samples in this interval combination
            in_rule = np.ones(len(X), dtype=bool)
            
            for feat_idx, interval_idx in enumerate(rule_combo):
                boundaries = self.intervals.get(feat_idx, [X[:, feat_idx].min(), X[:, feat_idx].max()])
                lower = boundaries[interval_idx]
                upper = boundaries[interval_idx + 1] if interval_idx + 1 < len(boundaries) else boundaries[-1]
                
                in_interval = (X[:, feat_idx] >= lower) & (X[:, feat_idx] <= upper)
                in_rule &= in_interval
            
            # Count class frequencies with Gaussian weighting
            if np.sum(in_rule) > 0:
                y_in_rule = y[in_rule]
                for c in range(n_classes):
                    count = np.sum(y_in_rule == c)
                    beliefs[r, c] = (count + 1) / (len(y_in_rule) + n_classes)
            else:
                # Global prior
                for c in range(n_classes):
                    beliefs[r, c] = (np.sum(y == c) + 1) / (len(y) + n_classes)
            
            # Normalize
            beliefs[r] /= np.sum(beliefs[r])
        
        return beliefs
    
    def _optimize_beliefs(self, X: np.ndarray, y: np.ndarray, 
                         beliefs_init: np.ndarray, n_classes: int,
                         n_iter: int) -> np.ndarray:
        """Optimize beliefs with constraints (simplified P-CMA-ES)."""
        beliefs = beliefs_init.copy()
        learning_rate = 0.02
        
        for iteration in range(n_iter):
            # Compute loss
            probs = self._interval_inference(X, n_classes, beliefs)
            loss = -np.mean(np.log(probs[np.arange(len(X)), y] + 1e-10))
            
            # Add regularization (stay close to init)
            reg_loss = 0.3 * np.sum((beliefs - beliefs_init) ** 2)
            total_loss = loss + reg_loss
            
            # Gradient estimation (sample-based)
            n_update = min(5, len(self.rules))
            for r in np.random.choice(len(self.rules), n_update, replace=False):
                for c in range(n_classes):
                    eps = 1e-4
                    beliefs[r, c] += eps
                    probs_plus = self._interval_inference(X, n_classes, beliefs)
                    loss_plus = -np.mean(np.log(probs_plus[np.arange(len(X)), y] + 1e-10))
                    
                    grad = (loss_plus - loss) / eps
                    beliefs[r, c] -= eps
                    beliefs[r, c] -= learning_rate * grad
                    beliefs[r, c] = np.clip(beliefs[r, c], 0.01, 0.99)
                
                beliefs[r] /= np.sum(beliefs[r])
            
            learning_rate *= 0.95
        
        return beliefs
    
    def _interval_inference(self, X: np.ndarray, n_classes: int,
                           beliefs: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform inference using interval matching."""
        if beliefs is None:
            beliefs = self.rule_beliefs
        
        n_samples = len(X)
        probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            # Find matching rules
            matching_scores = []
            
            for r, rule_combo in enumerate(self.rules):
                score = 1.0
                for feat_idx, interval_idx in enumerate(rule_combo):
                    boundaries = self.intervals.get(feat_idx, [])
                    if len(boundaries) > interval_idx + 1:
                        lower = boundaries[interval_idx]
                        upper = boundaries[interval_idx + 1]
                        
                        if lower <= X[i, feat_idx] <= upper:
                            score *= 1.0
                        else:
                            # Soft matching with distance
                            dist = min(abs(X[i, feat_idx] - lower), abs(X[i, feat_idx] - upper))
                            score *= np.exp(-dist ** 2)
                    
                matching_scores.append(score)
            
            matching_scores = np.array(matching_scores)
            
            # Weighted combination
            if np.sum(matching_scores) > 1e-10:
                probs[i] = np.sum(matching_scores[:, np.newaxis] * beliefs, axis=0) / np.sum(matching_scores)
            else:
                probs[i] = np.mean(beliefs, axis=0)
            
            # Normalize
            probs[i] /= np.sum(probs[i])
        
        return probs
    
    def _create_result(self, sys_proba: np.ndarray, sys_pred: np.ndarray,
                      n_test: int, infer_time_ms: float) -> Dict:
        """Create standardized result dict."""
        # Count interval statistics
        interval_stats = {f'feat_{i}': len(bounds) for i, bounds in self.intervals.items()}
        
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
                'n_features_used': self.n_features_used,
                'features_used': list(self.feature_indices) if self.feature_indices is not None else [],
                'interval_stats': interval_stats,
            }
        }
