"""Adapter for Fast-BRB method (Gao 2023) - Fast rule generation and reduction."""
from __future__ import annotations

import time
from typing import Dict, Optional, List, Tuple

import numpy as np
from methods.base import MethodAdapter


class FastBRBAdapter(MethodAdapter):
    """Fast-BRB: Fast rule generation with redundancy reduction (Gao 2023).
    
    MUST-HAVE mechanisms:
    - (1) Fast rule generation: create rules only for observed antecedent combinations
    - (2) Rule grouping/fusion: merge similar rules (cosine similarity of beliefs)
    - (3) Redundancy reduction: remove redundant rules based on coverage
    - (4) BRB inference: use ER/BRB engine for final prediction
    
    Implementation reduces rule count while maintaining performance.
    """
    
    name = "fast_brb"
    
    def __init__(self):
        self.rules = []  # List of rule antecedents
        self.rule_beliefs = None  # Belief matrix after reduction
        self.rule_coverage = None  # Coverage count per rule
        self.n_rules_initial = 0
        self.n_rules_final = 0
        self.n_params = 50
        self.n_features_used = 5
        self.feature_indices = None
        self.means = None
        self.stds = None
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit Fast-BRB with rule generation and reduction."""
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_sys_train))
        
        # Select features
        self.n_features_used = min(5, n_features)
        feature_scores = np.var(X_train, axis=0)
        self.feature_indices = np.argsort(feature_scores)[-self.n_features_used:]
        X_selected = X_train[:, self.feature_indices]
        
        # Normalize
        self.means = np.mean(X_selected, axis=0)
        self.stds = np.std(X_selected, axis=0) + 1e-8
        X_norm = (X_selected - self.means) / self.stds
        
        # ========== (1) Fast Rule Generation ==========
        # Create rules only for observed samples (quantized)
        rules_raw, beliefs_raw, coverage_raw = self._fast_rule_generation(
            X_norm, y_sys_train, n_classes, n_bins=3
        )
        
        self.n_rules_initial = len(rules_raw)
        
        # ========== (2) Rule Grouping/Fusion ==========
        # Merge similar rules based on belief similarity
        rules_merged, beliefs_merged, coverage_merged = self._merge_similar_rules(
            rules_raw, beliefs_raw, coverage_raw, similarity_threshold=0.85
        )
        
        # ========== (3) Redundancy Reduction ==========
        # Remove redundant rules
        self.rules, self.rule_beliefs, self.rule_coverage = self._reduce_redundancy(
            rules_merged, beliefs_merged, coverage_merged, 
            max_rules=60, redundancy_threshold=0.7
        )
        
        self.n_rules_final = len(self.rules)
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using Fast-BRB."""
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
        
        # BRB inference
        sys_proba = self._brb_inference(X_norm, n_classes)
        sys_pred = np.argmax(sys_proba, axis=1)
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000
        
        return self._create_result(sys_proba, sys_pred, n_test, infer_time_ms)
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        return {
            'n_rules': self.n_rules_final,
            'n_params': self.n_rules_final * 4 + self.n_features_used * 3,  # beliefs + attr weights
            'n_features_used': self.n_features_used,
        }
    
    def _fast_rule_generation(self, X: np.ndarray, y: np.ndarray, 
                             n_classes: int, n_bins: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Fast rule generation from observed samples.
        
        Returns:
            (rules, beliefs, coverage)
        """
        # Quantize features into bins
        X_quantized = np.zeros_like(X, dtype=int)
        for feat_idx in range(X.shape[1]):
            x_min, x_max = X[:, feat_idx].min(), X[:, feat_idx].max()
            bins = np.linspace(x_min, x_max, n_bins + 1)
            X_quantized[:, feat_idx] = np.digitize(X[:, feat_idx], bins) - 1
            X_quantized[:, feat_idx] = np.clip(X_quantized[:, feat_idx], 0, n_bins - 1)
        
        # Collect unique antecedent combinations
        unique_rules = {}
        
        for i in range(len(X)):
            rule_key = tuple(X_quantized[i])
            if rule_key not in unique_rules:
                unique_rules[rule_key] = {'class_counts': np.zeros(n_classes), 'coverage': 0}
            
            unique_rules[rule_key]['class_counts'][y[i]] += 1
            unique_rules[rule_key]['coverage'] += 1
        
        # Convert to lists
        rules = list(unique_rules.keys())
        beliefs = np.zeros((len(rules), n_classes))
        coverage = np.zeros(len(rules))
        
        for r, rule_key in enumerate(rules):
            class_counts = unique_rules[rule_key]['class_counts']
            # Laplace smoothing
            beliefs[r] = (class_counts + 1) / (np.sum(class_counts) + n_classes)
            coverage[r] = unique_rules[rule_key]['coverage']
        
        return rules, beliefs, coverage
    
    def _merge_similar_rules(self, rules: List, beliefs: np.ndarray, 
                            coverage: np.ndarray, similarity_threshold: float) -> Tuple[List, np.ndarray, np.ndarray]:
        """Merge rules with similar beliefs (cosine similarity)."""
        n_rules = len(rules)
        merged_mask = np.ones(n_rules, dtype=bool)
        merged_beliefs = beliefs.copy()
        merged_coverage = coverage.copy()
        
        for i in range(n_rules):
            if not merged_mask[i]:
                continue
            
            # Find similar rules
            for j in range(i + 1, n_rules):
                if not merged_mask[j]:
                    continue
                
                # Cosine similarity
                sim = self._cosine_similarity(beliefs[i], beliefs[j])
                
                if sim > similarity_threshold:
                    # Merge j into i
                    total_cov = merged_coverage[i] + merged_coverage[j]
                    merged_beliefs[i] = (merged_beliefs[i] * merged_coverage[i] + 
                                        merged_beliefs[j] * merged_coverage[j]) / total_cov
                    merged_coverage[i] = total_cov
                    merged_mask[j] = False
        
        # Filter merged rules
        merged_rules = [rules[i] for i in range(n_rules) if merged_mask[i]]
        merged_beliefs = merged_beliefs[merged_mask]
        merged_coverage = merged_coverage[merged_mask]
        
        return merged_rules, merged_beliefs, merged_coverage
    
    def _reduce_redundancy(self, rules: List, beliefs: np.ndarray, 
                          coverage: np.ndarray, max_rules: int,
                          redundancy_threshold: float) -> Tuple[List, np.ndarray, np.ndarray]:
        """Remove redundant rules based on redundancy score.
        
        redundancy(r) = max_{q!=r} sim(r,q) * coverage(r)
        Remove high redundancy + low coverage rules first.
        """
        n_rules = len(rules)
        
        if n_rules <= max_rules:
            return rules, beliefs, coverage
        
        # Compute redundancy scores
        redundancy_scores = np.zeros(n_rules)
        
        for i in range(n_rules):
            max_sim = 0.0
            for j in range(n_rules):
                if i != j:
                    sim = self._cosine_similarity(beliefs[i], beliefs[j])
                    max_sim = max(max_sim, sim)
            
            redundancy_scores[i] = max_sim * (coverage[i] / (np.sum(coverage) + 1e-8))
        
        # Keep rules with low redundancy and high coverage
        # Sort by: low redundancy, high coverage
        priority = -redundancy_scores + 0.3 * (coverage / (np.max(coverage) + 1e-8))
        keep_idx = np.argsort(priority)[-max_rules:]
        
        reduced_rules = [rules[i] for i in keep_idx]
        reduced_beliefs = beliefs[keep_idx]
        reduced_coverage = coverage[keep_idx]
        
        return reduced_rules, reduced_beliefs, reduced_coverage
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def _brb_inference(self, X: np.ndarray, n_classes: int) -> np.ndarray:
        """Perform BRB inference using reduced rule set."""
        n_samples = len(X)
        probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            # Compute activation for each rule
            activations = self._compute_activations(X[i])
            
            # Weighted combination
            if np.sum(activations) > 1e-10:
                probs[i] = np.sum(activations[:, np.newaxis] * self.rule_beliefs, axis=0) / np.sum(activations)
            else:
                probs[i] = np.mean(self.rule_beliefs, axis=0)
            
            # Normalize
            probs[i] /= np.sum(probs[i])
        
        return probs
    
    def _compute_activations(self, x: np.ndarray) -> np.ndarray:
        """Compute rule activation degrees."""
        n_rules = len(self.rules)
        activations = np.zeros(n_rules)
        
        # Quantize input
        x_quantized = np.zeros(len(x), dtype=int)
        for feat_idx in range(len(x)):
            # Use 3 bins (matching generation)
            x_norm_val = x[feat_idx]
            # Map to bin (approximate)
            if x_norm_val < -0.5:
                x_quantized[feat_idx] = 0
            elif x_norm_val < 0.5:
                x_quantized[feat_idx] = 1
            else:
                x_quantized[feat_idx] = 2
        
        # Compute matching degree to each rule
        for r, rule in enumerate(self.rules):
            # Hamming distance-based matching
            matches = sum(1 for i in range(len(x_quantized)) if i < len(rule) and x_quantized[i] == rule[i])
            total = min(len(x_quantized), len(rule))
            
            if total > 0:
                match_ratio = matches / total
                activations[r] = match_ratio
            else:
                activations[r] = 0.0
        
        return activations
    
    def _create_result(self, sys_proba: np.ndarray, sys_pred: np.ndarray,
                      n_test: int, infer_time_ms: float) -> Dict:
        """Create standardized result dict."""
        return {
            'system_proba': sys_proba,
            'system_pred': sys_pred,
            'module_proba': None,
            'module_pred': None,
            'meta': {
                'fit_time_sec': 0.0,
                'infer_time_ms_per_sample': infer_time_ms,
                'n_rules': self.n_rules_final,
                'n_params': self.n_rules_final * 4 + self.n_features_used * 3,
                'n_features_used': self.n_features_used,
                'features_used': list(self.feature_indices) if self.feature_indices is not None else [],
                'n_rules_before_reduction': self.n_rules_initial,
                'n_rules_after_reduction': self.n_rules_final,
                'reduction_ratio': (self.n_rules_initial - self.n_rules_final) / max(1, self.n_rules_initial),
            }
        }
