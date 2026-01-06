# Method Comparison Implementation Guide

## Overview

This document describes the implementation of 8 comparison methods for fault diagnosis in the frequency spectrum analyzer project. All methods follow a unified `MethodAdapter` interface and implement their core mechanisms as described in their respective papers.

## Unified Interface

All methods implement the `MethodAdapter` base class defined in `methods/base.py`:

```python
class MethodAdapter:
    name: str
    
    def fit(X_train, y_sys_train, y_mod_train=None, meta=None) -> None
    def predict(X_test, meta=None) -> Dict
    def complexity() -> Dict
```

### Return Format

The `predict()` method returns a standardized dictionary:

```python
{
    "system_proba": np.ndarray,      # Shape (N, C_sys) - system-level probabilities
    "system_pred": np.ndarray,       # Shape (N,) - system-level predictions
    "module_proba": np.ndarray,      # Shape (N, 21) or None - module probabilities
    "module_pred": np.ndarray,       # Shape (N,) or None - module predictions
    "meta": {
        "fit_time_sec": float,
        "infer_time_ms_per_sample": float,
        "n_rules": int,
        "n_params": int,
        "n_features_used": int,
        "features_used": list
    }
}
```

## Method Implementations

### 1. Ours (Knowledge-Driven Hierarchical BRB)
**File:** `methods/ours_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ Two-layer inference: System BRB → Module BRB
- ✅ System result gating: Only activate physically-related module subset
- ✅ Knowledge mapping: Modules use relevant frequency bands/features only

**Implementation Details:**
- Uses existing `BRB.system_brb.system_level_infer()` and `BRB.module_brb.module_level_infer()`
- Features: X1-X5 (bias, ripple_var, res_slope, df, scale_consistency)
- Rules: 45 total (12 system + 33 module)
- Parameters: 38 (attribute weights + rule weights + belief degrees)

**Key Code:**
```python
# System-level inference
sys_result = system_level_infer(features, self.config)

# Module-level inference with knowledge-driven gating
mod_probs_dict = module_level_infer(features, sys_probs)
```

---

### 2. HCF (Hierarchical Cognitive Framework, Zhang 2022)
**File:** `methods/hcf_adapter.py`

**MUST-HAVE Three-Tier Process:**
- ✅ Level-a: Feature cognitive (FN3WD approximation: Fisher score + CV error)
- ✅ Level-b: Single-source pattern cognitive (GMM clustering per source)
- ✅ Level-c: Data climate cognitive (LogisticRegression on encoded features)

**Implementation Details:**
- Feature selection using F-test for separability
- Semantic grouping into sources: amplitude, frequency, noise, other
- GMM with 2-3 components per source for pattern encoding
- Logistic regression for final classification
- Rules: 90, Parameters: 130, Features: 6-53 (uses pool features)

**Key Code:**
```python
# Level-a: Feature selection
f_scores, _ = f_classif(X_train, y_sys_train)

# Level-b: GMM clustering per source
gmm = GaussianMixture(n_components=n_clusters)
cluster_labels = gmm.predict(X_source)

# Level-c: Classification
classifier = LogisticRegression()
classifier.fit(X_encoded, y_sys_train)
```

---

### 3. AIFD (Adaptive Interpretable Fault Diagnosis, Li 2022)
**File:** `methods/aifd_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ Adaptive rule weight update based on sensitivity
- ✅ Sensitivity estimation using finite differences (dL/dw)
- ✅ Projection to valid weight space (non-negative, normalized)
- ✅ Iterative optimization (20 epochs) with interpretability preservation

**Implementation Details:**
- BRB structure with adaptive rule weight learning
- Gradient estimation via finite differences: `grad = (loss_plus - loss) / eps`
- Learning rate decay: `lr *= 0.95` per epoch
- Rules: 72, Parameters: 110, Features: 6

**Key Code:**
```python
# Sensitivity estimation
for i in range(len(self.rule_weights)):
    self.rule_weights[i] += eps
    loss_plus = self._cross_entropy_loss(probs_plus, y_train)
    gradients[i] = (loss_plus - loss) / eps
    self.rule_weights[i] -= eps

# Update with gradient descent
self.rule_weights -= learning_rate * gradients
self.rule_weights = np.maximum(self.rule_weights, 0.01)
self.rule_weights /= np.sum(self.rule_weights)
```

---

### 4. BRB-P (Probability-Constrained BRB, Ming 2023)
**File:** `methods/brb_p_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ Probability table initialization from training data statistics
- ✅ Laplace smoothing for sparse neighborhoods
- ✅ Interpretability constraint optimization (L2 penalty from init)
- ✅ Semantic constraints (sum-to-one preservation)

**Implementation Details:**
- Initialize beta[r,c] = (count_c + 1) / (total + n_classes) for samples in rule neighborhood
- Optimize: `loss = CE + λ1*||beta - beta_init||² + λ2*semantic_penalty`
- Uses finite differences for gradient estimation
- Rules: 81, Parameters: 571 (full belief matrix), Features: 15

**Key Code:**
```python
# Probability table initialization
for r in range(self.n_rules):
    in_neighborhood = distances < threshold
    y_neighborhood = y[in_neighborhood]
    for c in range(n_classes):
        count = np.sum(y_neighborhood == c)
        beta_init[r, c] = (count + 1) / (len(y_neighborhood) + n_classes)

# Constrained optimization
total_loss = ce_loss + lambda1 * np.sum((beta - beta_init)**2)
```

---

### 5. BRB-MU (Multi-Source Uncertainty Fusion, Feng 2024)
**File:** `methods/brb_mu_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ Multi-source feature grouping (4 sources: amplitude, frequency, noise, switching)
- ✅ Uncertainty modeling: u_s = f(SNR, SVD)
  - SNR component: `u_snr = 1 / (1 + mean/std)`
  - SVD component: `u_svd = 1 - (sigma_1 / sum(sigma))`
- ✅ Fusion weights: `w_s ∝ (1 - u_s)`
- ✅ Weighted prediction fusion

**Implementation Details:**
- Semantic source grouping based on feature names
- Per-source Gaussian models
- Uncertainty combines SNR (0.6 weight) and SVD (0.4 weight)
- Rules: 72, Parameters: 110, Features: 53 (uses all pool features)

**Key Code:**
```python
# SNR uncertainty
snr = mean_signal / (std_noise + 1e-8)
u_snr = 1.0 / (1.0 + snr)

# SVD uncertainty
U, S, Vt = np.linalg.svd(X)
u_svd = 1.0 - (S[0] / np.sum(S))

# Combined uncertainty
u_combined = 0.6 * u_snr + 0.4 * u_svd

# Fusion
sys_proba = sum(w_s * source_probs[s] for s in sources)
```

---

### 6. DBRB (Deep BRB, Zhao 2024)
**File:** `methods/dbrb_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ XGBoost/GradientBoosting for feature importance ranking
- ✅ Layered structure (3 layers)
- ✅ Progressive input: Layer2 = features + z1, Layer3 = features + z2
- ✅ Layer-by-layer training

**Implementation Details:**
- Feature importance from XGBoost (fallback to variance if unavailable)
- Layer 1: top 5 features → z1 (probabilities)
- Layer 2: next 10 features + z1 → z2
- Layer 3: remaining features + z2 → final output
- Rules: 60, Parameters: 90, Features: 53

**Key Code:**
```python
# Get feature importance
from xgboost import XGBClassifier
gb_model = XGBClassifier(n_estimators=50, max_depth=3)
gb_model.fit(X_train, y_sys_train)
importance = gb_model.feature_importances_

# Layer-by-layer training
z1_train = self._predict_layer(X_layer1, self.layer1_model)
X_layer2 = np.hstack([X_train[:, layer2_features], z1_train])
z2_train = self._predict_layer(X_layer2, self.layer2_model)
```

---

### 7. A-IBRB (Automatic Interval-BRB, Wan 2025)
**File:** `methods/a_ibrb_adapter.py`

**MUST-HAVE Four-Step Process:**
- ✅ Interval construction: error-constrained k-means++ 1D clustering
- ✅ Interval rule generation: only observed combinations
- ✅ Belief initialization (GIBM): Gaussian-weighted interval statistics
- ✅ Optimization (P-CMA-ES approximation): constrained tuning

**Implementation Details:**
- k-means++ with reconstruction error threshold (epsilon = 0.3 * std)
- Interval boundaries at midpoints between cluster centers
- Rules only for observed sample interval combinations
- Rules: 4-50 (adaptive), Parameters: 65, Features: 5

**Key Code:**
```python
# Interval construction
for k in range(2, max_k + 1):
    centers = self._kmeans_1d(x, k)
    boundaries = [x.min()]
    for i in range(len(sorted_centers) - 1):
        mid = (sorted_centers[i] + sorted_centers[i+1]) / 2
        boundaries.append(mid)
    
    # Check reconstruction error
    error = np.std(x - reconstructed)
    if error <= epsilon:
        break

# GIBM belief initialization
for r, rule_combo in enumerate(self.rules):
    in_rule = samples_in_interval_combination
    beliefs[r] = (class_counts + 1) / (total + n_classes)
```

---

### 8. Fast-BRB (Gao 2023)
**File:** `methods/fast_brb_adapter.py`

**MUST-HAVE Mechanisms:**
- ✅ Fast rule generation: only observed antecedent combinations
- ✅ Rule grouping/fusion: merge similar rules (cosine similarity > 0.85)
- ✅ Redundancy reduction: remove high redundancy + low coverage
- ✅ BRB inference with reduced rule set

**Implementation Details:**
- Quantize features into 3 bins
- Generate rules from unique quantized combinations
- Merge similar rules: `cosine_sim(belief_i, belief_j) > threshold`
- Redundancy score: `max_sim * coverage_ratio`
- Rules: 2-60 (after reduction), Parameters: 23-240, Features: 5

**Key Code:**
```python
# Fast rule generation (quantization)
X_quantized = np.digitize(X, bins) - 1
unique_rules = {tuple(X_quantized[i]): ... for i in range(len(X))}

# Merge similar rules
sim = cosine_similarity(beliefs[i], beliefs[j])
if sim > 0.85:
    merged_beliefs[i] = weighted_average(beliefs[i], beliefs[j])

# Redundancy reduction
redundancy_scores[i] = max_sim * (coverage[i] / total_coverage)
keep_idx = argsort(priority)[-max_rules:]
```

---

## Feature Views

### Knowledge-Driven Features (Ours)
Used by the proposed method:
- X1: bias (amplitude offset)
- X2: ripple_var (in-band flatness)
- X3: res_slope (HF attenuation slope)
- X4: df (frequency scale nonlinearity)
- X5: scale_consistency (amplitude scaling consistency)

**Total: 10 features (5 + 5 aliases)**

### Pool Features (Comparison Methods)
Generated by `features/feature_pool.py` from raw frequency response curves:

**Amplitude Global (11 features):**
- amp_mean, amp_std, amp_min, amp_max, amp_range
- amp_median, amp_q25, amp_q75, amp_iqr
- amp_skewness, amp_kurtosis

**Frequency Scale (6 features):**
- freq_min, freq_max, freq_span
- freq_step_mean, freq_step_std, freq_step_cv

**Noise & Ripple (7 features):**
- ripple_var, ripple_std, ripple_max_dev
- trend_slope, trend_intercept
- noise_level, noise_peak

**Switching/Transition (1 feature):**
- switching_rate

**Band-Local (16 features):**
- band1_mean, band1_std, band1_max, band1_min
- band2_mean, band2_std, band2_max, band2_min
- band3_mean, band3_std, band3_max, band3_min
- band4_mean, band4_std, band4_max, band4_min

**Spectral Shape (2 features):**
- hf_attenuation_slope
- band1_energy_ratio

**Compatibility (10 features):**
- bias, X1-X5, scale_consistency, df, res_slope (aliases/mappings)

**Total: 53 features**

---

## Running the Comparison Pipeline

### Prerequisites

```bash
pip install numpy scipy scikit-learn matplotlib
```

Optional (for DBRB):
```bash
pip install xgboost
```

### Generate Features (if not already done)

```bash
python pipelines/generate_features.py
```

This creates `Output/sim_spectrum/features_brb.csv` from raw curves.

### Run Full Comparison

```bash
python pipelines/compare_methods.py --data_dir Output/sim_spectrum --output_dir Output/sim_spectrum
```

### Run with Small-Sample Experiments

```bash
python pipelines/compare_methods.py --data_dir Output/sim_spectrum --output_dir Output/sim_spectrum --small_sample
```

This will additionally test each method with training sizes [5, 10, 20, 30] repeated 5 times.

---

## Output Files

After running the comparison pipeline, the following files are generated in the output directory:

1. **comparison_table.csv** - Main results table with columns:
   - method, sys_accuracy, sys_macro_f1, mod_top1_accuracy
   - fit_time_sec, infer_ms_per_sample
   - n_rules, n_params, n_features_used

2. **confusion_matrix_<method>.png** - System-level confusion matrix for each method

3. **compare_barplot.png** - Bar chart comparing rules, parameters, and inference time

4. **small_sample_curve.csv** - Small-sample adaptability results (if --small_sample used)

5. **small_sample_curve.png** - Learning curves for different training sizes (if --small_sample used)

---

## Results Interpretation

### Accuracy vs Complexity Trade-off

From the comparison table, you can analyze:
- **Best accuracy:** Which method achieves highest system-level accuracy
- **Best efficiency:** Which method has lowest inference time
- **Best compactness:** Which method uses fewest rules/parameters
- **Best interpretability:** Consider rule count + parameter count + feature count

### Expected Patterns

Based on the implementation:
- **Ours:** Moderate accuracy with low complexity (knowledge-driven pruning)
- **HCF:** Good accuracy with moderate complexity (hierarchical cognitive)
- **AIFD:** Variable accuracy depending on sample size (adaptive to data)
- **BRB-P:** Good accuracy with high parameter count (full belief matrix)
- **BRB-MU:** Good accuracy on noisy data (uncertainty handling)
- **DBRB:** High accuracy with moderate complexity (deep structure)
- **A-IBRB:** Adaptive rule count (automatic structure)
- **Fast-BRB:** Low complexity (aggressive rule reduction)

---

## Verification Checklist

To verify that each method implements its core mechanisms:

- [ ] **Ours:** Check meta['features_used'] contains only X1-X5 subset
- [ ] **HCF:** Check meta for 'primary_features', 'secondary_features', GMM cluster info
- [ ] **AIFD:** Verify fit_time > 0 (indicates sensitivity-based optimization ran)
- [ ] **BRB-P:** Check n_params >> n_rules (indicates full belief matrix)
- [ ] **BRB-MU:** Check meta['source_uncertainties'] and meta['source_weights']
- [ ] **DBRB:** Check meta['feature_importance'] and meta['layer_sizes'] = [5, 10, ...]
- [ ] **A-IBRB:** Check meta['interval_stats'] shows K intervals per feature
- [ ] **Fast-BRB:** Check meta['n_rules_before_reduction'] > meta['n_rules_after_reduction']

---

## Troubleshooting

### ImportError: No module named 'xgboost'

DBRB will fall back to sklearn's GradientBoostingClassifier if xgboost is not installed. To use XGBoost:

```bash
pip install xgboost
```

### sklearn Version Compatibility

If you get errors about `multi_class` parameter in LogisticRegression, the HCF adapter has a try/except fallback for older sklearn versions.

### Low Accuracy for All Methods

This can happen if:
- Features are not properly normalized
- Training set is too small
- Labels are imbalanced (check label distribution in output)

Try adjusting `--train_size` parameter or checking feature extraction quality.

### Memory Issues

If you run out of memory with large datasets:
- Reduce `n_rules` in BRB-P, A-IBRB, Fast-BRB adapters
- Use smaller `perPage` in data loading
- Process in batches

---

## Citation

If you use this implementation in your research, please cite the original papers:

1. **Ours:** [Your paper title and citation]
2. **HCF:** Zhang et al., "Hierarchical Cognitive Framework...", 2022
3. **AIFD:** Li et al., "Adaptive Interpretable Fault Diagnosis...", 2022
4. **BRB-P:** Ming et al., "Probability-constrained BRB...", 2023
5. **BRB-MU:** Feng et al., "Multi-source Uncertain Information Fusion...", 2024
6. **DBRB:** Zhao et al., "Deep BRB...", 2024
7. **A-IBRB:** Wan et al., "Automatic Interval-BRB...", 2025
8. **Fast-BRB:** Gao et al., "Fast rule generation and reduction...", 2023

---

## License

This implementation follows the same license as the main FMFD_python repository.
