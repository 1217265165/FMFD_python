#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate Ours Method (v8: Multi-evidence Ref Gate)
====================================================

This module performs automatic calibration of the ours method parameters
by grid search on the training/validation set.

v8 improvements (Multi-evidence Ref Gate for Amp vs Ref):
- Added X29, X30, X31 features for Amp vs Ref discrimination
- Ref Gate Score: I(X14>t14) + I(X29>t29) + I(X30>t30) >= 2
- Only enable ref boosting when consistent evidence across multiple features
- Output: feature_separation_amp_ref.csv for verification

v5-v7 features retained:
- alpha: Softmax temperature (base)
- gamma: Reliability adaptive temperature factor
- T_low, T_high: Dual thresholds for soft gating
- k_normal_prior: Normal logit scaling factor
- beta_freq, beta_ref, beta_amp: Evidence gating boost factors

Optimization objective:
- Primary: Balanced Accuracy (mean recall)
- Secondary: Macro-F1

Output:
- Output/calibration.json containing all calibration parameters
- Output/sim_spectrum/anchor_score_by_class.csv
- Output/sim_spectrum/pred_distribution_ours.csv
- Output/sim_spectrum/ours_debug_system.csv
- Output/sim_spectrum/calibration_leaderboard.csv
- Output/sim_spectrum/feature_separation_amp_ref.csv (NEW in v8)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.default_paths import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    SIM_DIR,
    SEED,
    SINGLE_BAND,
    DISABLE_PREAMP,
    build_run_snapshot,
)


def _hash_ids(ids: List[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for item in ids:
        h.update(str(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _stratified_split_ids(
    sample_ids: List[str],
    labels: List[int],
    train_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    indices = np.arange(len(sample_ids))
    train_indices: List[int] = []
    val_indices: List[int] = []

    for class_label in sorted(set(labels)):
        class_indices = indices[np.array(labels) == class_label]
        rng = np.random.RandomState(seed)
        rng.shuffle(class_indices)
        n_class = len(class_indices)
        n_train = max(1, int(n_class * train_size))
        n_val = max(1, int(n_class * val_size))
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])

    rng = np.random.RandomState(seed)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_ids = [sample_ids[i] for i in train_indices]
    val_ids = [sample_ids[i] for i in val_indices]
    return train_ids, val_ids


def load_features_and_labels(data_dir: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load features and labels from data directory.
    
    Returns:
        (features_dict, labels_dict)
    """
    features_path = data_dir / "features_brb.csv"
    labels_path = data_dir / "labels.json"
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing features or labels in {data_dir}")
    
    # Load features
    features_dict = {}
    with open(features_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get('sample_id') or row.get('id')
            if sample_id:
                feat_row = {}
                for k, v in row.items():
                    if k not in ['sample_id', 'id']:
                        try:
                            feat_row[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                features_dict[sample_id] = feat_row
    
    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    return features_dict, labels_dict


def extract_system_label(entry: Dict) -> int:
    """Extract numeric system label from entry.
    
    Returns: 0=Normal, 1=Amp, 2=Freq, 3=Ref
    """
    if 'system_label' in entry:
        label = str(entry['system_label'])
    elif entry.get('type') == 'normal':
        return 0
    elif entry.get('type') == 'fault':
        fault_cls = entry.get('system_fault_class', '')
        mapping = {
            'amp_error': 1,
            'freq_error': 2,
            'ref_error': 3,
        }
        return mapping.get(fault_cls, 0)
    else:
        return 0
    
    # Map string labels
    mapping = {
        '正常': 0,
        '幅度失准': 1,
        '频率失准': 2,
        '参考电平失准': 3,
    }
    return mapping.get(label, 0)


def compute_anchor_scores_for_samples(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict]
) -> List[Dict]:
    """Compute anchor scores for all samples.
    
    Returns list of dicts with sample_id, true_label, anchor_score, components.
    """
    from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
    
    config = NormalAnchorConfig()
    results = []
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label_entry = labels_dict[sample_id]
        true_label = extract_system_label(label_entry)
        
        anchor_result = compute_anchor_score(features, config)
        
        row = {
            'sample_id': sample_id,
            'true_label': true_label,
            'true_class': ['Normal', 'Amp', 'Freq', 'Ref'][true_label],
            'anchor_score': anchor_result['anchor_score'],
        }
        row.update({f'comp_{k}': v for k, v in anchor_result['components'].items()})
        results.append(row)
    
    return results


def save_anchor_score_by_class(results: List[Dict], output_path: Path):
    """Save anchor score statistics by true class."""
    by_class = {0: [], 1: [], 2: [], 3: []}
    
    for r in results:
        by_class[r['true_label']].append(r['anchor_score'])
    
    stats = []
    class_names = ['Normal', 'Amp', 'Freq', 'Ref']
    
    for cls, scores in by_class.items():
        if scores:
            scores = np.array(scores)
            stats.append({
                'class': class_names[cls],
                'n': len(scores),
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'p25': np.percentile(scores, 25),
                'p75': np.percentile(scores, 75),
                'p90': np.percentile(scores, 90),
                'p95': np.percentile(scores, 95),
            })
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if stats:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
    
    print(f"Saved anchor_score_by_class to: {output_path}")
    
    # Also print summary
    print("\n=== Anchor Score by Class ===")
    for s in stats:
        print(f"  {s['class']:8s}: mean={s['mean']:.4f}, p90={s['p90']:.4f}, p95={s['p95']:.4f}")


def evaluate_with_params_v2(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    alpha: float,
    T_low: float,
    T_high: float,
    k_normal_prior: float,
    beta_freq: float,
    beta_ref: float,
    gamma: float = 0.5,
    beta_amp: float = 0.5
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate accuracy and F1 with given v5 parameters.
    
    Returns:
        (accuracy, macro_f1, balanced_acc, y_true, y_pred)
    """
    from BRB.aggregator import system_level_infer_with_sub_brbs
    
    # Build calibration dict for v5
    calibration = {
        'alpha': alpha,
        'gamma': gamma,  # v5 NEW: reliability adaptive temperature
        'T_low': T_low,
        'T_high': T_high,
        'k_normal_prior': k_normal_prior,
        'beta_freq': beta_freq,
        'beta_ref': beta_ref,
        'beta_amp': beta_amp,  # v6 NEW: also calibrate amp boost
    }
    
    # Align samples
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    if not common_ids:
        return 0.0, 0.0, [], []
    
    y_true = []
    y_pred = []
    
    label_map = {'正常': 0, '幅度失准': 1, '频率失准': 2, '参考电平失准': 3}
    
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label_entry = labels_dict[sample_id]
        
        # Get true label
        true_label = extract_system_label(label_entry)
        y_true.append(true_label)
        
        # Get prediction using v2 inference
        result = system_level_infer_with_sub_brbs(features, alpha, calibration=calibration)
        probs = result.get('probabilities', {})
        
        # Map predicted class
        pred_class = max(probs, key=probs.get) if probs else '正常'
        pred_label = label_map.get(pred_class, 0)
        y_pred.append(pred_label)
    
    # Calculate metrics
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    accuracy = np.mean(y_true_np == y_pred_np)
    
    # Macro F1 and per-class recall (for balanced accuracy)
    n_classes = 4
    f1_scores = []
    recalls = []
    for c in range(n_classes):
        tp = np.sum((y_true_np == c) & (y_pred_np == c))
        fp = np.sum((y_true_np != c) & (y_pred_np == c))
        fn = np.sum((y_true_np == c) & (y_pred_np != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        recalls.append(recall)
    
    macro_f1 = np.mean(f1_scores)
    balanced_accuracy = np.mean(recalls)  # Balanced accuracy = mean recall across classes
    
    return accuracy, macro_f1, balanced_accuracy, y_true, y_pred


def grid_search_calibration_v2(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    verbose: bool = True
) -> Tuple[Dict, float, float]:
    """Perform grid search to find optimal v3 calibration parameters.
    
    v3 improvements:
    - Changed optimization objective to BALANCED ACCURACY (mean recall)
      as primary metric, macro-F1 as secondary.
    - Uses grouped-max anchor scoring
    - Saves calibration_leaderboard.csv with top configurations
    - v7: Auto-calibrates T_low/T_high using percentiles from Normal class
    
    Returns:
        (best_params, best_balanced_accuracy, best_f1, leaderboard)
    """
    # First compute anchor scores to determine T_low/T_high ranges
    from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
    
    config = NormalAnchorConfig()
    normal_scores = []
    fault_scores = {'amp': [], 'freq': [], 'ref': []}
    
    # Also track jump_energy and env_violation for forced not-normal rule
    normal_jump_energies = []
    normal_env_violations = []
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label = extract_system_label(labels_dict[sample_id])
        result = compute_anchor_score(features, config)
        score = result['anchor_score']
        score_amp = result.get('score_amp', 0)
        score_freq = result.get('score_freq', 0)
        score_ref = result.get('score_ref', 0)
        
        if label == 0:
            normal_scores.append(score)
            # Track key features for forced not-normal rule
            normal_jump_energies.append(features.get('X7', 0))
            normal_env_violations.append(features.get('X13', 0))
        elif label == 1:
            fault_scores['amp'].append(score)
        elif label == 2:
            fault_scores['freq'].append(score)
        elif label == 3:
            fault_scores['ref'].append(score)
    
    normal_scores = np.array(normal_scores) if normal_scores else np.array([0.1])
    all_fault_scores = []
    for k, v in fault_scores.items():
        all_fault_scores.extend(v)
    all_fault_scores = np.array(all_fault_scores) if all_fault_scores else np.array([0.5])
    
    # v7: Auto-calibrate T_low and T_high based on Normal class percentiles
    # T_low = 85th percentile of normal scores
    # T_high = 97th percentile of normal scores
    T_low_auto = float(np.percentile(normal_scores, 85))
    T_high_auto = float(np.percentile(normal_scores, 97))
    
    # Also compute thresholds for forced not-normal rule
    jump_energy_p99 = float(np.percentile(normal_jump_energies, 99)) if normal_jump_energies else 0.5
    env_violation_p99 = float(np.percentile(normal_env_violations, 99)) if normal_env_violations else 100.0
    
    # v7: Include auto-calibrated values in search plus some variations
    # Constants for threshold range calculation
    T_LOW_MIN = 0.05  # Minimum T_low to avoid too strict normal detection
    T_LOW_MAX = 0.4   # Maximum T_low to avoid too loose normal detection
    T_HIGH_MAX = 0.9  # Maximum T_high to ensure abnormal samples are detected
    T_low_range = [max(T_LOW_MIN, T_low_auto - 0.05), T_low_auto, min(T_LOW_MAX, T_low_auto + 0.1)]
    T_high_range = [max(T_low_auto + 0.1, T_high_auto - 0.1), T_high_auto, min(T_HIGH_MAX, T_high_auto + 0.1)]
    
    # Grid ranges (v7: narrowed to auto-calibrated values + variations)
    alpha_range = [1.5, 2.0, 2.5]
    gamma_range = [0.0, 0.3]  # v7: reliability adaptive temperature (reduced for stability)
    k_normal_prior_range = [0.0, 1.0, 2.0]  # Can be 0 to disable normal boost
    beta_amp_range = [1.0, 2.0, 4.0]  # amp boost to balance freq/ref
    beta_freq_range = [0.0, 2.0, 4.0]  # Increased to help freq detection
    beta_ref_range = [0.0, 2.0, 4.0]   # Increased to help ref detection
    
    best_balanced_acc = 0.0
    best_f1 = 0.0
    best_params = None
    leaderboard = []  # Store top configurations
    
    total_configs = (
        len(alpha_range) * len(gamma_range) * len(T_low_range) * len(T_high_range) *
        len(k_normal_prior_range) * len(beta_amp_range) * len(beta_freq_range) * len(beta_ref_range)
    )
    
    if verbose:
        print(f"Running v7 grid search (with auto-calibrated T_low/T_high) with {total_configs} configurations...")
        print(f"  Auto-calibrated: T_low={T_low_auto:.4f} (p85), T_high={T_high_auto:.4f} (p97)")
        print(f"  T_low range: {T_low_range}")
        print(f"  T_high range: {T_high_range}")
        print(f"  gamma range: {gamma_range} (reliability adaptive temperature)")
        print(f"  beta_amp range: {beta_amp_range}")
        print(f"  Forced not-normal thresholds: jump_energy>{jump_energy_p99:.4f}, env_violation>{env_violation_p99:.4f}")
        print(f"  Normal scores: mean={np.mean(normal_scores):.4f}, p95={np.percentile(normal_scores, 95):.4f}")
        print(f"  Fault scores: mean={np.mean(all_fault_scores):.4f}, p25={np.percentile(all_fault_scores, 25):.4f}")
        for k, v in fault_scores.items():
            if v:
                print(f"    {k}: mean={np.mean(v):.4f}, p90={np.percentile(v, 90):.4f}")
    
    config_count = 0
    
    for alpha in alpha_range:
        for gamma in gamma_range:
            for T_low in T_low_range:
                for T_high in T_high_range:
                    if T_high <= T_low:
                        continue  # Skip invalid combinations
                        
                    for k_normal_prior in k_normal_prior_range:
                        for beta_amp in beta_amp_range:
                            for beta_freq in beta_freq_range:
                                for beta_ref in beta_ref_range:
                                    config_count += 1
                                    
                                    try:
                                        accuracy, macro_f1, balanced_acc, _, _ = evaluate_with_params_v2(
                                            features_dict, labels_dict,
                                            alpha, T_low, T_high, k_normal_prior,
                                            beta_freq, beta_ref, gamma, beta_amp
                                        )
                                    except Exception as e:
                                        if verbose and config_count < 10:
                                            print(f"Error with config {config_count}: {e}")
                                        continue
                                    
                                    # Store in leaderboard
                                    config_result = {
                                        'alpha': alpha,
                                        'gamma': gamma,
                                        'T_low': T_low,
                                        'T_high': T_high,
                                        'k_normal_prior': k_normal_prior,
                                        'beta_amp': beta_amp,
                                        'beta_freq': beta_freq,
                                        'beta_ref': beta_ref,
                                        'balanced_accuracy': balanced_acc,
                                        'macro_f1': macro_f1,
                                        'accuracy': accuracy,
                                    }
                                    leaderboard.append(config_result)
                                    
                                    # v7: Primary = balanced_accuracy, Secondary = macro_f1
                                    if (balanced_acc > best_balanced_acc or 
                                        (balanced_acc == best_balanced_acc and macro_f1 > best_f1)):
                                        best_balanced_acc = balanced_acc
                                        best_f1 = macro_f1
                                        best_params = {
                                            'alpha': alpha,
                                            'gamma': gamma,
                                            'T_low': T_low,
                                            'T_high': T_high,
                                            'k_normal_prior': k_normal_prior,
                                            'beta_amp': beta_amp,
                                            'beta_freq': beta_freq,
                                            'beta_ref': beta_ref,
                                            # v7: Add forced not-normal thresholds
                                            'jump_energy_p99': jump_energy_p99,
                                            'env_violation_p99': env_violation_p99,
                                            # Compatibility
                                            'overall_threshold': T_low,
                                            'max_prob_threshold': 0.35,
                                            'T_normal': T_low,
                                            'T_prob': 0.35,
                                    }
                                
                                if verbose and config_count % 2000 == 0:
                                    print(f"  Progress: {config_count}/{total_configs}, "
                                          f"Best bal_acc={best_balanced_acc:.4f}, F1={best_f1:.4f}")
    
    # Save leaderboard (top 20)
    leaderboard.sort(key=lambda x: (x['balanced_accuracy'], x['macro_f1']), reverse=True)
    
    if verbose:
        print(f"\nGrid search complete!")
        print(f"Best balanced accuracy: {best_balanced_acc:.4f}")
        print(f"Best macro-F1: {best_f1:.4f}")
        print(f"Best params: {best_params}")
        print("\n=== Top 5 Configurations ===")
        for i, cfg in enumerate(leaderboard[:5]):
            print(f"  #{i+1}: bal_acc={cfg['balanced_accuracy']:.4f}, F1={cfg['macro_f1']:.4f}, "
                  f"alpha={cfg['alpha']}, gamma={cfg['gamma']}, T_low={cfg['T_low']}, "
                  f"T_high={cfg['T_high']}, beta_amp={cfg.get('beta_amp', 0.5)}, beta_freq={cfg['beta_freq']}, beta_ref={cfg['beta_ref']}")
    
    return best_params, best_balanced_acc, best_f1, leaderboard[:20]


def save_debug_output(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    calibration: Dict,
    output_dir: Path
):
    """Save detailed debug output for ours method."""
    from BRB.aggregator import system_level_infer_with_sub_brbs
    from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    
    debug_rows = []
    label_map = {'正常': 0, '幅度失准': 1, '频率失准': 2, '参考电平失准': 3}
    class_names = ['Normal', 'Amp', 'Freq', 'Ref']
    
    pred_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    config = NormalAnchorConfig()
    config.T_low = calibration.get('T_low', 0.10)
    config.T_high = calibration.get('T_high', 0.35)
    config.k_normal_prior = calibration.get('k_normal_prior', 4.0)
    
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label_entry = labels_dict[sample_id]
        true_label = extract_system_label(label_entry)
        
        # Get anchor score components
        anchor_result = compute_anchor_score(features, config)
        
        # Get full inference result
        result = system_level_infer_with_sub_brbs(features, calibration=calibration)
        probs = result.get('probabilities', {})
        logits = result.get('logits', {})
        
        pred_class = max(probs, key=probs.get) if probs else '正常'
        pred_label = label_map.get(pred_class, 0)
        pred_counts[pred_label] += 1
        
        row = {
            'sample_id': sample_id,
            'true_label': true_label,
            'true_class': class_names[true_label],
            'pred_label': pred_label,
            'pred_class': class_names[pred_label],
            'correct': int(true_label == pred_label),
            'anchor_score': anchor_result['anchor_score'],
            'anchor_score_aligned': anchor_result['anchor_score'],
            'logit_normal': logits.get('normal', 0),
            'logit_amp': logits.get('amp', 0),
            'logit_freq': logits.get('freq', 0),
            'logit_ref': logits.get('ref', 0),
            'p_normal': probs.get('正常', 0),
            'p_amp': probs.get('幅度失准', 0),
            'p_freq': probs.get('频率失准', 0),
            'p_ref': probs.get('参考电平失准', 0),
            'offset_db': features.get('offset_db', 0.0),
            'viol_rate_aligned': features.get('viol_rate_aligned', features.get('viol_rate', 0.0)),
            'viol_energy_aligned': features.get('viol_energy_aligned', 0.0),
        }
        # Add anchor components
        for k, v in anchor_result['components'].items():
            row[f'comp_{k}'] = v
        
        debug_rows.append(row)
    
    # Save debug CSV
    debug_path = output_dir / 'ours_debug_system.csv'
    if debug_rows:
        with open(debug_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=debug_rows[0].keys())
            writer.writeheader()
            writer.writerows(debug_rows)
        print(f"Saved ours_debug_system.csv to: {debug_path}")
    
    # Save prediction distribution
    dist_path = output_dir / 'pred_distribution_ours.csv'
    total = sum(pred_counts.values())
    with open(dist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'count', 'ratio'])
        for cls, count in pred_counts.items():
            ratio = count / total if total > 0 else 0
            writer.writerow([class_names[cls], count, f'{ratio:.4f}'])
    print(f"Saved pred_distribution_ours.csv to: {dist_path}")
    
    # Print prediction distribution
    print("\n=== Prediction Distribution (ours) ===")
    for cls, count in pred_counts.items():
        ratio = count / total if total > 0 else 0
        print(f"  {class_names[cls]:8s}: {count:4d} ({ratio*100:5.1f}%)")
    
    pred_normal_ratio = pred_counts[0] / total if total > 0 else 0
    print(f"\n  pred_normal_ratio = {pred_normal_ratio:.4f}")


def generate_feature_separation_amp_ref(features_dict: Dict, labels_dict: Dict, output_path: Path):
    """v8: Generate feature_separation_amp_ref.csv to verify Amp vs Ref discrimination.
    
    Outputs statistics for v8 features (X14, X29, X30, X31) across three groups:
    - Amp correct: Amp samples correctly classified
    - Amp→Ref errors: Amp samples misclassified as Ref
    - Ref correct: Ref samples correctly classified
    
    This helps verify that X29, X30, X31 can separate Amp errors from Ref errors.
    """
    from BRB.aggregator import system_level_infer_with_sub_brbs, load_calibration
    
    calibration = load_calibration()
    
    # Collect samples by class and prediction
    amp_correct = []  # Amp samples correctly predicted as Amp
    amp_to_ref = []   # Amp samples incorrectly predicted as Ref
    ref_correct = []  # Ref samples correctly predicted as Ref
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label = extract_system_label(labels_dict[sample_id])
        
        # Get prediction
        result = system_level_infer_with_sub_brbs(features, calibration=calibration)
        pred_class = result['predicted_class']
        
        # Map prediction to label index
        pred_idx = {'正常': 0, '幅度失准': 1, '频率失准': 2, '参考电平失准': 3}.get(pred_class, -1)
        
        # Extract v8 features
        f_row = {
            'X14': features.get('X14', 0),
            'X29': features.get('X29', 0),
            'X30': features.get('X30', 0),
            'X31': features.get('X31', 0),
        }
        
        if label == 1:  # Amp
            if pred_idx == 1:
                amp_correct.append(f_row)
            elif pred_idx == 3:  # Amp→Ref error
                amp_to_ref.append(f_row)
        elif label == 3:  # Ref
            if pred_idx == 3:
                ref_correct.append(f_row)
    
    # Compute statistics
    def compute_stats(group_name, rows):
        if not rows:
            return {
                'group': group_name, 'n': 0,
                'X14_mean': 0, 'X14_std': 0,
                'X29_mean': 0, 'X29_std': 0,
                'X30_mean': 0, 'X30_std': 0,
                'X31_mean': 0, 'X31_std': 0,
            }
        
        x14_vals = [r['X14'] for r in rows]
        x29_vals = [r['X29'] for r in rows]
        x30_vals = [r['X30'] for r in rows]
        x31_vals = [r['X31'] for r in rows]
        
        return {
            'group': group_name,
            'n': len(rows),
            'X14_mean': np.mean(x14_vals),
            'X14_std': np.std(x14_vals),
            'X29_mean': np.mean(x29_vals),
            'X29_std': np.std(x29_vals),
            'X30_mean': np.mean(x30_vals),
            'X30_std': np.std(x30_vals),
            'X31_mean': np.mean(x31_vals),
            'X31_std': np.std(x31_vals),
        }
    
    stats_rows = [
        compute_stats('Amp_correct', amp_correct),
        compute_stats('Amp_to_Ref_error', amp_to_ref),
        compute_stats('Ref_correct', ref_correct),
    ]
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
        writer.writeheader()
        writer.writerows(stats_rows)
    
    print(f"\nSaved feature_separation_amp_ref.csv to: {output_path}")
    print("\n=== v8 Feature Separation (Amp vs Ref) ===")
    for row in stats_rows:
        print(f"  {row['group']:20s} (n={row['n']:3d}): "
              f"X14={row['X14_mean']:.4f}, X29={row['X29_mean']:.4f}, "
              f"X30={row['X30_mean']:.4f}, X31={row['X31_mean']:.4f}")


def save_calibration(calibration: Dict, output_path: Path):
    """Save calibration parameters to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    print(f"Saved calibration to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ours method parameters (v2)")
    parser.add_argument(
        '--data_dir',
        default=SIM_DIR,
        help='Directory containing features_brb.csv and labels.json',
    )
    parser.add_argument(
        '--output_dir',
        default=OUTPUT_DIR,
        help='Output directory for calibration.json',
    )
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    build_run_snapshot(data_dir)
    print(f"[INFO] project_root={PROJECT_ROOT}")
    print(f"[INFO] single_band={SINGLE_BAND}")
    print(f"[INFO] disable_preamp={DISABLE_PREAMP}")
    print(f"[INFO] seed={SEED}")
    print(f"[INFO] output_dir={output_dir}")
    
    print(f"Loading data from: {data_dir}")
    
    # Load data
    try:
        features_dict, labels_dict = load_features_and_labels(data_dir)
        print(f"Loaded {len(features_dict)} features, {len(labels_dict)} labels")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run simulation first: python pipelines/simulate/run_simulation_brb.py")
        return 1

    # Align samples and split (train/val only; test is excluded from calibration)
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    if not common_ids:
        print("Error: No common samples between features and labels")
        return 1
    label_values = [extract_system_label(labels_dict[sample_id]) for sample_id in common_ids]
    train_ids, val_ids = _stratified_split_ids(
        common_ids,
        label_values,
        train_size=0.6,
        val_size=0.2,
        seed=SEED,
    )
    train_val_ids = set(train_ids + val_ids)
    features_dict = {k: v for k, v in features_dict.items() if k in train_val_ids}
    labels_dict = {k: v for k, v in labels_dict.items() if k in train_val_ids}
    print(f"[INFO] Calibration split: train={len(train_ids)}, val={len(val_ids)} (test excluded)")
    print(f"[INFO] Train hash: {_hash_ids(train_ids)}")
    print(f"[INFO] Val hash: {_hash_ids(val_ids)}")
    
    # Step 0: v8 - Generate feature_separation_amp_ref.csv for verification
    generate_feature_separation_amp_ref(features_dict, labels_dict, data_dir / 'feature_separation_amp_ref.csv')
    
    # Step 1: Compute and save anchor score statistics by class (with grouped scores)
    anchor_results = compute_anchor_scores_for_samples(features_dict, labels_dict)
    save_anchor_score_by_class(anchor_results, data_dir / 'anchor_score_by_class.csv')
    
    # Also save detailed anchor components by class
    save_anchor_components_by_class(features_dict, labels_dict, data_dir / 'anchor_components_by_class.csv')
    
    # Step 2: Run v3 grid search (balanced_accuracy primary)
    best_params, best_balanced_acc, best_f1, leaderboard = grid_search_calibration_v2(
        features_dict, labels_dict, verbose=not args.quiet
    )
    
    if best_params is None:
        print("Error: Grid search failed to find valid parameters")
        return 1
    
    # Add metadata
    best_params['calibration_balanced_accuracy'] = best_balanced_acc
    best_params['calibration_macro_f1'] = best_f1
    best_params['single_band_mode'] = True
    best_params['version'] = 'v8_X30_amp_ref_gate'
    best_params['calibration_split'] = {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_hash": _hash_ids(train_ids),
        "val_hash": _hash_ids(val_ids),
        "note": "calibration uses train/val only; test excluded",
    }
    
    # Save calibration
    calibration_path = output_dir / "calibration.json"
    save_calibration(best_params, calibration_path)
    
    # Also save to sim_spectrum directory if different
    if data_dir != output_dir:
        save_calibration(best_params, data_dir / "calibration.json")

    # Save best params for compare_methods.py
    best_params_path = data_dir / "best_params.json"
    save_calibration(best_params, best_params_path)

    report_path = data_dir / "calibration_report.json"
    report = {
        "best_params_path": str(best_params_path),
        "calibration_balanced_accuracy": best_balanced_acc,
        "calibration_macro_f1": best_f1,
        "objective": "improve_compare_system_accuracy",
        "single_band": SINGLE_BAND,
        "disable_preamp": DISABLE_PREAMP,
        "seed": SEED,
        "calibration_split": best_params["calibration_split"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved calibration_report.json to: {report_path}")
    
    # Save calibration leaderboard
    if leaderboard:
        leaderboard_path = data_dir / 'calibration_leaderboard.csv'
        with open(leaderboard_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=leaderboard[0].keys())
            writer.writeheader()
            writer.writerows(leaderboard)
        print(f"Saved calibration_leaderboard.csv to: {leaderboard_path}")
    
    # Step 3: Save debug output with best parameters
    save_debug_output(features_dict, labels_dict, best_params, data_dir)
    
    return 0


def save_anchor_components_by_class(features_dict: Dict, labels_dict: Dict, output_path: Path):
    """Save detailed anchor component statistics by class for debugging."""
    from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
    
    config = NormalAnchorConfig()
    by_class = {0: [], 1: [], 2: [], 3: []}
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label = extract_system_label(labels_dict[sample_id])
        result = compute_anchor_score(features, config)
        by_class[label].append(result)
    
    class_names = ['Normal', 'Amp', 'Freq', 'Ref']
    rows = []
    
    for cls, results in by_class.items():
        if not results:
            continue
        
        # Aggregate group scores
        score_amps = [r.get('score_amp', 0) for r in results]
        score_freqs = [r.get('score_freq', 0) for r in results]
        score_refs = [r.get('score_ref', 0) for r in results]
        anchor_scores = [r.get('anchor_score', 0) for r in results]
        
        row = {
            'class': class_names[cls],
            'n': len(results),
            'anchor_mean': np.mean(anchor_scores),
            'anchor_p90': np.percentile(anchor_scores, 90),
            'score_amp_mean': np.mean(score_amps),
            'score_amp_p90': np.percentile(score_amps, 90),
            'score_freq_mean': np.mean(score_freqs),
            'score_freq_p90': np.percentile(score_freqs, 90),
            'score_ref_mean': np.mean(score_refs),
            'score_ref_p90': np.percentile(score_refs, 90),
        }
        rows.append(row)
    
    if rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved anchor_components_by_class.csv to: {output_path}")
        
        # Print summary
        print("\n=== Anchor Group Scores by Class ===")
        for row in rows:
            print(f"  {row['class']:8s}: anchor={row['anchor_mean']:.4f}, "
                  f"amp={row['score_amp_mean']:.4f}, freq={row['score_freq_mean']:.4f}, ref={row['score_ref_mean']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
