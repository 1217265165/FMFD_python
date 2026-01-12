#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate Ours Method (v2: Soft Gating with Dual Thresholds)
============================================================

This module performs automatic calibration of the ours method parameters
by grid search on the training/validation set.

v2 improvements:
- T_low, T_high dual thresholds (replaces single T_normal)
- k_normal_prior for Normal logit scaling
- Diagnostic outputs: anchor_score_by_class.csv, pred_distribution_ours.csv
- ours_debug_system.csv for per-sample debugging

Calibration targets:
- alpha: Softmax temperature
- T_low, T_high: Dual thresholds for soft gating
- k_normal_prior: Normal logit scaling factor
- beta_freq, beta_ref: Evidence gating boost factors

Optimization objective:
- Primary: Accuracy
- Secondary: Macro-F1

Output:
- Output/calibration.json containing all calibration parameters
- Output/sim_spectrum/anchor_score_by_class.csv
- Output/sim_spectrum/pred_distribution_ours.csv
- Output/sim_spectrum/ours_debug_system.csv
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
    beta_ref: float
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate accuracy and F1 with given v2 parameters.
    
    Returns:
        (accuracy, macro_f1, y_true, y_pred)
    """
    from BRB.aggregator import system_level_infer_with_sub_brbs
    
    # Build calibration dict for v2
    calibration = {
        'alpha': alpha,
        'T_low': T_low,
        'T_high': T_high,
        'k_normal_prior': k_normal_prior,
        'beta_freq': beta_freq,
        'beta_ref': beta_ref,
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
    
    # Macro F1
    n_classes = 4
    f1_scores = []
    for c in range(n_classes):
        tp = np.sum((y_true_np == c) & (y_pred_np == c))
        fp = np.sum((y_true_np != c) & (y_pred_np == c))
        fn = np.sum((y_true_np == c) & (y_pred_np != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    return accuracy, macro_f1, y_true, y_pred


def grid_search_calibration_v2(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    verbose: bool = True
) -> Tuple[Dict, float, float]:
    """Perform grid search to find optimal v2 calibration parameters.
    
    Returns:
        (best_params, best_accuracy, best_f1)
    """
    # First compute anchor scores to determine T_low/T_high ranges
    from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
    
    config = NormalAnchorConfig()
    normal_scores = []
    fault_scores = []
    
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label = extract_system_label(labels_dict[sample_id])
        result = compute_anchor_score(features, config)
        score = result['anchor_score']
        
        if label == 0:
            normal_scores.append(score)
        else:
            fault_scores.append(score)
    
    normal_scores = np.array(normal_scores) if normal_scores else np.array([0.1])
    fault_scores = np.array(fault_scores) if fault_scores else np.array([0.5])
    
    # Use wider, fixed ranges since data-driven ranges are too narrow
    # Key insight: T_low should be LOW to avoid giving normal prior to most samples
    # T_high should capture where faults clearly separate
    T_low_range = [0.05, 0.08, 0.10, 0.12, 0.15]
    T_high_range = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    # Grid ranges - k_normal_prior should be smaller to not dominate
    alpha_range = [1.5, 2.0, 2.5, 3.0, 4.0]
    k_normal_prior_range = [0.5, 1.0, 1.5, 2.0, 3.0]  # Reduced from [2,4,6,8]
    beta_freq_range = [0.0, 0.5, 1.0, 1.5, 2.0]  # Increased upper range
    beta_ref_range = [0.0, 0.5, 1.0, 1.5, 2.0]   # Increased upper range
    
    best_accuracy = 0.0
    best_f1 = 0.0
    best_params = None
    
    total_configs = (
        len(alpha_range) * len(T_low_range) * len(T_high_range) *
        len(k_normal_prior_range) * len(beta_freq_range) * len(beta_ref_range)
    )
    
    if verbose:
        print(f"Running v2 grid search with {total_configs} configurations...")
        print(f"  T_low range: {T_low_range}")
        print(f"  T_high range: {T_high_range}")
        print(f"  Normal scores: mean={np.mean(normal_scores):.4f}, p95={np.percentile(normal_scores, 95):.4f}")
        print(f"  Fault scores: mean={np.mean(fault_scores):.4f}, p25={np.percentile(fault_scores, 25):.4f}")
    
    config_count = 0
    
    for alpha in alpha_range:
        for T_low in T_low_range:
            for T_high in T_high_range:
                if T_high <= T_low:
                    continue  # Skip invalid combinations
                    
                for k_normal_prior in k_normal_prior_range:
                    for beta_freq in beta_freq_range:
                        for beta_ref in beta_ref_range:
                            config_count += 1
                            
                            try:
                                accuracy, macro_f1, _, _ = evaluate_with_params_v2(
                                    features_dict, labels_dict,
                                    alpha, T_low, T_high, k_normal_prior,
                                    beta_freq, beta_ref
                                )
                            except Exception as e:
                                if verbose:
                                    print(f"Error with config {config_count}: {e}")
                                continue
                            
                            # Primary: accuracy, Secondary: macro_f1
                            if (accuracy > best_accuracy or 
                                (accuracy == best_accuracy and macro_f1 > best_f1)):
                                best_accuracy = accuracy
                                best_f1 = macro_f1
                                best_params = {
                                    'alpha': alpha,
                                    'T_low': T_low,
                                    'T_high': T_high,
                                    'k_normal_prior': k_normal_prior,
                                    'beta_freq': beta_freq,
                                    'beta_ref': beta_ref,
                                    # Compatibility
                                    'overall_threshold': T_low,
                                    'max_prob_threshold': 0.35,
                                    'T_normal': T_low,
                                    'T_prob': 0.35,
                                }
                            
                            if verbose and config_count % 200 == 0:
                                print(f"  Progress: {config_count}/{total_configs}, "
                                      f"Best acc={best_accuracy:.4f}, F1={best_f1:.4f}")
    
    if verbose:
        print(f"\nGrid search complete!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best macro-F1: {best_f1:.4f}")
        print(f"Best params: {best_params}")
    
    return best_params, best_accuracy, best_f1


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
            'logit_normal': logits.get('normal', 0),
            'logit_amp': logits.get('amp', 0),
            'logit_freq': logits.get('freq', 0),
            'logit_ref': logits.get('ref', 0),
            'p_normal': probs.get('正常', 0),
            'p_amp': probs.get('幅度失准', 0),
            'p_freq': probs.get('频率失准', 0),
            'p_ref': probs.get('参考电平失准', 0),
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


def save_calibration(calibration: Dict, output_path: Path):
    """Save calibration parameters to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    print(f"Saved calibration to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ours method parameters (v2)")
    parser.add_argument('--data_dir', default='Output/sim_spectrum',
                       help='Directory containing features_brb.csv and labels.json')
    parser.add_argument('--output_dir', default='Output',
                       help='Output directory for calibration.json')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else ROOT / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    
    # Load data
    try:
        features_dict, labels_dict = load_features_and_labels(data_dir)
        print(f"Loaded {len(features_dict)} features, {len(labels_dict)} labels")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run simulation first: python pipelines/simulate/run_simulation_brb.py")
        return 1
    
    # Step 1: Compute and save anchor score statistics by class
    anchor_results = compute_anchor_scores_for_samples(features_dict, labels_dict)
    save_anchor_score_by_class(anchor_results, data_dir / 'anchor_score_by_class.csv')
    
    # Step 2: Run v2 grid search
    best_params, best_acc, best_f1 = grid_search_calibration_v2(
        features_dict, labels_dict, verbose=not args.quiet
    )
    
    if best_params is None:
        print("Error: Grid search failed to find valid parameters")
        return 1
    
    # Add metadata
    best_params['calibration_accuracy'] = best_acc
    best_params['calibration_macro_f1'] = best_f1
    best_params['single_band_mode'] = True
    best_params['version'] = 'v2_soft_gating'
    
    # Save calibration
    calibration_path = output_dir / "calibration.json"
    save_calibration(best_params, calibration_path)
    
    # Also save to sim_spectrum directory if different
    if data_dir != output_dir:
        save_calibration(best_params, data_dir / "calibration.json")
    
    # Step 3: Save debug output with best parameters
    save_debug_output(features_dict, labels_dict, best_params, data_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
