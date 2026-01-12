#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration script for OursAdapter thresholds.

Step3 requirement: Automatic calibration via grid search.

This script performs grid search over:
- T_normal: {p95, p97, p99} normal anchor threshold
- alpha: 1.5~4.0 softmax temperature
- β_freq/β_ref: 0.5~2.0 branch boost amounts

Outputs calibration.json with optimal parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features and labels from data directory."""
    from pipelines.compare_methods import discover_dataset_files, load_labels, load_features_csv
    
    features_path, labels_path = discover_dataset_files(data_dir)
    
    if not features_path or not labels_path:
        raise FileNotFoundError(f"Could not find features or labels in {data_dir}")
    
    labels_dict = load_labels(labels_path)
    features_dict = load_features_csv(features_path)
    
    # Align samples
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    
    if not common_ids:
        raise ValueError("No common samples found between features and labels")
    
    # Build feature matrix
    feature_names = list(features_dict[common_ids[0]].keys())
    n_features = len(feature_names)
    n_samples = len(common_ids)
    
    X = np.zeros((n_samples, n_features))
    y_list = []
    
    # Fixed label mapping
    FIXED_SYS_LABEL_TO_IDX = {
        '正常': 0, 'Normal': 0, 'normal': 0,
        '幅度失准': 1, 'Amp': 1, 'amp': 1, 'amp_error': 1,
        '频率失准': 2, 'Freq': 2, 'freq': 2, 'freq_error': 2,
        '参考电平失准': 3, 'Ref': 3, 'ref': 3, 'ref_error': 3,
    }
    
    for i, sample_id in enumerate(common_ids):
        feat_dict = features_dict[sample_id]
        for j, fname in enumerate(feature_names):
            X[i, j] = feat_dict.get(fname, 0.0)
        
        # Extract label
        entry = labels_dict[sample_id]
        if 'system_label' in entry:
            label_str = str(entry['system_label'])
        elif entry.get('type') == 'normal':
            label_str = '正常'
        elif entry.get('type') == 'fault':
            fault_cls = entry.get('system_fault_class', '')
            mapping = {'amp_error': '幅度失准', 'freq_error': '频率失准', 'ref_error': '参考电平失准'}
            label_str = mapping.get(fault_cls, '正常')
        else:
            label_str = '正常'
        
        y_list.append(FIXED_SYS_LABEL_TO_IDX.get(label_str, 0))
    
    y = np.array(y_list)
    
    return X, y, feature_names


def evaluate_with_params(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    normal_threshold: float, alpha: float, freq_boost: float, ref_boost: float,
    freq_gate: float = 0.3, ref_gate: float = 0.3
) -> Dict:
    """Evaluate OursAdapter with specific parameters."""
    from methods.ours_adapter import OursAdapter
    
    # Create adapter with custom parameters
    adapter = OursAdapter()
    # Enable normal anchor and evidence gating for calibration
    adapter.use_normal_anchor = True
    adapter.use_evidence_gating = True
    adapter.normal_anchor_threshold = normal_threshold
    adapter.config.alpha = alpha
    adapter.freq_boost = freq_boost
    adapter.ref_boost = ref_boost
    adapter.freq_gate = freq_gate
    adapter.ref_gate = ref_gate
    
    # Predict
    adapter.feature_names = feature_names
    pred_result = adapter.predict(X, {'feature_names': feature_names})
    y_pred = pred_result['system_pred']
    
    # Calculate metrics
    accuracy = np.mean(y == y_pred)
    
    # Per-class recall
    recalls = []
    for c in range(4):
        mask = y == c
        if mask.sum() > 0:
            recalls.append(np.mean(y_pred[mask] == c))
        else:
            recalls.append(0.0)
    
    normal_recall = recalls[0]
    macro_recall = np.mean(recalls)
    
    # Calculate F1
    f1_scores = []
    for c in range(4):
        tp = np.sum((y == c) & (y_pred == c))
        fp = np.sum((y != c) & (y_pred == c))
        fn = np.sum((y == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    return {
        'accuracy': accuracy,
        'normal_recall': normal_recall,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_recall': recalls,
    }


def grid_search(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    min_normal_recall: float = 0.4
) -> Dict:
    """Perform grid search to find optimal parameters.
    
    Objective: Maximize accuracy while constraining Normal recall >= min_normal_recall
    """
    # Grid search parameters
    # Step3 requirement: T_normal: {p95, p97, p99}
    normal_thresholds = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    alphas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    freq_boosts = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ref_boosts = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    best_result = None
    best_params = None
    best_score = -1
    
    all_results = []
    
    total_combos = len(normal_thresholds) * len(alphas) * len(freq_boosts) * len(ref_boosts)
    combo_idx = 0
    
    for normal_thresh in normal_thresholds:
        for alpha in alphas:
            for freq_boost in freq_boosts:
                for ref_boost in ref_boosts:
                    combo_idx += 1
                    if combo_idx % 100 == 0:
                        print(f"  Progress: {combo_idx}/{total_combos}")
                    
                    try:
                        result = evaluate_with_params(
                            X, y, feature_names,
                            normal_thresh, alpha, freq_boost, ref_boost
                        )
                        
                        result['normal_threshold'] = normal_thresh
                        result['alpha'] = alpha
                        result['freq_boost'] = freq_boost
                        result['ref_boost'] = ref_boost
                        
                        all_results.append(result)
                        
                        # Objective: maximize accuracy + 0.3*macro_f1, subject to Normal recall constraint
                        score = result['accuracy'] + 0.3 * result['macro_f1']
                        
                        # Apply Normal recall constraint
                        if result['normal_recall'] >= min_normal_recall:
                            if score > best_score:
                                best_score = score
                                best_result = result
                                best_params = {
                                    'normal_anchor_threshold': normal_thresh,
                                    'alpha': alpha,
                                    'freq_boost': freq_boost,
                                    'ref_boost': ref_boost,
                                    'freq_gate': 0.3,
                                    'ref_gate': 0.3,
                                }
                    except Exception as e:
                        print(f"    Error with params: {e}")
                        continue
    
    return {
        'best_params': best_params,
        'best_result': best_result,
        'all_results': all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate OursAdapter thresholds via grid search")
    parser.add_argument('--data_dir', default='Output/sim_spectrum',
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', default='Output',
                       help='Output directory for calibration.json')
    parser.add_argument('--min_normal_recall', type=float, default=0.4,
                       help='Minimum Normal recall constraint')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else ROOT / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from: {data_dir}")
    X, y, feature_names = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"Label distribution: {np.bincount(y)}")
    
    print(f"\nRunning grid search (min_normal_recall={args.min_normal_recall})...")
    search_result = grid_search(X, y, feature_names, args.min_normal_recall)
    
    if search_result['best_params'] is None:
        print("WARNING: No valid parameter combination found!")
        # Use defaults
        best_params = {
            'use_normal_anchor': True,
            'use_evidence_gating': True,
            'normal_anchor_threshold': 0.12,
            'alpha': 3.0,
            'freq_boost': 1.0,
            'ref_boost': 1.0,
            'freq_gate': 0.3,
            'ref_gate': 0.3,
        }
    else:
        best_params = search_result['best_params']
        # Add enable flags
        best_params['use_normal_anchor'] = True
        best_params['use_evidence_gating'] = True
        best_result = search_result['best_result']
        
        print(f"\nBest parameters found:")
        print(f"  normal_anchor_threshold: {best_params['normal_anchor_threshold']}")
        print(f"  alpha: {best_params['alpha']}")
        print(f"  freq_boost: {best_params['freq_boost']}")
        print(f"  ref_boost: {best_params['ref_boost']}")
        print(f"\nBest results:")
        print(f"  Accuracy: {best_result['accuracy']:.4f}")
        print(f"  Normal Recall: {best_result['normal_recall']:.4f}")
        print(f"  Macro F1: {best_result['macro_f1']:.4f}")
        print(f"  Per-class Recall: {best_result['per_class_recall']}")
    
    # Save calibration file
    calib_path = output_dir / "calibration.json"
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved calibration to: {calib_path}")
    
    # Also save to sim_spectrum directory for convenience
    sim_calib_path = output_dir / "sim_spectrum" / "calibration.json"
    sim_calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sim_calib_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"Also saved to: {sim_calib_path}")
    
    # Save all results for analysis
    if search_result['all_results']:
        results_path = output_dir / "calibration_search_results.csv"
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['normal_threshold', 'alpha', 'freq_boost', 'ref_boost',
                         'accuracy', 'normal_recall', 'macro_recall', 'macro_f1']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in search_result['all_results']:
                row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(row)
        print(f"Saved search results to: {results_path}")


if __name__ == '__main__':
    main()
