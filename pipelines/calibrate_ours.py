#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate Ours Method (自动校准)
================================

This module performs automatic calibration of the ours method parameters
by grid search on a validation set.

Calibration targets:
- alpha: Softmax temperature
- T_normal, T_prob: Normal detection thresholds
- beta_freq, beta_ref: Evidence gating boost factors

Optimization objective:
- Primary: Accuracy
- Secondary: Macro-F1

Output:
- Output/calibration.json containing all calibration parameters
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


def evaluate_with_params(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    alpha: float,
    overall_threshold: float,
    max_prob_threshold: float,
    beta_freq: float,
    beta_ref: float
) -> Tuple[float, float]:
    """Evaluate accuracy and F1 with given parameters.
    
    Returns:
        (accuracy, macro_f1)
    """
    from BRB.system_brb import system_level_infer, SystemBRBConfig
    from BRB.aggregator import load_calibration, DEFAULT_CALIBRATION
    
    # Build calibration dict
    calibration = {
        'alpha': alpha,
        'overall_threshold': overall_threshold,
        'max_prob_threshold': max_prob_threshold,
        'beta_freq': beta_freq,
        'beta_ref': beta_ref,
        'T_normal': overall_threshold,
        'T_prob': max_prob_threshold,
    }
    
    config = SystemBRBConfig(
        alpha=alpha,
        overall_threshold=overall_threshold,
        max_prob_threshold=max_prob_threshold,
    )
    
    # Align samples
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    if not common_ids:
        return 0.0, 0.0
    
    y_true = []
    y_pred = []
    
    label_map = {'正常': 0, '幅度失准': 1, '频率失准': 2, '参考电平失准': 3}
    
    for sample_id in common_ids:
        features = features_dict[sample_id]
        label_entry = labels_dict[sample_id]
        
        # Get true label
        true_label = extract_system_label(label_entry)
        y_true.append(true_label)
        
        # Get prediction
        result = system_level_infer(features, config, mode='sub_brb')
        probs = result.get('probabilities', {})
        
        # Map predicted class
        pred_class = max(probs, key=probs.get) if probs else '正常'
        pred_label = label_map.get(pred_class, 0)
        y_pred.append(pred_label)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    
    # Macro F1
    n_classes = 4
    f1_scores = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    return accuracy, macro_f1


def grid_search_calibration(
    features_dict: Dict[str, Dict],
    labels_dict: Dict[str, Dict],
    verbose: bool = True
) -> Dict:
    """Perform grid search to find optimal calibration parameters.
    
    Returns:
        Best calibration parameters dictionary.
    """
    # Grid ranges
    alpha_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    overall_threshold_range = [0.10, 0.15, 0.20, 0.25]
    max_prob_threshold_range = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    beta_freq_range = [0.0, 0.3, 0.5, 0.7, 1.0]
    beta_ref_range = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    best_accuracy = 0.0
    best_f1 = 0.0
    best_params = None
    
    total_configs = (
        len(alpha_range) * len(overall_threshold_range) * 
        len(max_prob_threshold_range) * len(beta_freq_range) * len(beta_ref_range)
    )
    
    if verbose:
        print(f"Running grid search with {total_configs} configurations...")
    
    config_count = 0
    
    for alpha in alpha_range:
        for overall_threshold in overall_threshold_range:
            for max_prob_threshold in max_prob_threshold_range:
                for beta_freq in beta_freq_range:
                    for beta_ref in beta_ref_range:
                        config_count += 1
                        
                        try:
                            accuracy, macro_f1 = evaluate_with_params(
                                features_dict, labels_dict,
                                alpha, overall_threshold, max_prob_threshold,
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
                                'overall_threshold': overall_threshold,
                                'max_prob_threshold': max_prob_threshold,
                                'T_normal': overall_threshold,
                                'T_prob': max_prob_threshold,
                                'beta_freq': beta_freq,
                                'beta_ref': beta_ref,
                            }
                        
                        if verbose and config_count % 100 == 0:
                            print(f"  Progress: {config_count}/{total_configs}, "
                                  f"Best acc={best_accuracy:.4f}, F1={best_f1:.4f}")
    
    if verbose:
        print(f"\nGrid search complete!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best macro-F1: {best_f1:.4f}")
        print(f"Best params: {best_params}")
    
    return best_params, best_accuracy, best_f1


def save_calibration(calibration: Dict, output_path: Path):
    """Save calibration parameters to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    print(f"Saved calibration to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ours method parameters")
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
    
    # Run grid search
    best_params, best_acc, best_f1 = grid_search_calibration(
        features_dict, labels_dict, verbose=not args.quiet
    )
    
    if best_params is None:
        print("Error: Grid search failed to find valid parameters")
        return 1
    
    # Add metadata
    best_params['calibration_accuracy'] = best_acc
    best_params['calibration_macro_f1'] = best_f1
    best_params['single_band_mode'] = True
    
    # Save calibration
    calibration_path = output_dir / "calibration.json"
    save_calibration(best_params, calibration_path)
    
    # Also save to sim_spectrum directory if different
    if data_dir != output_dir:
        save_calibration(best_params, data_dir / "calibration.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
