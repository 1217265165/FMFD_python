#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive method comparison pipeline with unified training/testing interface.

This module implements the complete experimental validation framework for comparing
the proposed method against 7 baseline methods. All methods implement the MethodAdapter
interface with fit/predict methods.

Key features:
- Automatic dataset discovery and loading
- Stratified train/val/test split with fixed seeds
- Unified evaluation metrics (system + module level)
- Complexity analysis (rules, params, features)
- Small-sample adaptability experiments
- Comprehensive output generation (CSV tables + plots)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn
        # sklearn doesn't have global seed, but individual estimators do
    except ImportError:
        pass


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def discover_dataset_files(data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-discover features and labels files in data directory.
    
    Priority:
    1. features_brb.csv / labels.json
    2. First file containing 'features' / 'labels'
    3. Single CSV with label columns
    
    Returns:
        (features_path, labels_path) or (None, None) if not found
    """
    # Try standard names first
    features_path = data_dir / "features_brb.csv"
    labels_path = data_dir / "labels.json"
    
    if features_path.exists() and labels_path.exists():
        return features_path, labels_path
    
    # Search for pattern matches
    all_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    
    feat_file = None
    label_file = None
    
    for f in all_files:
        fname_lower = f.name.lower()
        if 'feature' in fname_lower and f.suffix == '.csv' and feat_file is None:
            feat_file = f
        if 'label' in fname_lower and label_file is None:
            label_file = f
    
    return feat_file, label_file


def load_labels(labels_path: Path) -> Dict:
    """Load labels from JSON or CSV file."""
    if labels_path.suffix == '.json':
        with open(labels_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    elif labels_path.suffix == '.csv':
        # Parse CSV labels
        labels_dict = {}
        with open(labels_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get('sample_id') or row.get('id')
                if sample_id:
                    labels_dict[sample_id] = row
        return labels_dict
    else:
        raise ValueError(f"Unsupported label file format: {labels_path.suffix}")


def load_features_csv(features_path: Path) -> Dict[str, Dict[str, float]]:
    """Load features from CSV file."""
    features_dict = {}
    with open(features_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get('sample_id') or row.get('id')
            if sample_id:
                # Convert all values to float, skip non-numeric columns
                feat_row = {}
                for k, v in row.items():
                    if k not in ['sample_id', 'id']:
                        try:
                            feat_row[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                features_dict[sample_id] = feat_row
    return features_dict


def extract_system_label(entry: Dict) -> str:
    """Extract system-level label from entry.
    
    Supports:
    - entry['type'] = 'normal'/'fault' + entry['system_fault_class']
    - entry['system_label'] or entry['y_sys']
    
    Returns normalized label: '正常', '幅度失准', '频率失准', '参考电平失准'
    """
    # Check direct label fields
    if 'system_label' in entry:
        return str(entry['system_label'])
    if 'y_sys' in entry:
        label_val = entry['y_sys']
        if isinstance(label_val, (int, float)):
            # Map numeric to string
            mapping = {0: '正常', 1: '幅度失准', 2: '频率失准', 3: '参考电平失准'}
            return mapping.get(int(label_val), '正常')
        return str(label_val)
    
    # Parse from type + fault_class
    if entry.get('type') == 'normal':
        return '正常'
    elif entry.get('type') == 'fault':
        fault_cls = entry.get('system_fault_class', '')
        mapping = {
            'amp_error': '幅度失准',
            'freq_error': '频率失准',
            'ref_error': '参考电平失准',
        }
        return mapping.get(fault_cls, '正常')
    
    return '正常'


def extract_module_label(entry: Dict) -> Optional[int]:
    """Extract module-level label (single-label for now).
    
    Returns module ID (1-21) or None if not available.
    """
    if 'module_id' in entry:
        return int(entry['module_id'])
    if 'y_mod' in entry:
        mod_val = entry['y_mod']
        if isinstance(mod_val, (int, float)):
            return int(mod_val)
    if 'module' in entry:
        # Map module name to ID (simplified - would need full mapping)
        # For now, return None if it's a string
        if isinstance(entry['module'], (int, float)):
            return int(entry['module'])
    return None


def prepare_dataset(data_dir: Path, use_pool_features: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], List[str]]:
    """Load and prepare dataset with features and labels.
    
    Args:
        data_dir: Directory containing dataset files
        use_pool_features: If True, use/generate pool features for broader feature set
        
    Returns:
        (X, y_sys, y_mod, feature_names, sample_ids)
    """
    # Discover files
    features_path, labels_path = discover_dataset_files(data_dir)
    
    if not features_path or not labels_path:
        raise FileNotFoundError(
            f"Could not find features or labels in {data_dir}. "
            f"Expected features_brb.csv and labels.json"
        )
    
    print(f"Loading features from: {features_path}")
    print(f"Loading labels from: {labels_path}")
    
    # Load data
    labels_dict = load_labels(labels_path)
    features_dict = load_features_csv(features_path)
    
    # If using pool features and raw curves available, augment features
    if use_pool_features:
        raw_curves_dir = data_dir / "raw_curves"
        if raw_curves_dir.exists():
            from features.feature_pool import augment_features_with_pool
            for sample_id, feats in features_dict.items():
                curve_path = raw_curves_dir / f"{sample_id}.csv"
                features_dict[sample_id] = augment_features_with_pool(feats, curve_path)
        else:
            # Synthesize pool features from existing features
            from features.feature_pool import _synthesize_pool_from_base
            for sample_id, feats in features_dict.items():
                pool_feats = _synthesize_pool_from_base(feats)
                features_dict[sample_id] = {**feats, **pool_feats}
    
    # Align samples (only use samples with both features and labels)
    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    
    if not common_ids:
        raise ValueError("No common samples found between features and labels")
    
    print(f"Found {len(common_ids)} samples with both features and labels")
    
    # Build feature matrix and label vectors
    # First, determine feature names from first sample
    feature_names = list(features_dict[common_ids[0]].keys())
    n_features = len(feature_names)
    n_samples = len(common_ids)
    
    X = np.zeros((n_samples, n_features))
    y_sys_list = []
    y_mod_list = []
    
    for i, sample_id in enumerate(common_ids):
        # Features
        feat_dict = features_dict[sample_id]
        for j, fname in enumerate(feature_names):
            X[i, j] = feat_dict.get(fname, 0.0)
        
        # Labels
        label_entry = labels_dict[sample_id]
        y_sys_list.append(extract_system_label(label_entry))
        y_mod_val = extract_module_label(label_entry)
        y_mod_list.append(y_mod_val if y_mod_val is not None else -1)
    
    # Convert system labels to numeric
    unique_sys_labels = sorted(set(y_sys_list))
    sys_label_to_idx = {label: idx for idx, label in enumerate(unique_sys_labels)}
    y_sys = np.array([sys_label_to_idx[label] for label in y_sys_list])
    
    # Module labels (may have -1 for missing)
    y_mod = np.array(y_mod_list)
    if np.all(y_mod == -1):
        y_mod = None  # No module labels available
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"System labels: {unique_sys_labels}")
    print(f"System label distribution: {np.bincount(y_sys)}")
    if y_mod is not None:
        print(f"Module labels available: {np.sum(y_mod >= 0)} samples")
    
    return X, y_sys, y_mod, feature_names, common_ids


def stratified_split(X: np.ndarray, y: np.ndarray, 
                     train_size: float = 0.6, val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """Stratified train/val/test split.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
    """
    n_samples = len(X)
    n_classes = len(np.unique(y))
    
    # Create stratified splits
    indices = np.arange(n_samples)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_label in np.unique(y):
        class_indices = indices[y == class_label]
        n_class = len(class_indices)
        
        # Shuffle class indices
        rng = np.random.RandomState(random_state)
        rng.shuffle(class_indices)
        
        # Split
        n_train = max(1, int(n_class * train_size))
        n_val = max(1, int(n_class * val_size))
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train+n_val])
        test_indices.extend(class_indices[n_train+n_val:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    rng = np.random.RandomState(random_state)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return (
        X[train_indices], X[val_indices], X[test_indices],
        y[train_indices], y[val_indices], y[test_indices],
        train_indices, val_indices, test_indices
    )


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy."""
    return float(np.mean(y_true == y_pred))


def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Calculate macro F1 score."""
    f1_scores = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return float(np.mean(f1_scores))


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Calculate confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         output_path: Path, title: str):
    """Plot and save confusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved confusion matrix to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")


def plot_comparison_bar(results: List[Dict], output_dir: Path):
    """Plot comparison bar charts for rules, params, and inference time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        methods = [r['method'] for r in results]
        rules = [r.get('n_rules', 0) for r in results]
        params = [r.get('n_params', 0) for r in results]
        infer_ms = [r.get('infer_ms_per_sample', 0) for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Rules
        axes[0].bar(methods, rules, color='skyblue')
        axes[0].set_ylabel('Number of Rules')
        axes[0].set_title('Model Complexity: Rules')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Params
        axes[1].bar(methods, params, color='lightcoral')
        axes[1].set_ylabel('Number of Parameters')
        axes[1].set_title('Model Complexity: Parameters')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Inference time
        axes[2].bar(methods, infer_ms, color='lightgreen')
        axes[2].set_ylabel('Inference Time (ms/sample)')
        axes[2].set_title('Inference Efficiency')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = output_dir / "compare_barplot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison bar plot to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping bar plot")


def plot_small_sample_curve(small_sample_results: List[Dict], output_dir: Path):
    """Plot small-sample learning curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Group by method
        methods_data = {}
        for entry in small_sample_results:
            method = entry['method']
            if method not in methods_data:
                methods_data[method] = {'sizes': [], 'means': [], 'stds': []}
            methods_data[method]['sizes'].append(entry['train_size'])
            methods_data[method]['means'].append(entry['mean_acc'])
            methods_data[method]['stds'].append(entry['std_acc'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, data in methods_data.items():
            sizes = np.array(data['sizes'])
            means = np.array(data['means'])
            stds = np.array(data['stds'])
            
            ax.plot(sizes, means, marker='o', label=method)
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.2)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Small-Sample Adaptability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / "small_sample_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved small-sample curve to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping small-sample curve plot")


def plot_comprehensive_comparison(all_results: List[Dict], output_dir: Path):
    """Plot comprehensive comparison visualization combining multiple metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        methods = [r['method'] for r in all_results]
        accuracies = [r['sys_accuracy'] * 100 for r in all_results]  # Convert to percentage
        f1_scores = [r['sys_macro_f1'] * 100 for r in all_results]
        n_rules = [r['n_rules'] for r in all_results]
        n_params = [r['n_params'] for r in all_results]
        infer_ms = [r['infer_ms_per_sample'] for r in all_results]
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(methods, accuracies, color='steelblue', alpha=0.8)
        ax1.set_ylabel('System Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Classification Accuracy', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. F1-score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(methods, f1_scores, color='coral', alpha=0.8)
        ax2.set_ylabel('Macro F1-Score (%)', fontsize=11)
        ax2.set_title('(b) F1-Score Performance', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Rules count
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(methods, n_rules, color='mediumseagreen', alpha=0.8)
        ax3.set_ylabel('Number of Rules', fontsize=11)
        ax3.set_title('(c) Model Complexity (Rules)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, n_rules):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 4. Parameters count
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(methods, n_params, color='mediumpurple', alpha=0.8)
        ax4.set_ylabel('Number of Parameters', fontsize=11)
        ax4.set_title('(d) Parameter Count', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars4, n_params):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 5. Inference time
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(methods, infer_ms, color='lightgreen', alpha=0.8)
        ax5.set_ylabel('Inference Time (ms/sample)', fontsize=11)
        ax5.set_title('(e) Inference Efficiency', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars5, infer_ms):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Accuracy vs Complexity scatter
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(n_rules, accuracies, s=100, c=infer_ms, 
                             cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
        ax6.set_xlabel('Number of Rules', fontsize=11)
        ax6.set_ylabel('Accuracy (%)', fontsize=11)
        ax6.set_title('(f) Accuracy vs Complexity', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        # Add method labels
        for i, method in enumerate(methods):
            ax6.annotate(method, (n_rules[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        # Add colorbar for inference time
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Infer Time (ms)', fontsize=9)
        
        plt.suptitle('Comprehensive Method Comparison Results', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        output_path = output_dir / "comparison_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comprehensive comparison plot to: {output_path}")
    except ImportError as e:
        print(f"matplotlib not available, skipping comprehensive plot: {e}")
    except Exception as e:
        print(f"Error creating comprehensive plot: {e}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_method(method, X_train, y_sys_train, y_mod_train, 
                   X_test, y_sys_test, y_mod_test, 
                   feature_names, n_sys_classes):
    """Evaluate a single method."""
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method.name}")
    print(f"{'='*60}")
    
    # Fit
    start_fit = time.time()
    meta_train = {'feature_names': feature_names}
    method.fit(X_train, y_sys_train, y_mod_train, meta_train)
    fit_time = time.time() - start_fit
    
    # Predict
    start_infer = time.time()
    predictions = method.predict(X_test, meta={'feature_names': feature_names})
    infer_time_total = time.time() - start_infer
    infer_time_per_sample = (infer_time_total / len(X_test)) * 1000  # ms
    
    # Extract predictions
    y_sys_pred = predictions['system_pred']
    y_mod_pred = predictions.get('module_pred', None)
    
    # System-level metrics
    sys_acc = calculate_accuracy(y_sys_test, y_sys_pred)
    sys_f1 = calculate_macro_f1(y_sys_test, y_sys_pred, n_sys_classes)
    sys_cm = calculate_confusion_matrix(y_sys_test, y_sys_pred, n_sys_classes)
    
    # Module-level metrics (if available)
    mod_acc = None
    if y_mod_pred is not None and y_mod_test is not None:
        valid_mask = y_mod_test >= 0
        if np.sum(valid_mask) > 0:
            mod_acc = calculate_accuracy(y_mod_test[valid_mask], y_mod_pred[valid_mask])
    
    # Complexity metrics
    complexity = method.complexity()
    
    # Combine all results
    results = {
        'method': method.name,
        'sys_accuracy': sys_acc,
        'sys_macro_f1': sys_f1,
        'mod_top1_accuracy': mod_acc if mod_acc is not None else 0.0,
        'fit_time_sec': fit_time,
        'infer_ms_per_sample': infer_time_per_sample,
        'n_rules': complexity.get('n_rules', 0),
        'n_params': complexity.get('n_params', 0),
        'n_features_used': complexity.get('n_features_used', 0),
        'confusion_matrix': sys_cm,
    }
    
    # Update with method metadata
    if 'meta' in predictions:
        pred_meta = predictions['meta']
        results['features_used'] = pred_meta.get('features_used', [])
    
    print(f"System Accuracy: {sys_acc:.4f}")
    print(f"System Macro-F1: {sys_f1:.4f}")
    if mod_acc is not None:
        print(f"Module Top-1 Accuracy: {mod_acc:.4f}")
    print(f"Fit Time: {fit_time:.2f} sec")
    print(f"Inference Time: {infer_time_per_sample:.4f} ms/sample")
    print(f"Rules: {results['n_rules']}, Params: {results['n_params']}, Features: {results['n_features_used']}")
    
    return results


def save_diagnostic_outputs(output_dir: Path, y_sys: np.ndarray, y_sys_test: np.ndarray,
                           all_results: List[Dict], class_names: List[str]):
    """Save diagnostic outputs to help debug accuracy issues.
    
    Section A of the fix: Generate diagnostic files to understand why ours 
    accuracy is below majority-class baseline.
    """
    # A1: Dataset class distribution and majority baseline
    unique, counts = np.unique(y_sys, return_counts=True)
    total = len(y_sys)
    majority_class = unique[np.argmax(counts)]
    majority_count = np.max(counts)
    majority_baseline_acc = majority_count / total
    
    distribution_data = []
    for i, (cls, cnt) in enumerate(zip(unique, counts)):
        cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
        distribution_data.append({
            'class_id': int(cls),
            'class_name': cls_name,
            'count': int(cnt),
            'percentage': float(cnt / total * 100),
        })
    
    distribution_data.append({
        'class_id': -1,
        'class_name': 'MAJORITY_BASELINE',
        'count': int(majority_count),
        'percentage': float(majority_baseline_acc * 100),
    })
    
    dist_path = output_dir / "dataset_class_distribution.csv"
    with open(dist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name', 'count', 'percentage'])
        writer.writeheader()
        writer.writerows(distribution_data)
    print(f"Saved class distribution to: {dist_path}")
    print(f"  Majority class baseline accuracy: {majority_baseline_acc:.4f} ({majority_baseline_acc*100:.2f}%)")
    
    # A2 & A3: Per-method diagnostics (focus on "ours")
    for result in all_results:
        method_name = result['method']
        cm = result.get('confusion_matrix')
        
        if cm is None:
            continue
        
        # Per-class metrics
        n_classes = len(cm)
        per_class_metrics = []
        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = int(cm[c, :].sum())
            
            cls_name = class_names[c] if c < len(class_names) else f"Class_{c}"
            per_class_metrics.append({
                'class_id': c,
                'class_name': cls_name,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': support,
            })
        
        metrics_path = output_dir / f"{method_name}_per_class_metrics.csv"
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name', 'precision', 'recall', 'f1', 'support'])
            writer.writeheader()
            writer.writerows(per_class_metrics)
        print(f"Saved per-class metrics for {method_name} to: {metrics_path}")
        
        # Prediction distribution
        pred_counts = cm.sum(axis=0)
        pred_dist = []
        for c in range(n_classes):
            cls_name = class_names[c] if c < len(class_names) else f"Class_{c}"
            pred_dist.append({
                'class_id': c,
                'class_name': cls_name,
                'pred_count': int(pred_counts[c]),
                'pred_percentage': float(pred_counts[c] / pred_counts.sum() * 100) if pred_counts.sum() > 0 else 0.0,
            })
        
        pred_path = output_dir / f"{method_name}_pred_distribution.csv"
        with open(pred_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_id', 'class_name', 'pred_count', 'pred_percentage'])
            writer.writeheader()
            writer.writerows(pred_dist)
        
        # Save confusion matrix
        cm_path = output_dir / f"{method_name}_confusion_matrix.csv"
        cm_df_data = []
        for i in range(n_classes):
            row = {'true_class': class_names[i] if i < len(class_names) else f"Class_{i}"}
            for j in range(n_classes):
                row[class_names[j] if j < len(class_names) else f"Class_{j}"] = int(cm[i, j])
            cm_df_data.append(row)
        
        with open(cm_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['true_class'] + [class_names[j] if j < len(class_names) else f"Class_{j}" for j in range(n_classes)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cm_df_data)
        print(f"Saved confusion matrix for {method_name} to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive method comparison")
    parser.add_argument('--data_dir', default='Output/sim_spectrum', 
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', default='Output/sim_spectrum',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_size', type=float, default=0.6, help='Training set ratio')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--small_sample', action='store_true', 
                       help='Run small-sample adaptability experiments')
    parser.add_argument('--save_diagnostics', action='store_true',
                       help='Save diagnostic outputs for debugging accuracy issues (default: enabled)')
    args = parser.parse_args()
    
    # Setup
    set_global_seed(args.seed)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else ROOT / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    X, y_sys, y_mod, feature_names, sample_ids = prepare_dataset(data_dir, use_pool_features=True)
    
    n_sys_classes = len(np.unique(y_sys))
    
    # Split dataset
    print("\n" + "="*60)
    print("Splitting dataset...")
    print("="*60)
    X_train, X_val, X_test, y_sys_train, y_sys_val, y_sys_test, train_idx, val_idx, test_idx = \
        stratified_split(X, y_sys, args.train_size, args.val_size, args.seed)
    
    y_mod_train = y_mod[train_idx] if y_mod is not None else None
    y_mod_val = y_mod[val_idx] if y_mod is not None else None
    y_mod_test = y_mod[test_idx] if y_mod is not None else None
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Import methods (will be implemented)
    print("\n" + "="*60)
    print("Importing methods...")
    print("="*60)
    
    # For now, create placeholder - will implement actual methods
    from methods.ours_adapter import OursAdapter
    from methods.hcf_adapter import HCFAdapter
    from methods.aifd_adapter import AIFDAdapter
    from methods.brb_p_adapter import BRBPAdapter
    from methods.brb_mu_adapter import BRBMUAdapter
    from methods.dbrb_adapter import DBRBAdapter
    from methods.a_ibrb_adapter import AIBRBAdapter
    from methods.fast_brb_adapter import FastBRBAdapter
    
    methods = [
        OursAdapter(),
        HCFAdapter(),
        AIFDAdapter(),
        BRBPAdapter(),
        BRBMUAdapter(),
        DBRBAdapter(),
        AIBRBAdapter(),
        FastBRBAdapter(),
    ]
    
    # Evaluate all methods
    all_results = []
    for method in methods:
        try:
            results = evaluate_method(
                method, X_train, y_sys_train, y_mod_train,
                X_test, y_sys_test, y_mod_test,
                feature_names, n_sys_classes
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {method.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save comparison table
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    comparison_path = output_dir / "comparison_table.csv"
    with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'sys_accuracy', 'sys_macro_f1', 'mod_top1_accuracy',
                     'fit_time_sec', 'infer_ms_per_sample', 
                     'n_rules', 'n_params', 'n_features_used']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            row = {k: result[k] for k in fieldnames if k in result}
            writer.writerow(row)
    print(f"Saved comparison table to: {comparison_path}")
    
    # Plot confusion matrices
    sys_label_names = ['Normal', 'Amp', 'Freq', 'Ref'][:n_sys_classes]
    for result in all_results:
        cm_path = output_dir / f"confusion_matrix_{result['method']}.png"
        plot_confusion_matrix(
            result['confusion_matrix'], 
            sys_label_names,
            cm_path,
            f"System-Level Confusion Matrix: {result['method']}"
        )
    
    # Plot comparison bars
    plot_comparison_bar(all_results, output_dir)
    
    # Plot comprehensive comparison
    plot_comprehensive_comparison(all_results, output_dir)
    
    # Save diagnostic outputs (Section A of accuracy fix) - always run by default
    print("\n" + "="*60)
    print("Saving diagnostic outputs...")
    print("="*60)
    save_diagnostic_outputs(output_dir, y_sys, y_sys_test, all_results, sys_label_names)
    
    # Small-sample experiments
    if args.small_sample:
        print("\n" + "="*60)
        print("Running small-sample adaptability experiments...")
        print("="*60)
        
        train_sizes = [5, 10, 20, 30]
        n_repeats = 5
        small_sample_results = []
        
        for train_size in train_sizes:
            if train_size > len(X_train):
                print(f"Skipping train_size={train_size} (exceeds available training data)")
                continue
            
            print(f"\nTrain size: {train_size}")
            
            for method in methods:
                method_accs = []
                
                for rep in range(n_repeats):
                    # Sample subset
                    rep_seed = args.seed + rep
                    rng = np.random.RandomState(rep_seed)
                    indices = rng.choice(len(X_train), size=train_size, replace=False)
                    X_small = X_train[indices]
                    y_small = y_sys_train[indices]
                    
                    # Train and evaluate
                    try:
                        method_copy = method.__class__()  # Create fresh instance
                        method_copy.fit(X_small, y_small, None, {'feature_names': feature_names})
                        pred = method_copy.predict(X_test, {'feature_names': feature_names})
                        acc = calculate_accuracy(y_sys_test, pred['system_pred'])
                        method_accs.append(acc)
                    except Exception as e:
                        print(f"Error in small-sample experiment for {method.name}: {e}")
                        method_accs.append(0.0)
                
                mean_acc = np.mean(method_accs)
                std_acc = np.std(method_accs)
                small_sample_results.append({
                    'method': method.name,
                    'train_size': train_size,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                })
                print(f"  {method.name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Save small-sample results
        small_sample_path = output_dir / "small_sample_curve.csv"
        with open(small_sample_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['method', 'train_size', 'mean_acc', 'std_acc']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(small_sample_results)
        print(f"Saved small-sample results to: {small_sample_path}")
        
        # Plot small-sample curve
        plot_small_sample_curve(small_sample_results, output_dir)
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == '__main__':
    main()
