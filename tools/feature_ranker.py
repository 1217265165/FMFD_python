#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Ranker Tool (XGBoost-based)
===================================

This tool uses XGBoost to rank features by importance for the staged
injection framework. XGBoost is ONLY used for feature ranking/grouping,
NOT for final classification.

Output: Sorted feature importance list that can be used to update
the feature_injection_plan.yaml stage assignments.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Suppress only specific warnings related to XGBoost
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


def rank_features_xgb(
    X: np.ndarray,
    y: np.ndarray,
    feature_list: List[str],
    n_estimators: int = 100,
    max_depth: int = 4,
    random_state: int = 42
) -> List[Tuple[str, float]]:
    """Rank features by XGBoost importance.
    
    Uses XGBoost to compute feature importances for ranking purposes.
    This is used ONLY for feature selection/grouping, NOT classification.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    feature_list : List[str]
        List of feature names corresponding to columns of X.
    n_estimators : int
        Number of trees in XGBoost.
    max_depth : int
        Maximum tree depth.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    List[Tuple[str, float]]
        List of (feature_name, importance) tuples, sorted by importance descending.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        # Fallback to sklearn RandomForest if XGBoost not available
        from sklearn.ensemble import RandomForestClassifier
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X, y)
        importances = clf.feature_importances_
        
        result = list(zip(feature_list, importances))
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    # Use XGBoost
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    
    result = list(zip(feature_list, importances))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def rank_features_for_fault_type(
    X: np.ndarray,
    y: np.ndarray,
    feature_list: List[str],
    fault_label: int,
    n_estimators: int = 100,
    max_depth: int = 4,
    random_state: int = 42
) -> List[Tuple[str, float]]:
    """Rank features for detecting a specific fault type (binary classification).
    
    Creates a binary classification problem: fault_label vs others.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Multi-class labels.
    feature_list : List[str]
        Feature names.
    fault_label : int
        Target fault class to rank features for.
        
    Returns
    -------
    List[Tuple[str, float]]
        Sorted feature importances for this fault type.
    """
    # Create binary labels
    y_binary = (y == fault_label).astype(int)
    
    return rank_features_xgb(X, y_binary, feature_list, n_estimators, max_depth, random_state)


def suggest_stage_assignment(
    feature_importances: List[Tuple[str, float]],
    n_stage1: int = 3,
    n_stage2: int = 3,
    n_stage3: int = 4
) -> Dict[str, List[str]]:
    """Suggest stage assignment based on feature importance ranking.
    
    Parameters
    ----------
    feature_importances : List[Tuple[str, float]]
        Sorted feature importances.
    n_stage1 : int
        Number of features for stage1 (most important).
    n_stage2 : int
        Number of features for stage2.
    n_stage3 : int
        Number of features for stage3.
        
    Returns
    -------
    dict
        Stage assignments: {stage1: [...], stage2: [...], stage3: [...]}
    """
    features = [f for f, _ in feature_importances]
    
    total_needed = n_stage1 + n_stage2 + n_stage3
    available = min(len(features), total_needed)
    
    stage1 = features[:min(n_stage1, available)]
    stage2 = features[n_stage1:min(n_stage1 + n_stage2, available)]
    stage3 = features[n_stage1 + n_stage2:min(total_needed, available)]
    
    return {
        'stage1': stage1,
        'stage2': stage2,
        'stage3': stage3,
    }


def analyze_feature_importance_per_branch(
    X: np.ndarray,
    y: np.ndarray,
    feature_list: List[str],
    label_mapping: Optional[Dict[int, str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """Analyze feature importance for each fault branch.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels (0=Normal, 1=Amp, 2=Freq, 3=Ref).
    feature_list : List[str]
        Feature names.
    label_mapping : dict, optional
        Mapping from label index to name.
        
    Returns
    -------
    dict
        Feature importances per branch:
        {
            'amp': [(feat, imp), ...],
            'freq': [(feat, imp), ...],
            'ref': [(feat, imp), ...],
            'overall': [(feat, imp), ...],
        }
    """
    if label_mapping is None:
        label_mapping = {0: 'normal', 1: 'amp', 2: 'freq', 3: 'ref'}
    
    result = {}
    
    # Overall importance (multi-class)
    result['overall'] = rank_features_xgb(X, y, feature_list)
    
    # Per-branch importance (binary: this fault vs others)
    for label_idx, branch_name in label_mapping.items():
        if branch_name != 'normal':
            result[branch_name] = rank_features_for_fault_type(
                X, y, feature_list, label_idx
            )
    
    return result


def generate_stage_recommendations(
    X: np.ndarray,
    y: np.ndarray,
    feature_list: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """Generate complete stage recommendations for all branches.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    feature_list : List[str]
        Feature names.
        
    Returns
    -------
    dict
        Recommended stage assignments for amp, freq, ref branches.
    """
    importance_per_branch = analyze_feature_importance_per_branch(X, y, feature_list)
    
    recommendations = {}
    
    # Amplitude branch
    if 'amp' in importance_per_branch:
        recommendations['amp'] = suggest_stage_assignment(
            importance_per_branch['amp'], n_stage1=3, n_stage2=3, n_stage3=4
        )
    
    # Frequency branch
    if 'freq' in importance_per_branch:
        recommendations['freq'] = suggest_stage_assignment(
            importance_per_branch['freq'], n_stage1=2, n_stage2=2, n_stage3=2
        )
    
    # Reference level branch
    if 'ref' in importance_per_branch:
        recommendations['ref'] = suggest_stage_assignment(
            importance_per_branch['ref'], n_stage1=2, n_stage2=2, n_stage3=2
        )
    
    return recommendations


def update_yaml_config(
    config_path: Path,
    recommendations: Dict[str, Dict[str, List[str]]],
    output_path: Optional[Path] = None
) -> str:
    """Update YAML configuration with new stage assignments.
    
    Parameters
    ----------
    config_path : Path
        Path to existing feature_injection_plan.yaml.
    recommendations : dict
        Stage recommendations from generate_stage_recommendations().
    output_path : Path, optional
        Output path. If None, returns YAML string without writing.
        
    Returns
    -------
    str
        Updated YAML content.
    """
    try:
        import yaml
    except ImportError:
        # Fallback: return JSON representation
        return json.dumps(recommendations, indent=2)
    
    # Load existing config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {'system': {}}
    
    # Update stage assignments
    for branch_name, stages in recommendations.items():
        if branch_name in config.get('system', {}):
            for stage_name, features in stages.items():
                config['system'][branch_name][stage_name] = features
    
    # Convert to YAML string
    yaml_str = yaml.dump(config, allow_unicode=True, default_flow_style=False)
    
    # Write if output path specified
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_str)
    
    return yaml_str


def print_importance_report(
    importance_per_branch: Dict[str, List[Tuple[str, float]]],
    top_k: int = 10
) -> None:
    """Print a formatted importance report.
    
    Parameters
    ----------
    importance_per_branch : dict
        Output from analyze_feature_importance_per_branch().
    top_k : int
        Number of top features to show per branch.
    """
    print("=" * 60)
    print("Feature Importance Report (XGBoost)")
    print("=" * 60)
    
    for branch_name, importances in importance_per_branch.items():
        print(f"\n{branch_name.upper()} Branch:")
        print("-" * 40)
        for i, (feat, imp) in enumerate(importances[:top_k]):
            print(f"  {i+1:2d}. {feat:6s}: {imp:.4f}")


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 22
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 4, n_samples)
    feature_list = [f'X{i+1}' for i in range(n_features)]
    
    print("Running XGBoost feature ranking demo...")
    
    # Get importance per branch
    importance_per_branch = analyze_feature_importance_per_branch(X, y, feature_list)
    
    # Print report
    print_importance_report(importance_per_branch)
    
    # Generate recommendations
    recommendations = generate_stage_recommendations(X, y, feature_list)
    
    print("\n" + "=" * 60)
    print("Stage Assignment Recommendations:")
    print("=" * 60)
    for branch, stages in recommendations.items():
        print(f"\n{branch.upper()}:")
        for stage, features in stages.items():
            print(f"  {stage}: {features}")
