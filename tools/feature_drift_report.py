#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate feature drift report for current dataset (train split).

Outputs:
- Output/compare_methods/feature_stats_train.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SYS_LABEL_ORDER = ['正常', '幅度失准', '频率失准', '参考电平失准']
LEAK_PREFIXES = ("sys_", "label", "target", "gt_", "y_", "truth", "class_")
LEAK_SUBSTRINGS = ("label", "target", "truth")


def detect_leakage_columns(feature_names: List[str]) -> List[str]:
    suspicious = []
    for name in feature_names:
        lower = name.lower()
        if lower.startswith(LEAK_PREFIXES):
            suspicious.append(name)
            continue
        if any(sub in lower for sub in LEAK_SUBSTRINGS):
            suspicious.append(name)
    return sorted(set(suspicious))


def load_labels(labels_path: Path) -> Dict:
    if labels_path.suffix == '.json':
        with open(labels_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    raise ValueError(f"Unsupported label file format: {labels_path.suffix}")


def extract_system_label(entry: Dict) -> str:
    if entry.get('type') == 'normal':
        return '正常'
    if entry.get('type') == 'fault':
        mapping = {
            'amp_error': '幅度失准',
            'freq_error': '频率失准',
            'ref_error': '参考电平失准',
        }
        return mapping.get(entry.get('system_fault_class', ''), '正常')
    return '正常'


def read_feature_rows(features_path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    features_dict: Dict[str, Dict[str, float]] = {}
    with features_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feature_names = [name for name in fieldnames if name not in ("sample_id", "id")]
        for row in reader:
            sample_id = row.get('sample_id') or row.get('id')
            if not sample_id:
                continue
            feat_row: Dict[str, float] = {}
            for k in feature_names:
                v = row.get(k, "")
                try:
                    feat_row[k] = float(v)
                except (ValueError, TypeError):
                    feat_row[k] = np.nan
            features_dict[sample_id] = feat_row
    return feature_names, features_dict


def stratified_split_indices(y: np.ndarray, train_size: float, val_size: float, seed: int) -> np.ndarray:
    indices = np.arange(len(y))
    train_indices = []
    rng = np.random.RandomState(seed)
    for class_label in np.unique(y):
        class_indices = indices[y == class_label]
        rng.shuffle(class_indices)
        n_train = max(1, int(len(class_indices) * train_size))
        train_indices.extend(class_indices[:n_train])
    return np.array(train_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature drift report (train split)")
    parser.add_argument("--features", default="Output/sim_spectrum/features_brb.csv")
    parser.add_argument("--labels", default="Output/sim_spectrum/labels.json")
    parser.add_argument("--output_dir", default="Output/compare_methods")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--train_size", type=float, default=0.6)
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists() or not labels_path.exists():
        raise SystemExit("[ERROR] features or labels file missing.")

    feature_names, features_dict = read_feature_rows(features_path)
    labels_dict = load_labels(labels_path)

    common_ids = sorted(set(features_dict.keys()) & set(labels_dict.keys()))
    if not common_ids:
        raise SystemExit("[ERROR] No common sample IDs between features and labels.")

    leak_columns = detect_leakage_columns(feature_names)
    filtered_features = [f for f in feature_names if f not in leak_columns]

    X = np.array([[features_dict[sid].get(f, np.nan) for f in filtered_features] for sid in common_ids])
    y_labels = [extract_system_label(labels_dict[sid]) for sid in common_ids]
    label_to_idx = {label: idx for idx, label in enumerate(SYS_LABEL_ORDER)}
    y = np.array([label_to_idx.get(lbl, 0) for lbl in y_labels])

    train_indices = stratified_split_indices(y, args.train_size, args.val_size, args.seed)
    X_train = X[train_indices]

    rows = []
    for j, fname in enumerate(filtered_features):
        col = X_train[:, j]
        nan_mask = np.isnan(col)
        valid = col[~nan_mask]
        missing_rate = float(np.mean(nan_mask))
        zero_rate = float(np.mean(valid == 0.0)) if valid.size else 1.0
        if valid.size == 0:
            stats = {
                "feature": fname,
                "mean": "",
                "std": "",
                "min": "",
                "p5": "",
                "p50": "",
                "p95": "",
                "max": "",
                "missing_rate": missing_rate,
                "zero_rate": zero_rate,
                "flag": "all_missing",
            }
        else:
            stats = {
                "feature": fname,
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "p5": float(np.percentile(valid, 5)),
                "p50": float(np.percentile(valid, 50)),
                "p95": float(np.percentile(valid, 95)),
                "max": float(np.max(valid)),
                "missing_rate": missing_rate,
                "zero_rate": zero_rate,
                "flag": "",
            }
            if stats["std"] <= 1e-6:
                stats["flag"] = "std_near_zero"
        rows.append(stats)

    output_path = output_dir / "feature_stats_train.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "feature",
            "mean",
            "std",
            "min",
            "p5",
            "p50",
            "p95",
            "max",
            "missing_rate",
            "zero_rate",
            "flag",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved feature drift report to: {output_path}")
    if leak_columns:
        print(f"[WARN] Leak-like columns excluded from report: {leak_columns}")


if __name__ == "__main__":
    main()
