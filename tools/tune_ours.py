#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from BRB.aggregator import set_calibration_override
from methods.ours_adapter import OursAdapter

CLASS_ORDER = ["ref_error", "amp_error", "normal", "freq_error"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def _load_features(features_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(features_path, encoding="utf-8-sig")
    if "sample_id" not in df.columns:
        raise KeyError("features_brb.csv must contain sample_id column")
    feature_cols = [c for c in df.columns if c != "sample_id"]
    return df, feature_cols


def _load_labels(labels_path: Path) -> Dict[str, Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align_dataset(df: pd.DataFrame, labels: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    y = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in labels:
            continue
        sys_class = labels[sample_id].get("system_fault_class", "normal") or "normal"
        sys_class = sys_class if sys_class in CLASS_TO_IDX else "normal"
        rows.append(row)
        y.append(CLASS_TO_IDX[sys_class])
    if not rows:
        raise ValueError("No aligned samples between features and labels")
    aligned = pd.DataFrame(rows)
    features = aligned.drop(columns=["sample_id"]).to_numpy(dtype=float)
    return features, np.array(y, dtype=int)


def _train_val_split(n_samples: int, seed: int, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split = int(n_samples * (1.0 - val_ratio))
    return indices[:split], indices[split:]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, object]:
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = []
    f1s = []
    for i in range(n_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1})
        f1s.append(f1)

    accuracy = float(np.mean(y_true == y_pred))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "confusion": confusion,
        "per_class": per_class,
    }


def _evaluate(config: Dict, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, object]:
    set_calibration_override(config)
    model = OursAdapter(calibration_override=config)
    result = model.predict(X, meta={"feature_names": feature_names})
    y_pred = result["system_pred"]
    metrics = _metrics(y, y_pred, len(CLASS_ORDER))
    return metrics


def _search_configs(base_config: Dict, n_random: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    grid = []
    for alpha in [1.6, 2.0, 2.4]:
        for beta_amp in [0.2, 0.4]:
            for beta_freq in [0.4, 0.6]:
                for beta_ref in [0.4, 0.6]:
                    for amp_w in [0.9, 1.0, 1.1]:
                        for freq_w in [0.9, 1.0, 1.1]:
                            for ref_w in [0.9, 1.0, 1.1]:
                                grid.append(
                                    {
                                        **base_config,
                                        "alpha": alpha,
                                        "beta_amp": beta_amp,
                                        "beta_freq": beta_freq,
                                        "beta_ref": beta_ref,
                                        "branch_weights": {
                                            "amp": amp_w,
                                            "freq": freq_w,
                                            "ref": ref_w,
                                        },
                                        "gate_z_hi": 3.0,
                                        "gate_z_mid": 2.0,
                                    }
                                )

    random_configs = []
    for _ in range(n_random):
        random_configs.append(
            {
                **base_config,
                "alpha": rng.uniform(1.4, 2.6),
                "beta_amp": rng.uniform(0.15, 0.6),
                "beta_freq": rng.uniform(0.3, 0.8),
                "beta_ref": rng.uniform(0.3, 0.8),
                "branch_weights": {
                    "amp": rng.uniform(0.8, 1.2),
                    "freq": rng.uniform(0.8, 1.2),
                    "ref": rng.uniform(0.8, 1.2),
                },
                "gate_z_hi": rng.uniform(2.5, 3.5),
                "gate_z_mid": rng.uniform(1.5, 2.5),
                "ref_boost_on_offset": rng.uniform(0.2, 0.5),
                "amp_suppress_on_offset": rng.uniform(0.2, 0.5),
                "freq_boost_on_shift": rng.uniform(0.2, 0.5),
                "amp_boost_on_ripple": rng.uniform(0.2, 0.5),
            }
        )

    return grid + random_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ours calibration parameters")
    parser.add_argument("--features", required=True, help="Path to features_brb.csv")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--out", default="Output/ours_best_config.json", help="Output config JSON")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_trials", type=int, default=30)
    args = parser.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, feature_cols = _load_features(features_path)
    labels = _load_labels(labels_path)
    X, y = _align_dataset(df, labels)

    train_idx, val_idx = _train_val_split(len(X), seed=args.seed, val_ratio=args.val_ratio)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    base_config: Dict[str, object] = {
        "alpha": 2.0,
        "beta_amp": 0.3,
        "beta_freq": 0.5,
        "beta_ref": 0.5,
        "branch_weights": {"amp": 1.0, "freq": 1.0, "ref": 1.0},
        "gate_z_hi": 3.0,
        "gate_z_mid": 2.0,
        "ref_boost_on_offset": 0.35,
        "amp_suppress_on_offset": 0.35,
        "freq_boost_on_shift": 0.35,
        "amp_boost_on_ripple": 0.30,
    }

    baseline_metrics = _evaluate(base_config, X_val, y_val, feature_cols)
    baseline_f1 = baseline_metrics["macro_f1"]

    best = {
        "config": base_config,
        "metrics": baseline_metrics,
    }

    configs = _search_configs(base_config, args.random_trials, args.seed)
    for cfg in configs:
        metrics = _evaluate(cfg, X_val, y_val, feature_cols)
        if metrics["macro_f1"] + 1e-6 < baseline_f1 - 0.01:
            continue
        if metrics["accuracy"] > best["metrics"]["accuracy"]:
            best = {"config": cfg, "metrics": metrics}

    # Evaluate best on full dataset
    final_metrics = _evaluate(best["config"], X, y, feature_cols)

    result = {
        **best["config"],
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "confusion_matrix": final_metrics["confusion"].tolist(),
            "class_order": CLASS_ORDER,
        },
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Best config saved:", out_path)
    print("Accuracy:", final_metrics["accuracy"])
    print("Macro-F1:", final_metrics["macro_f1"])
    print("Confusion matrix (rows=true, cols=pred):")
    print(final_metrics["confusion"])


if __name__ == "__main__":
    main()
