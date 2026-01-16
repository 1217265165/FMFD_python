#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BRB.aggregator import set_calibration_override
from methods.ours_adapter import OursAdapter
from pipelines.compare_methods import stratified_split, set_global_seed

CLASS_ORDER = ["ref_error", "amp_error", "normal", "freq_error"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}

SEED = 42
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2


def _load_features(features_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(features_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        feature_cols = []
        for row in reader:
            if not feature_cols:
                feature_cols = [c for c in row.keys() if c != "sample_id"]
            rows.append(row)
    return rows, feature_cols


def _load_labels(labels_path: Path) -> Dict[str, Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align_dataset(rows: List[Dict[str, str]], labels: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    aligned_rows = []
    y = []
    sample_ids = []
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in labels:
            continue
        sys_class = labels[sample_id].get("system_fault_class", "normal") or "normal"
        sys_class = sys_class if sys_class in CLASS_TO_IDX else "normal"
        aligned_rows.append(row)
        y.append(CLASS_TO_IDX[sys_class])
        sample_ids.append(sample_id)
    if not aligned_rows:
        raise ValueError("No aligned samples between features and labels")
    feature_cols = [k for k in aligned_rows[0].keys() if k != "sample_id"]
    features = np.array(
        [[float(row.get(k, 0.0)) for k in feature_cols] for row in aligned_rows],
        dtype=float,
    )
    return features, np.array(y, dtype=int), sample_ids


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


def _evaluate(config: Dict, X: np.ndarray, y: np.ndarray, feature_names: List[str], normal_quantiles: Dict) -> Dict[str, object]:
    if normal_quantiles:
        config = {**config, "normal_quantiles": normal_quantiles}
    set_calibration_override(config)
    model = OursAdapter(calibration_override=config)
    result = model.predict(X, meta={"feature_names": feature_names})
    y_pred = result["system_pred"]
    return _metrics(y, y_pred, len(CLASS_ORDER))


def _objective(metrics: Dict[str, object]) -> float:
    accuracy = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    amp_recall = metrics["per_class"][CLASS_TO_IDX["amp_error"]]["recall"]
    ref_precision = metrics["per_class"][CLASS_TO_IDX["ref_error"]]["precision"]

    penalty = 0.0
    if amp_recall < 0.55:
        penalty += 1.0
    if ref_precision < 0.65:
        penalty += 1.0

    return accuracy + 0.3 * macro_f1 - penalty


def _build_coarse_grid(max_samples: int = 3500) -> List[Dict]:
    beta_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    normal_prior_values = [0.5, 1.0, 1.5, 2.0]
    weight_values = [0.0, 0.5, 1.0, 1.5]
    suppress_values = [0.3, 0.5, 0.7, 1.0]
    alpha_values = [1.0, 1.5, 2.0, 3.0]
    margin_values = [0.10, 0.15, 0.20]
    pmax_values = [0.50, 0.55, 0.60]
    t_low_values = [0.4, 0.55, 0.7]
    t_high_values = [0.6, 0.75, 0.85]

    all_combos = list(itertools.product(
        beta_values,
        beta_values,
        beta_values,
        normal_prior_values,
        weight_values,
        weight_values,
        weight_values,
        suppress_values,
        suppress_values,
        alpha_values,
        margin_values,
        pmax_values,
        t_low_values,
        t_high_values,
    ))

    rng = np.random.RandomState(SEED)
    rng.shuffle(all_combos)
    selected = all_combos[:max_samples]

    configs = []
    for combo in selected:
        (
            beta_amp,
            beta_freq,
            beta_ref,
            normal_prior_k,
            w_offset_to_ref,
            w_ripple_to_amp,
            w_shift_to_freq,
            ref_suppress_multiplier,
            freq_suppress_multiplier,
            alpha_base,
            margin_threshold,
            pmax_threshold,
            t_low,
            t_high,
        ) = combo

        configs.append(
            {
                "alpha": alpha_base,
                "alpha_base": alpha_base,
                "beta_amp": beta_amp,
                "beta_freq": beta_freq,
                "beta_ref": beta_ref,
                "normal_prior_k": normal_prior_k,
                "w_offset_to_ref": w_offset_to_ref,
                "w_ripple_to_amp": w_ripple_to_amp,
                "w_shift_to_freq": w_shift_to_freq,
                "ref_suppress_multiplier": ref_suppress_multiplier,
                "freq_suppress_multiplier": freq_suppress_multiplier,
                "margin_threshold": margin_threshold,
                "pmax_threshold": pmax_threshold,
                "T_low": t_low,
                "T_high": t_high,
            }
        )

    return configs


def _local_search(base_config: Dict, n_trials: int, rng: np.random.RandomState) -> List[Dict]:
    beta_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    normal_prior_values = [0.5, 1.0, 1.5, 2.0]
    weight_values = [0.0, 0.5, 1.0, 1.5]
    suppress_values = [0.3, 0.5, 0.7, 1.0]
    alpha_values = [1.0, 1.5, 2.0, 3.0]
    margin_values = [0.10, 0.15, 0.20]
    pmax_values = [0.50, 0.55, 0.60]
    t_low_values = [0.4, 0.55, 0.7]
    t_high_values = [0.6, 0.75, 0.85]

    configs = []
    for _ in range(n_trials):
        configs.append(
            {
                "alpha": rng.choice(alpha_values),
                "alpha_base": rng.choice(alpha_values),
                "beta_amp": rng.choice(beta_values),
                "beta_freq": rng.choice(beta_values),
                "beta_ref": rng.choice(beta_values),
                "normal_prior_k": rng.choice(normal_prior_values),
                "w_offset_to_ref": rng.choice(weight_values),
                "w_ripple_to_amp": rng.choice(weight_values),
                "w_shift_to_freq": rng.choice(weight_values),
                "ref_suppress_multiplier": rng.choice(suppress_values),
                "freq_suppress_multiplier": rng.choice(suppress_values),
                "margin_threshold": rng.choice(margin_values),
                "pmax_threshold": rng.choice(pmax_values),
                "T_low": rng.choice(t_low_values),
                "T_high": rng.choice(t_high_values),
            }
        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict tuning for ours method")
    parser.add_argument("--dataset", default="Output/sim_spectrum", help="Dataset directory with features_brb.csv and labels.json")
    parser.add_argument("--out_config", default="Output/ours_best_config.json")
    parser.add_argument("--out_report", default="Output/tuning_report.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    dataset_dir = Path(args.dataset)
    features_path = dataset_dir / "features_brb.csv"
    labels_path = dataset_dir / "labels.json"

    rows, feature_cols = _load_features(features_path)
    labels = _load_labels(labels_path)
    X, y, _ = _align_dataset(rows, labels)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_idx,
        val_idx,
        test_idx,
    ) = stratified_split(X, y, train_size=TRAIN_SIZE, val_size=VAL_SIZE, random_state=SEED)

    normal_indices = y_train == CLASS_TO_IDX["normal"]
    normal_quantiles = {}
    if np.any(normal_indices):
        normal_data = X_train[normal_indices]
        feature_map = {name: idx for idx, name in enumerate(feature_cols)}
        p95 = {}
        median = {}
        for key in [
            "global_offset_db",
            "shape_rmse",
            "ripple_hp",
            "freq_shift_score",
            "compress_ratio",
            "compress_ratio_high",
            "band_offset_db_1",
            "env_overrun_rate",
            "env_overrun_max",
            "env_overrun_mean",
        ]:
            idx = feature_map.get(key)
            if idx is None:
                continue
            values = normal_data[:, idx]
            p95[key] = float(np.percentile(values, 95))
            median[key] = float(np.median(values))
        normal_quantiles = {"p95": p95, "median": median}

    coarse_configs = _build_coarse_grid(max_samples=3500)

    scored = []
    for config in coarse_configs:
        metrics = _evaluate(config, X_val, y_val, feature_cols, normal_quantiles)
        score = _objective(metrics)
        scored.append({"config": config, "score": score, "metrics": metrics})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top10 = scored[:10]

    rng = np.random.RandomState(SEED)
    for rank, entry in enumerate(top10):
        local_configs = _local_search(entry["config"], 200, rng)
        for config in local_configs:
            metrics = _evaluate(config, X_val, y_val, feature_cols, normal_quantiles)
            score = _objective(metrics)
            scored.append({"config": config, "score": score, "metrics": metrics})

    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]

    test_metrics = _evaluate(best["config"], X_test, y_test, feature_cols, normal_quantiles)
    train_metrics = _evaluate(best["config"], X_train, y_train, feature_cols, normal_quantiles)

    report = {
        "seed": SEED,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "class_order": CLASS_ORDER,
        "normal_quantiles": normal_quantiles,
        "best": {
            "config": best["config"],
            "val_metrics": {
                "accuracy": best["metrics"]["accuracy"],
                "macro_f1": best["metrics"]["macro_f1"],
            },
            "test_metrics": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            },
            "train_metrics": {
                "accuracy": train_metrics["accuracy"],
                "macro_f1": train_metrics["macro_f1"],
            },
        },
        "top10": [
            {
                "score": entry["score"],
                "config": entry["config"],
                "val_metrics": {
                    "accuracy": entry["metrics"]["accuracy"],
                    "macro_f1": entry["metrics"]["macro_f1"],
                },
            }
            for entry in scored[:10]
        ],
    }

    out_config = Path(args.out_config)
    out_report = Path(args.out_report)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    out_config.write_text(
        json.dumps({**best["config"], "normal_quantiles": normal_quantiles, "metrics": report["best"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Best config saved:", out_config)
    print("Tuning report saved:", out_report)
    print("Val accuracy:", best["metrics"]["accuracy"], "Val macro-F1:", best["metrics"]["macro_f1"])
    print("Test accuracy:", test_metrics["accuracy"], "Test macro-F1:", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
