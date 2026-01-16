#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from methods.brb_mu.interface import BRBMUMethod
from methods.ours_adapter import OursAdapter

CLASS_ORDER = ["ref_error", "amp_error", "normal", "freq_error"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
PRED_LABEL_MAP = {
    "正常": "normal",
    "幅度失准": "amp_error",
    "频率失准": "freq_error",
    "参考电平失准": "ref_error",
}


def _load_features(features_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(features_path, encoding="utf-8-sig")
    if "sample_id" not in df.columns:
        raise KeyError("features_brb.csv must contain sample_id column")
    feature_cols = [c for c in df.columns if c != "sample_id"]
    return df, feature_cols


def _load_labels(labels_path: Path) -> Dict[str, Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _align_dataset(df: pd.DataFrame, labels: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = []
    y = []
    sample_ids = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in labels:
            continue
        sys_class = labels[sample_id].get("system_fault_class", "normal") or "normal"
        sys_class = sys_class if sys_class in CLASS_TO_IDX else "normal"
        rows.append(row)
        y.append(CLASS_TO_IDX[sys_class])
        sample_ids.append(sample_id)
    if not rows:
        raise ValueError("No aligned samples between features and labels")
    aligned = pd.DataFrame(rows)
    features = aligned.drop(columns=["sample_id"]).to_numpy(dtype=float)
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


def _run_ours(X: np.ndarray, feature_names: List[str], config: Dict) -> np.ndarray:
    model = OursAdapter(calibration_override=config)
    result = model.predict(X, meta={"feature_names": feature_names})
    return result["system_pred"]


def _run_brb_mu(df: pd.DataFrame) -> np.ndarray:
    method = BRBMUMethod()
    preds = []
    for _, row in df.iterrows():
        features = row.drop(labels=["sample_id"]).to_dict()
        sys_result = method.infer_system(features)
        pred_label = sys_result.get("predicted_label", "正常")
        pred_class = PRED_LABEL_MAP.get(pred_label, "normal")
        preds.append(CLASS_TO_IDX[pred_class])
    return np.array(preds, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare brb_mu vs ours on a dataset")
    parser.add_argument("--dataset", required=True, help="Directory containing features_brb.csv and labels.json")
    parser.add_argument("--config", default="Output/ours_best_config.json", help="Ours config JSON")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    features_path = dataset_dir / "features_brb.csv"
    labels_path = dataset_dir / "labels.json"

    df, feature_cols = _load_features(features_path)
    labels = _load_labels(labels_path)
    X, y_true, _ = _align_dataset(df, labels)

    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    ours_pred = _run_ours(X, feature_cols, config)
    brb_pred = _run_brb_mu(df)

    ours_metrics = _metrics(y_true, ours_pred, len(CLASS_ORDER))
    brb_metrics = _metrics(y_true, brb_pred, len(CLASS_ORDER))

    print("=== Accuracy ===")
    print(f"brb_mu: {brb_metrics['accuracy']:.4f}")
    print(f"ours:   {ours_metrics['accuracy']:.4f}")
    print(f"delta:  {ours_metrics['accuracy'] - brb_metrics['accuracy']:.4f}")

    print("\n=== Per-class Metrics (order: ref, amp, normal, freq) ===")
    for idx, name in enumerate(CLASS_ORDER):
        o = ours_metrics["per_class"][idx]
        b = brb_metrics["per_class"][idx]
        print(
            f"{name}: ours P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} | "
            f"brb_mu P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}"
        )

    print("\n=== Confusion Matrices (rows=true, cols=pred) ===")
    print("ours:\n", ours_metrics["confusion"])
    print("brb_mu:\n", brb_metrics["confusion"])


if __name__ == "__main__":
    main()
