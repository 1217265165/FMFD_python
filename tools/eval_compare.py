#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _load_features(features_path: Path) -> Tuple[List[Dict[str, float]], List[str]]:
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
        support = confusion[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(support)})
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


def _run_brb_mu(rows: List[Dict[str, str]]) -> np.ndarray:
    method = BRBMUMethod()
    preds = []
    for row in rows:
        features = {k: float(v) for k, v in row.items() if k != "sample_id"}
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

    rows, feature_cols = _load_features(features_path)
    labels = _load_labels(labels_path)
    X, y_true, _ = _align_dataset(rows, labels)

    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    ours_pred = _run_ours(X, feature_cols, config)
    brb_pred = _run_brb_mu(rows)

    ours_metrics = _metrics(y_true, ours_pred, len(CLASS_ORDER))
    brb_metrics = _metrics(y_true, brb_pred, len(CLASS_ORDER))

    output_dir = dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write ours confusion matrix counts
    cm_path = output_dir / "ours_confusion_matrix_counts.csv"
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write("true/pred," + ",".join(CLASS_ORDER) + "\n")
        for i, label in enumerate(CLASS_ORDER):
            row = ",".join(str(v) for v in ours_metrics["confusion"][i])
            f.write(f"{label},{row}\n")

    # Write ours per-class metrics
    metrics_path = output_dir / "ours_per_class_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for idx, label in enumerate(CLASS_ORDER):
            m = ours_metrics["per_class"][idx]
            f.write(
                f"{label},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['support']}\n"
            )

    # Write system_features_v2.csv
    sys_feat_path = Path("Output") / "system_features_v2.csv"
    sys_feat_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {
        "amp_error": "amp_error",
        "freq_error": "freq_error",
        "ref_error": "ref_error",
        "normal": "normal",
    }
    required_cols = [
        "global_offset_db",
        "band_offset_db_1",
        "band_offset_db_2",
        "band_offset_db_3",
        "band_offset_db_4",
        "offset_slope",
        "ripple_hp",
        "shape_rmse",
        "tail_asym",
        "compress_ratio",
        "compress_ratio_high",
        "freq_shift_score",
    ]
    with open(sys_feat_path, "w", encoding="utf-8") as f:
        f.write("sample_id,label," + ",".join(required_cols) + "\n")
        for row in rows:
            sample_id = row.get("sample_id")
            label_entry = labels.get(sample_id, {})
            sys_label = label_map.get(label_entry.get("system_fault_class", "normal"), "normal")
            values = []
            for col in required_cols:
                val = float(row.get(col, 0.0))
                values.append(str(val))
            f.write(f"{sample_id},{sys_label}," + ",".join(values) + "\n")

    print("=== Accuracy / Macro-F1 ===")
    print(f"brb_mu: acc={brb_metrics['accuracy']:.4f} macroF1={brb_metrics['macro_f1']:.4f}")
    print(f"ours:   acc={ours_metrics['accuracy']:.4f} macroF1={ours_metrics['macro_f1']:.4f}")
    print(f"delta:  acc={ours_metrics['accuracy'] - brb_metrics['accuracy']:.4f} macroF1={ours_metrics['macro_f1'] - brb_metrics['macro_f1']:.4f}")

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
    print(f"\nSaved ours_confusion_matrix_counts.csv to: {cm_path}")
    print(f"Saved ours_per_class_metrics.csv to: {metrics_path}")
    print(f"Saved system_features_v2.csv to: {sys_feat_path}")


if __name__ == "__main__":
    main()
