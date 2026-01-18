#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CLASS_ORDER = ["ref_error", "amp_error", "normal", "freq_error"]


def _load_confusion(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Empty confusion matrix file")
    header = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([int(x) for x in row[1:]])
    return header, matrix


def _load_per_class(path: Path):
    metrics = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["class"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
    return metrics


def _load_labels(labels_path: Path):
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = set()
    for entry in labels.values():
        cls = entry.get("system_fault_class", "normal") or "normal"
        classes.add(cls)
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardrails check for ours evaluation")
    parser.add_argument("--dataset", default="Output/sim_spectrum", help="Dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    cm_path = dataset_dir / "ours_confusion_matrix_counts.csv"
    per_class_path = dataset_dir / "ours_per_class_metrics.csv"
    labels_path = dataset_dir / "labels.json"

    if not cm_path.exists() or not per_class_path.exists():
        raise FileNotFoundError("Missing evaluation outputs for guardrails check")

    header, matrix = _load_confusion(cm_path)
    metrics = _load_per_class(per_class_path)

    if header != CLASS_ORDER:
        print("Label order mismatch in confusion matrix:", header)
        sys.exit(1)

    label_classes = _load_labels(labels_path)
    if not label_classes.issubset(set(CLASS_ORDER)):
        print("Label classes inconsistent with expected order:", label_classes)
        sys.exit(1)

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    accuracy = correct / total if total > 0 else 0.0

    amp_recall = metrics.get("amp_error", {}).get("recall", 0.0)
    freq_recall = metrics.get("freq_error", {}).get("recall", 0.0)
    normal_recall = metrics.get("normal", {}).get("recall", 0.0)

    failures = []
    if accuracy < 0.82:
        failures.append(f"accuracy {accuracy:.4f} < 0.82")
    if amp_recall < 0.55:
        failures.append(f"amp recall {amp_recall:.4f} < 0.55")
    if freq_recall < 0.85:
        failures.append(f"freq recall {freq_recall:.4f} < 0.85")
    if normal_recall > 0.97:
        failures.append(f"normal recall {normal_recall:.4f} > 0.97")

    if failures:
        print("Guardrails failed:")
        for failure in failures:
            print("-", failure)
        sys.exit(1)

    print("Guardrails passed")
    print(f"accuracy={accuracy:.4f}, amp_recall={amp_recall:.4f}, freq_recall={freq_recall:.4f}, normal_recall={normal_recall:.4f}")


if __name__ == "__main__":
    main()
