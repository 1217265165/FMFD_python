#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature integrity checker for features_brb.csv.

Checks:
- NaN / Inf ratio per feature
- zero variance features
- min/max ranges for quick sanity checks
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_report(features_path: Path, output_path: Path) -> Dict[str, object]:
    with features_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("features_brb.csv missing header")
        feature_names = [name for name in reader.fieldnames if name not in ("sample_id", "id")]
        values: Dict[str, List[float]] = {name: [] for name in feature_names}
        nan_counts = {name: 0 for name in feature_names}
        inf_counts = {name: 0 for name in feature_names}
        total_rows = 0

        for row in reader:
            total_rows += 1
            for name in feature_names:
                val = _safe_float(row.get(name, ""))
                if val is None or math.isnan(val):
                    nan_counts[name] += 1
                    continue
                if math.isinf(val):
                    inf_counts[name] += 1
                    continue
                values[name].append(val)

    anomalies = []
    zero_var = []
    feature_stats = {}
    for name in feature_names:
        vals = values[name]
        n_valid = len(vals)
        if n_valid == 0:
            var = 0.0
            min_val = 0.0
            max_val = 0.0
        else:
            mean = sum(vals) / n_valid
            var = sum((v - mean) ** 2 for v in vals) / max(n_valid - 1, 1)
            min_val = min(vals)
            max_val = max(vals)
        if var == 0.0:
            zero_var.append(name)

        nan_ratio = nan_counts[name] / max(total_rows, 1)
        inf_ratio = inf_counts[name] / max(total_rows, 1)
        feature_stats[name] = {
            "nan_ratio": nan_ratio,
            "inf_ratio": inf_ratio,
            "var": var,
            "min": min_val,
            "max": max_val,
            "n_valid": n_valid,
        }
        if nan_ratio > 0 or inf_ratio > 0 or var == 0.0:
            anomalies.append((name, nan_ratio, inf_ratio, var, min_val, max_val))

    anomalies_sorted = sorted(
        anomalies, key=lambda x: (-(x[1] + x[2]), x[3])
    )
    top_anomalies = [
        {
            "name": name,
            "nan_ratio": nan_ratio,
            "inf_ratio": inf_ratio,
            "var": var,
            "min": min_val,
            "max": max_val,
        }
        for name, nan_ratio, inf_ratio, var, min_val, max_val in anomalies_sorted[:20]
    ]

    report = {
        "features_path": str(features_path),
        "total_rows": total_rows,
        "total_features": len(feature_names),
        "zero_variance_features": zero_var,
        "top_anomalies": top_anomalies,
        "feature_stats": feature_stats,
        "required_checks": {
            "X16": feature_stats.get("X16"),
            "X17": feature_stats.get("X17"),
            "X18": feature_stats.get("X18"),
        },
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check feature integrity in features_brb.csv")
    parser.add_argument("--features", type=Path, required=True, help="Path to features_brb.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report path")
    args = parser.parse_args()

    report = generate_report(args.features, args.output)
    print(
        f"[INFO] Feature integrity report saved to {args.output} "
        f"(rows={report['total_rows']}, features={report['total_features']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
