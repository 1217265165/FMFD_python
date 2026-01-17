#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit feature schema for compare pipeline.

Outputs:
- Output/compare_methods/feature_columns.json
- Output/compare_methods/feature_leak_candidates.json
- Output/compare_methods/EXPECTED_FEATURES_<method>.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


LEAK_PREFIXES = ("sys_", "label", "target", "gt_", "y_", "truth", "class_")
LEAK_SUBSTRINGS = ("label", "target", "truth")

METHOD_NAMES = [
    "ours",
    "hcf",
    "aifd",
    "brb_p",
    "brb_mu",
    "dbrb",
    "a_ibrb",
    "fast_brb",
]


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


def read_feature_columns(features_path: Path) -> List[str]:
    with features_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return [name for name in reader.fieldnames if name not in ("sample_id", "id")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feature schema for compare pipeline")
    parser.add_argument(
        "--features",
        default="Output/sim_spectrum/features_brb.csv",
        help="Path to features_brb.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="Output/compare_methods",
        help="Directory to save schema outputs",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise SystemExit(f"[ERROR] Features file not found: {features_path}")

    feature_columns = read_feature_columns(features_path)
    leak_candidates = detect_leakage_columns(feature_columns)

    feature_columns_path = output_dir / "feature_columns.json"
    feature_columns_path.write_text(
        json.dumps(
            {
                "count": len(feature_columns),
                "features": feature_columns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leak_path = output_dir / "feature_leak_candidates.json"
    leak_path.write_text(
        json.dumps(
            {
                "count": len(leak_candidates),
                "features": leak_candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cleaned_columns = [c for c in feature_columns if c not in leak_candidates]
    for method in METHOD_NAMES:
        expected_path = output_dir / f"EXPECTED_FEATURES_{method}.json"
        expected_path.write_text(
            json.dumps(
                {
                    "method": method,
                    "feature_count": len(cleaned_columns),
                    "features": cleaned_columns,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"[INFO] Total columns: {len(feature_columns)}")
    print(f"[INFO] Leak candidates: {len(leak_candidates)}")
    if leak_candidates:
        print(f"[WARN] sys_/label-like columns detected: {leak_candidates}")
    else:
        print("[INFO] No sys_/label-like columns detected.")
    print(f"[INFO] Saved feature columns to: {feature_columns_path}")
    print(f"[INFO] Saved leak candidates to: {leak_path}")
    print(f"[INFO] Saved expected feature files to: {output_dir}")


if __name__ == "__main__":
    main()
