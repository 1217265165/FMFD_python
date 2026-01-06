#!/usr/bin/env python3
"""Generate features_brb.csv from raw curves for comparison experiments."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.feature_pool import read_curve_csv, build_feature_pool_from_curve


def main():
    data_dir = ROOT / "Output/sim_spectrum"
    raw_curves_dir = data_dir / "raw_curves"
    labels_path = data_dir / "labels.json"
    output_path = data_dir / "features_brb.csv"
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return
    
    if not raw_curves_dir.exists():
        print(f"Raw curves directory not found: {raw_curves_dir}")
        return
    
    # Load labels to get sample IDs
    with open(labels_path, 'r', encoding='utf-8-sig') as f:
        labels_dict = json.load(f)
    
    print(f"Found {len(labels_dict)} samples in labels")
    
    # Extract features for each sample
    features_list = []
    
    for sample_id in sorted(labels_dict.keys()):
        curve_path = raw_curves_dir / f"{sample_id}.csv"
        
        if not curve_path.exists():
            print(f"Warning: curve file not found for {sample_id}")
            continue
        
        # Read curve
        freq, amp = read_curve_csv(curve_path)
        
        if not freq or not amp:
            print(f"Warning: empty curve for {sample_id}")
            continue
        
        # Extract features
        features = build_feature_pool_from_curve(freq, amp)
        features['sample_id'] = sample_id
        features_list.append(features)
    
    if not features_list:
        print("No features extracted!")
        return
    
    # Write to CSV
    fieldnames = ['sample_id'] + sorted([k for k in features_list[0].keys() if k != 'sample_id'])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features_list)
    
    print(f"Successfully generated {output_path}")
    print(f"Extracted {len(features_list)} samples with {len(fieldnames)-1} features each")


if __name__ == '__main__':
    main()
