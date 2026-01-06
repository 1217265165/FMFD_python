import os
import json
import glob
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from baseline.baseline import align_to_frequency
from baseline.config import (
    BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES,
    OUTPUT_DIR, DETECTION_RESULTS
)
from features.extract import extract_system_features
from BRB.system_brb import system_level_infer
from BRB.module_brb import module_level_infer


def resolve(repo_root: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_thresholds(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_thresholds(features, thresholds):
    """返回告警标志 dict：warn / alarm / ok，双阈值策略。"""
    flags = {}
    for k, v in thresholds.items():
        val = features.get(k, None)
        if val is None:
            continue
        low = v.get("warn", None)
        high = v.get("alarm", None)
        if high is not None and abs(val) >= high:
            flags[k] = "alarm"
        elif low is not None and abs(val) >= low:
            flags[k] = "warn"
        else:
            flags[k] = "ok"
    return flags


def main():
    # 仓库根：当前文件在 FMFD/pipelines 下，因此 parents[1] 是 FMFD
    repo_root = Path(__file__).resolve().parents[1]

    # 路径锚定到仓库根，同级 Output / to_detect / thresholds.json
    baseline_artifacts = resolve(repo_root, BASELINE_ARTIFACTS)
    baseline_meta = resolve(repo_root, BASELINE_META)
    out_dir = resolve(repo_root, OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    detection_results = resolve(repo_root, DETECTION_RESULTS)
    thresholds_path = resolve(repo_root, "thresholds.json")
    to_detect_glob = str(resolve(repo_root, "to_detect")) + "/*.csv"

    # 1) 加载基线
    art = np.load(baseline_artifacts)
    frequency = art["frequency"]
    rrs = art["rrs"]
    bounds = (art["upper"], art["lower"])
    with open(baseline_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    band_ranges = meta.get("band_ranges", BAND_RANGES)

    # 2) 加载阈值
    thresholds = load_thresholds(thresholds_path)

    # 3) 读取待检数据（仓库根/to_detect 下的 csv）
    files = glob.glob(to_detect_glob)
    if not files:
        raise FileNotFoundError(f"未找到待检 CSV：{to_detect_glob}")

    rows = []
    for fpath in files:
        df = pd.read_csv(fpath)
        if df.shape[1] < 2:
            continue
        freq_raw = df.iloc[:, 0].values
        amp_raw = df.iloc[:, 1].values
        amp = align_to_frequency(frequency, freq_raw, amp_raw)

        feats = extract_system_features(frequency, rrs, bounds, band_ranges, amp)
        sys_probs = system_level_infer(feats)
        # sys_probs = system_level_infer(feats, mode="simple")
        mod_probs = module_level_infer(feats, sys_probs)
        flags = apply_thresholds(feats, thresholds)

        row = {
            "file": str(fpath),
            **feats,
            **{f"sys_{k}": v for k, v in sys_probs.items()},
            **{f"mod_{k}": v for k, v in mod_probs.items()},
            **{f"flag_{k}": v for k, v in flags.items()},
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(detection_results, index=False, encoding="utf-8")
    print(f"检测结果已保存: {detection_results}")


if __name__ == "__main__":
    main()