import os
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from FMFD.baseline.baseline import load_and_align, compute_rrs_bounds, detect_switch_steps
from FMFD.baseline.config import (
    BAND_RANGES, K_LIST, SWITCH_TOL,
    BASELINE_ARTIFACTS, BASELINE_META,
    NORMAL_FEATURE_STATS, SWITCH_CSV, SWITCH_JSON, PLOT_PATH,
    OUTPUT_DIR,
)
from FMFD.baseline.viz import plot_rrs_envelope_switch
from FMFD.features.extract import extract_system_features


def _resolve(repo_root: Path, p: Union[str, Path]) -> Path:
    """将相对路径锚定到仓库根目录"""
    p = Path(p)
    return p if p.is_absolute() else repo_root / p


def main():
    # 仓库根：当前文件在 repo_root/pipelines 下，parents[1] 即 repo_root
    repo_root = Path(__file__).resolve().parents[1]

    out_dir = _resolve(repo_root, OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径全部锚定到仓库根
    baseline_artifacts = _resolve(repo_root, BASELINE_ARTIFACTS)
    baseline_meta = _resolve(repo_root, BASELINE_META)
    switch_csv = _resolve(repo_root, SWITCH_CSV)
    switch_json = _resolve(repo_root, SWITCH_JSON)
    normal_feat_stats = _resolve(repo_root, NORMAL_FEATURE_STATS)
    plot_path = _resolve(repo_root, PLOT_PATH)

    # 1) 加载并对齐正常数据（仓库根下 normal_response_data）
    folder_path = repo_root / "normal_response_data"
    frequency, traces, names = load_and_align(folder_path)

    # 2) 计算 RRS 与分段包络
    rrs, bounds = compute_rrs_bounds(frequency, traces, BAND_RANGES, K_LIST)

    # 3) 切换点步进
    switch_feats = detect_switch_steps(frequency, traces, BAND_RANGES, tol=SWITCH_TOL)

    # 4) 可视化
    plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, plot_path)

    # 5) 保存基线产物（包含 traces，供仿真脚本使用）
    np.savez(
        baseline_artifacts,
        frequency=frequency,
        traces=traces,
        rrs=rrs,
        upper=bounds[0],
        lower=bounds[1],
    )
    with open(baseline_meta, "w", encoding="utf-8") as f:
        json.dump({"band_ranges": BAND_RANGES, "k_list": K_LIST}, f, ensure_ascii=False, indent=2)

    # 6) 保存切换点特性
    pd.DataFrame(switch_feats).to_csv(switch_csv, index=False)
    with open(switch_json, "w", encoding="utf-8") as f:
        json.dump(switch_feats, f, indent=4, ensure_ascii=False)

    # 7) 正常特征统计（用于阈值初设）
    feats_list = []
    for i in range(traces.shape[0]):
        amp = traces[i]
        feats = extract_system_features(frequency, rrs, bounds, BAND_RANGES, amp)
        feats_list.append(feats)
    stats_df = pd.DataFrame(feats_list)
    stats_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(normal_feat_stats)

    print("基线包络与RRS已保存:", baseline_artifacts, baseline_meta)
    print("切换点特性已保存:", switch_csv, switch_json)
    print("正常特征统计已保存:", normal_feat_stats)


if __name__ == "__main__":
    main()