import os
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from baseline.baseline import (
    load_and_align, 
    compute_rrs_bounds, 
    detect_switch_steps,
    compute_single_band_baseline,
    compute_abrupt_change_thresholds,
    compute_envelope_coverage,
    auto_widen_envelope,
)
from baseline.config import (
    BAND_RANGES, K_LIST, SWITCH_TOL,
    BASELINE_ARTIFACTS, BASELINE_META,
    NORMAL_FEATURE_STATS, SWITCH_CSV, SWITCH_JSON, PLOT_PATH,
    OUTPUT_DIR,
    SINGLE_BAND_MODE, SINGLE_BAND_RANGE, N_POINTS_REAL,
    ENVELOPE_Q_LOW, ENVELOPE_Q_HIGH, ENVELOPE_SMOOTH_WINDOW,
    ABRUPT_Q_DR, ABRUPT_Q_D2R,
)
from baseline.viz import plot_rrs_envelope_switch
from features.extract import extract_system_features


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
    
    # 选择点数：单频段模式使用较少点数
    n_points = N_POINTS_REAL if SINGLE_BAND_MODE else 10000
    
    print(f"加载正常数据: {folder_path}")
    print(f"模式: {'单频段' if SINGLE_BAND_MODE else '多频段'}")
    
    frequency, traces, names = load_and_align(folder_path, n_points=n_points)
    
    print(f"加载 {len(names)} 条正常曲线")
    print(f"频率范围: {frequency[0]:.2e} - {frequency[-1]:.2e} Hz")
    print(f"频率点数: {len(frequency)}")
    
    if SINGLE_BAND_MODE:
        # 单频段模式：使用新的baseline构建方法
        print("\n[单频段模式] 构建baseline...")
        
        # 构建单频段baseline
        baseline_stats = compute_single_band_baseline(
            frequency, traces,
            q_low=ENVELOPE_Q_LOW, 
            q_high=ENVELOPE_Q_HIGH,
            smooth_window=ENVELOPE_SMOOTH_WINDOW
        )
        
        center = baseline_stats['center']
        upper = baseline_stats['upper']
        lower = baseline_stats['lower']
        mad = baseline_stats['mad']
        
        # 包络覆盖率验证与自动增宽
        print("\n[包络覆盖率检查]")
        lower, upper, coverage_stats, k_widen = auto_widen_envelope(
            traces, lower, upper, mad,
            target_coverage=0.95,
            min_coverage=0.90,
            max_iterations=5
        )
        
        print(f"  最终覆盖率: mean={coverage_stats['coverage_mean']:.4f}, min={coverage_stats['coverage_min']:.4f}")
        print(f"  包络增宽系数: k={k_widen:.2f}")
        
        # 使用center作为rrs（实际是median）
        rrs = center
        bounds = (upper, lower)
        
        # 计算突变检测阈值
        abrupt_thresholds = compute_abrupt_change_thresholds(
            frequency, traces, center,
            q_dr=ABRUPT_Q_DR, q_d2r=ABRUPT_Q_D2R
        )
        
        print(f"  一阶差分阈值 (全局): {abrupt_thresholds['thr_dr_global']:.4f}")
        print(f"  二阶差分阈值 (全局): {abrupt_thresholds['thr_d2r_global']:.4f}")
        
        # 单频段无切换点
        switch_feats = []
        band_ranges_used = [SINGLE_BAND_RANGE]
        k_list_used = [3]  # 单频段默认k值
        
    else:
        # 多频段模式：使用原有方法
        print("\n[多频段模式] 构建baseline...")
        
        rrs, bounds = compute_rrs_bounds(frequency, traces, BAND_RANGES, K_LIST)
        center = rrs  # 多频段模式下rrs就是mean
        upper, lower = bounds
        mad = None
        abrupt_thresholds = None
        coverage_stats = None
        k_widen = 0.0
        
        switch_feats = detect_switch_steps(frequency, traces, BAND_RANGES, tol=SWITCH_TOL)
        band_ranges_used = BAND_RANGES
        k_list_used = K_LIST
    
    # 2) 可视化
    try:
        plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, plot_path)
        print(f"\n可视化已保存: {plot_path}")
    except Exception as e:
        print(f"\n可视化失败 (非致命): {e}")

    # 3) 保存基线产物（包含 traces，供仿真脚本使用）
    save_dict = {
        'frequency': frequency,
        'traces': traces,
        'rrs': rrs,
        'center': center,
        'upper': bounds[0],
        'lower': bounds[1],
    }
    
    # 单频段模式额外保存
    if SINGLE_BAND_MODE:
        save_dict['mad'] = mad
        save_dict['thr_dr_global'] = abrupt_thresholds['thr_dr_global']
        save_dict['thr_d2r_global'] = abrupt_thresholds['thr_d2r_global']
        save_dict['thr_dr_per_freq'] = abrupt_thresholds['thr_dr_per_freq']
        save_dict['thr_d2r_per_freq'] = abrupt_thresholds['thr_d2r_per_freq']
    
    np.savez(baseline_artifacts, **save_dict)
    
    meta_dict = {
        "single_band_mode": SINGLE_BAND_MODE,
        "band_ranges": band_ranges_used,
        "k_list": k_list_used,
        "n_traces": len(names),
        "n_points": len(frequency),
        "freq_min": float(frequency[0]),
        "freq_max": float(frequency[-1]),
    }
    
    if SINGLE_BAND_MODE:
        meta_dict.update({
            "envelope_q_low": ENVELOPE_Q_LOW,
            "envelope_q_high": ENVELOPE_Q_HIGH,
            "abrupt_q_dr": ABRUPT_Q_DR,
            "abrupt_q_d2r": ABRUPT_Q_D2R,
        })
        
        # 包络覆盖率统计
        if coverage_stats is not None:
            meta_dict.update({
                "envelope_coverage_mean": coverage_stats['coverage_mean'],
                "envelope_coverage_min": coverage_stats['coverage_min'],
                "envelope_coverage_p05": coverage_stats['coverage_p05'],
                "envelope_widen_k": k_widen,
            })
    
    with open(baseline_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    # 保存覆盖率质量报告
    if coverage_stats is not None:
        quality_path = out_dir / "baseline_quality.json"
        quality_report = {
            "coverage_mean": coverage_stats['coverage_mean'],
            "coverage_min": coverage_stats['coverage_min'],
            "coverage_p05": coverage_stats['coverage_p05'],
            "envelope_widen_k": k_widen,
            "target_coverage": 0.95,
            "min_coverage_threshold": 0.90,
            "quality_check_passed": coverage_stats['coverage_mean'] >= 0.95 and coverage_stats['coverage_min'] >= 0.90,
        }
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        print(f"  - 包络质量报告: {quality_path}")

    # 4) 保存切换点特性（单频段为空）
    pd.DataFrame(switch_feats).to_csv(switch_csv, index=False)
    with open(switch_json, "w", encoding="utf-8") as f:
        json.dump(switch_feats, f, indent=4, ensure_ascii=False)

    # 5) 正常特征统计（用于阈值初设）
    feats_list = []
    for i in range(traces.shape[0]):
        amp = traces[i]
        feats = extract_system_features(frequency, rrs, bounds, band_ranges_used, amp)
        feats_list.append(feats)
    stats_df = pd.DataFrame(feats_list)
    stats_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(normal_feat_stats)

    print(f"\n基线产物已保存:")
    print(f"  - 基线数据: {baseline_artifacts}")
    print(f"  - 元数据: {baseline_meta}")
    print(f"  - 切换点特性: {switch_csv}, {switch_json}")
    print(f"  - 正常特征统计: {normal_feat_stats}")


if __name__ == "__main__":
    main()