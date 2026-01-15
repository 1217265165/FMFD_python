import os
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from baseline.baseline import (
    load_and_align, compute_rrs_bounds, detect_switch_steps,
    compute_envelope_from_vendor_plus_quantile, vendor_tolerance_dbm,
)
from baseline.rrs_envelope import vendor_tolerance_db
from baseline.config import (
    BAND_RANGES, K_LIST, SWITCH_TOL,
    BASELINE_ARTIFACTS, BASELINE_META,
    NORMAL_FEATURE_STATS, SWITCH_CSV, SWITCH_JSON, PLOT_PATH,
    OUTPUT_DIR, SINGLE_BAND_MODE, COVERAGE_MEAN_MIN, COVERAGE_MIN_MIN,
)
from baseline.viz import plot_rrs_envelope_switch
from features.extract import extract_system_features

# Envelope algorithm v6 parameters
EXTRA_CLIP_MAX_DB = 0.25  # Maximum extra width above vendor tolerance


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
    print(f"Loading normal response data from: {folder_path}")
    frequency, traces, names = load_and_align(folder_path, use_spectrum_column=True)
    print(f"Loaded {len(names)} traces, frequency points: {len(frequency)}")
    print(f"Frequency range: {frequency[0]:.2e} Hz to {frequency[-1]:.2e} Hz")

    # 2) 使用新版包络算法 (v6): 基于厂商容差 + 分位数残差
    # 不再使用旧的 k*sigma 搜索逻辑
    print("\n[Baseline] Using new envelope algorithm (v6): vendor tolerance + quantile-based extra margin")
    
    rrs, upper, lower, envelope_info = compute_envelope_from_vendor_plus_quantile(
        frequency, traces,
        normalize_offset=True,
        max_offset_db=0.4,
        drop_outliers=True,
        exceed_rate_threshold=0.10,
        max_exceed_threshold=0.80,
        p95_exceed_threshold=0.30,
        quantile=0.97,
        smooth_sigma_hz=200e6,  # 200MHz 平滑
        extra_clip_max=EXTRA_CLIP_MAX_DB,
        target_coverage_mean=COVERAGE_MEAN_MIN,
        target_coverage_min=COVERAGE_MIN_MIN,
    )
    bounds = (upper, lower)
    
    # 构建 coverage_info 用于后续兼容
    coverage_info = {
        'coverage_mean': envelope_info['coverage_mean'],
        'coverage_min': envelope_info['coverage_min'],
        'k_final': None,  # 新算法不使用 k
        'rrs_mae': 0.0,   # pointwise median，MAE=0
        'width_min': envelope_info['width_min'],
        'width_median': envelope_info['width_median'],
        'width_max': envelope_info['width_max'],
        'half_width_max': envelope_info['half_width_max'],
        'half_width_p50': envelope_info['half_width_p50'],
        'offset_p95': envelope_info.get('offset_stats', {}).get('p95', 0),
        'rrs_smooth_enabled': False,
        'sliding_coverage_min': envelope_info['coverage_min'],  # 用 min 代替
    }
    
    print(f"RRS computed, coverage_mean: {coverage_info['coverage_mean']:.4f}, "
          f"coverage_min: {coverage_info['coverage_min']:.4f}")
    
    # Verify coverage meets requirements
    if coverage_info.get('coverage_mean', 0) < COVERAGE_MEAN_MIN:
        print(f"WARNING: coverage_mean {coverage_info['coverage_mean']:.4f} < {COVERAGE_MEAN_MIN}")
    if coverage_info.get('coverage_min', 0) < COVERAGE_MIN_MIN:
        print(f"WARNING: coverage_min {coverage_info['coverage_min']:.4f} < {COVERAGE_MIN_MIN}")

    # 3) 切换点步进 (empty in single-band mode)
    switch_feats = detect_switch_steps(frequency, traces, BAND_RANGES, tol=SWITCH_TOL)
    if SINGLE_BAND_MODE:
        print("Single-band mode: no switch points detected (disabled)")
    else:
        print(f"Detected {len(switch_feats)} switch points")

    # 4) 可视化
    plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, plot_path)
    
    # 4.1) 新增：包络宽度可视化（用于检查是否有局部鼓包）
    width = bounds[0] - bounds[1]  # upper - lower
    width_plot_path = _resolve(repo_root, "Output/baseline_width.png")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(frequency / 1e9, width, 'b-', linewidth=1, label='Envelope Width')
        plt.axhline(y=0.60, color='r', linestyle='--', alpha=0.7, label='Max threshold (0.60 dB)')
        plt.axhline(y=np.median(width), color='g', linestyle=':', alpha=0.7, 
                    label=f'Median ({np.median(width):.3f} dB)')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Envelope Width (dB)')
        plt.title('Envelope Width vs Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(width_plot_path, dpi=150)
        plt.close()
        print(f"Envelope width plot saved: {width_plot_path}")
    except Exception as e:
        print(f"Warning: Could not save width plot: {e}")

    # 5) 保存基线产物（包含 traces，供仿真脚本使用）
    # 计算基线整体电平中心
    center_level_db = float(np.median(rrs))
    
    # 厂商规格容差（系统级：-10 ± 0.4 dB）
    spec_center_db = -10.0
    spec_tol_db = 0.4
    
    np.savez(
        baseline_artifacts,
        frequency=frequency,
        traces=traces,
        rrs=rrs,
        upper=bounds[0],
        lower=bounds[1],
        center_level_db=center_level_db,
        spec_center_db=spec_center_db,
        spec_tol_db=spec_tol_db,
        vendor_tolerance_db=vendor_tolerance_db(frequency),
    )
    
    # Build comprehensive metadata (updated for v6 algorithm)
    width = bounds[0] - bounds[1]
    meta_dict = {
        "band_ranges": BAND_RANGES,
        "k_list": K_LIST,
        "single_band_mode": SINGLE_BAND_MODE,
        "envelope_version": envelope_info.get('version', 'v6'),
        "coverage_mean": coverage_info.get('coverage_mean'),
        "coverage_min": coverage_info.get('coverage_min'),
        "k_final": coverage_info.get('k_final'),  # None for v6
        "n_traces": len(names),
        "n_valid_traces": envelope_info.get('n_valid_traces'),
        "n_frequency_points": len(frequency),
        "freq_start_hz": float(frequency[0]),
        "freq_end_hz": float(frequency[-1]),
        "freq_step_hz": float(np.median(np.diff(frequency))),
        # 基线电平与规格容差
        "center_level_db": center_level_db,
        "spec_center_db": spec_center_db,
        "spec_tol_db": spec_tol_db,
        "spec_upper_db": spec_center_db + spec_tol_db,
        "spec_lower_db": spec_center_db - spec_tol_db,
        # width/half_width 相关元数据
        "width_min": float(np.min(width)),
        "width_median": float(np.median(width)),
        "width_max": float(np.max(width)),
        "width_smoothness": float(np.std(np.diff(width))),
        "half_width_max": envelope_info.get('half_width_max'),
        "half_width_p50": envelope_info.get('half_width_p50'),
        # offset 统计
        "offset_stats": envelope_info.get('offset_stats', {}),
        # 其他
        "rrs_mae": coverage_info.get('rrs_mae'),
        "dropped_trace_ids": envelope_info.get('dropped_trace_ids', []),
        "smooth_params": {
            "smooth_sigma_hz": envelope_info.get('smooth_sigma_hz', 200e6),
            "extra_clip_max": envelope_info.get('extra_clip_max', 0.25),
            "quantile": envelope_info.get('quantile', 0.97),
        },
    }
    
    with open(baseline_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)
    
    # 5.5) 保存 baseline_quality.json（质量指标，供前端和验收使用）
    # 新版 v6: 增加 half_width_max, half_width_p50, offset_p95 字段
    quality_json_path = _resolve(repo_root, "Output/baseline_quality.json")
    
    # 计算 half_width
    half_width = (bounds[0] - bounds[1]) / 2
    vendor_tol_max = np.max(vendor_tolerance_dbm(frequency))
    
    quality_dict = {
        "coverage_mean": coverage_info.get('coverage_mean'),
        "coverage_min": coverage_info.get('coverage_min'),
        "sliding_coverage_min": coverage_info.get('sliding_coverage_min'),
        # width 相关
        "width_min": float(np.min(width)),
        "width_median": float(np.median(width)),
        "width_max": float(np.max(width)),
        "width_smoothness": float(np.std(np.diff(width))),
        # half_width 相关 (新增)
        "half_width_max": float(np.max(half_width)),
        "half_width_p50": float(np.median(half_width)),
        # offset 相关 (新增)
        "offset_p95": coverage_info.get('offset_p95', 0.0),
        # 基本信息
        "center_level_db": center_level_db,
        "n_traces": len(names),
        "k_final": coverage_info.get('k_final'),
        "rrs_mae": coverage_info.get('rrs_mae'),
        "rrs_smooth_enabled": coverage_info.get('rrs_smooth_enabled', False),
        # 阈值定义
        "thresholds": {
            "coverage_mean_min": 0.97,
            "coverage_min_min": 0.93,
            "sliding_coverage_min": 0.93,
            "half_width_max": vendor_tol_max + EXTRA_CLIP_MAX_DB,  # vendor_tol_max + extra_clip_max
            "width_smoothness_max": 0.03,
        },
        # passed 规则: coverage_mean>=0.97 且 coverage_min>=0.93 且 half_width_max <= (vendor_tol_max+0.25)
        "passed": bool(
            coverage_info.get('coverage_mean', 0) >= 0.97 and
            coverage_info.get('coverage_min', 0) >= 0.93 and
            float(np.max(half_width)) <= (vendor_tol_max + 0.25)
        ),
    }
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_dict, f, ensure_ascii=False, indent=2)
    print(f"Baseline quality saved: {quality_json_path}")

    # 6) 保存切换点特性
    if switch_feats:
        pd.DataFrame(switch_feats).to_csv(switch_csv, index=False)
        with open(switch_json, "w", encoding="utf-8") as f:
            json.dump(switch_feats, f, indent=4, ensure_ascii=False)
    else:
        # Write empty files in single-band mode
        with open(switch_csv, 'w') as f:
            f.write('')
        with open(switch_json, "w", encoding="utf-8") as f:
            json.dump([], f)

    # 7) 正常特征统计（用于阈值初设）
    feats_list = []
    for i in range(traces.shape[0]):
        amp = traces[i]
        feats = extract_system_features(frequency, rrs, bounds, BAND_RANGES, amp)
        feats_list.append(feats)
    stats_df = pd.DataFrame(feats_list)
    stats_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(normal_feat_stats)

    print("\n" + "="*60)
    print("基线包络与RRS已保存:", baseline_artifacts, baseline_meta)
    print("切换点特性已保存:", switch_csv, switch_json)
    print("正常特征统计已保存:", normal_feat_stats)
    print("="*60)
    print(f"\nCoverage validation: mean={coverage_info.get('coverage_mean', 'N/A'):.4f}, "
          f"min={coverage_info.get('coverage_min', 'N/A'):.4f}")
    if SINGLE_BAND_MODE:
        print("Mode: SINGLE_BAND (10MHz-8.2GHz, preamp OFF)")


if __name__ == "__main__":
    main()