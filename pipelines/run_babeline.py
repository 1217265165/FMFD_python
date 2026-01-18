import os
import csv
import json
import sys
from pathlib import Path
from typing import Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.baseline import (
    load_and_align,
    compute_rrs_bounds,
    compute_offsets,
    align_traces_by_offsets,
    summarize_residuals,
    detect_switch_steps,
    vendor_tolerance_dbm,
)
from baseline.config import (
    BAND_RANGES, K_LIST, SWITCH_TOL,
    BASELINE_ARTIFACTS, BASELINE_META,
    NORMAL_FEATURE_STATS, SWITCH_CSV, SWITCH_JSON, PLOT_PATH,
    BASELINE_OFFSETS, BASELINE_RESIDUAL_STATS,
    NORMAL_STATS_JSON, NORMAL_STATS_NPZ,
    OUTPUT_DIR, SINGLE_BAND_MODE, COVERAGE_MEAN_MIN, COVERAGE_MIN_MIN,
)
from baseline.viz import plot_rrs_envelope_switch
from features.extract import extract_system_features
from pipelines.default_paths import PROJECT_ROOT, OUTPUT_DIR as DEFAULT_OUTPUT_DIR, SINGLE_BAND, DISABLE_PREAMP, SEED, build_run_snapshot


def _resolve(repo_root: Path, p: Union[str, Path]) -> Path:
    """将相对路径锚定到仓库根目录"""
    p = Path(p)
    return p if p.is_absolute() else repo_root / p


def _smooth_series(values: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    # 仓库根：当前文件在 repo_root/pipelines 下，parents[1] 即 repo_root
    repo_root = PROJECT_ROOT

    out_dir = _resolve(repo_root, DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_run_snapshot(out_dir)

    print(f"[INFO] project_root={repo_root}")
    print(f"[INFO] single_band={SINGLE_BAND}")
    print(f"[INFO] disable_preamp={DISABLE_PREAMP}")
    print(f"[INFO] seed={SEED}")
    print(f"[INFO] output_dir={out_dir}")

    # 输出文件路径全部锚定到仓库根
    baseline_artifacts = _resolve(repo_root, BASELINE_ARTIFACTS)
    baseline_meta = _resolve(repo_root, BASELINE_META)
    switch_csv = _resolve(repo_root, SWITCH_CSV)
    switch_json = _resolve(repo_root, SWITCH_JSON)
    normal_feat_stats = _resolve(repo_root, NORMAL_FEATURE_STATS)
    plot_path = _resolve(repo_root, PLOT_PATH)
    offsets_csv = _resolve(repo_root, BASELINE_OFFSETS)
    residual_stats_path = _resolve(repo_root, BASELINE_RESIDUAL_STATS)
    normal_stats_json_path = _resolve(repo_root, NORMAL_STATS_JSON)
    normal_stats_npz_path = _resolve(repo_root, NORMAL_STATS_NPZ)

    # 1) 加载并对齐正常数据（仓库根下 normal_response_data）
    folder_path = repo_root / "normal_response_data"
    print(f"Loading normal response data from: {folder_path}")
    frequency, traces, names = load_and_align(folder_path, use_spectrum_column=True)
    print(f"Loaded {len(names)} traces, frequency points: {len(frequency)}")
    print(f"Frequency range: {frequency[0]:.2e} Hz to {frequency[-1]:.2e} Hz")

    # 2) 使用单一权威包络算法：RRS pointwise median + quantile envelope（先对齐 offset）
    # 说明：RRS 默认不平滑（最多允许极轻度平滑，但默认关闭）
    print("\n[Baseline] Using quantile envelope with offset alignment")
    rrs, bounds, coverage_info = compute_rrs_bounds(
        frequency,
        traces,
        validate_coverage=True,
    )
    upper, lower = bounds
    
    coverage_info.setdefault("k_final", None)
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

    # 4) 可视化（使用 offset 对齐后的曲线）
    offsets = compute_offsets(traces, rrs)
    aligned_traces = align_traces_by_offsets(traces, offsets)
    plot_rrs_envelope_switch(frequency, aligned_traces, rrs, bounds, switch_feats, plot_path)
    
    # 4.1) 新增：包络宽度可视化（用于检查是否有局部鼓包）
    width = bounds[0] - bounds[1]  # upper - lower
    width_plot_path = _resolve(repo_root, "Output/baseline_width.png")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(frequency / 1e9, width, 'b-', linewidth=1, label='Envelope Width')
        plt.axhline(y=0.40, color='r', linestyle='--', alpha=0.7, label='Max threshold (0.40 dB)')
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

    # 4.2) 保存 offset 对齐统计与残差分布
    residual_before = traces - rrs
    residual_after = aligned_traces - rrs
    offsets_rows = []
    for name, offset, res_before, res_after in zip(names, offsets, residual_before, residual_after):
        offsets_rows.append(
            {
                "curve_id": name,
                "offset_db": float(offset),
                "median_residual_before": float(np.median(res_before)),
                "median_residual_after": float(np.median(res_after)),
            }
        )
    if offsets_rows:
        with open(offsets_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(offsets_rows[0].keys()))
            writer.writeheader()
            writer.writerows(offsets_rows)
        print(f"Baseline offsets saved: {offsets_csv}")

    residual_stats = {
        "before_alignment": summarize_residuals(residual_before),
        "after_alignment": summarize_residuals(residual_after),
    }
    with open(residual_stats_path, "w", encoding="utf-8") as f:
        json.dump(residual_stats, f, ensure_ascii=False, indent=2)
    print(f"Baseline residual stats saved: {residual_stats_path}")

    # 5) 保存基线产物（包含 traces，供仿真脚本使用）
    # 计算基线整体电平中心
    center_level_db = float(np.median(rrs))
    
    # 厂商规格容差（系统级：-10 ± 0.4 dB）
    spec_center_db = -10.0
    spec_tol_db = 0.4

    # 系统级规格命中率（每条曲线整体中位数落在范围内）
    trace_medians = np.median(traces, axis=1)
    spec_lower = spec_center_db - spec_tol_db
    spec_upper = spec_center_db + spec_tol_db
    spec_hit = np.mean((trace_medians >= spec_lower) & (trace_medians <= spec_upper))
    print(f"[Baseline] Spec hit rate: {spec_hit:.2%} (median within [{spec_lower:.2f}, {spec_upper:.2f}] dBm)")
    
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
        vendor_tolerance_db=vendor_tolerance_dbm(frequency),
    )
    
    # Build comprehensive metadata
    width = bounds[0] - bounds[1]
    offsets = compute_offsets(traces, rrs)
    offset_p95 = float(np.percentile(np.abs(offsets), 95)) if offsets.size else 0.0
    meta_dict = {
        "band_ranges": BAND_RANGES,
        "k_list": K_LIST,
        "single_band_mode": SINGLE_BAND_MODE,
        "envelope_version": "quantile_v1",
        "coverage_mean": coverage_info.get('coverage_mean'),
        "coverage_min": coverage_info.get('coverage_min'),
        "k_final": coverage_info.get('k_final'),  # None for v6
        "n_traces": len(names),
        "n_valid_traces": len(names),
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
        "half_width_max": float(np.max(width) / 2),
        "half_width_p50": float(np.median(width) / 2),
        # offset 统计
        "offset_stats": {
            "p95_abs": offset_p95,
            "median_abs": float(np.median(np.abs(offsets))) if offsets.size else 0.0,
        },
        # 其他
        "rrs_mae": coverage_info.get('rrs_mae'),
        "dropped_trace_ids": [],
        "smooth_params": {
            "width_smooth_sigma_hz": coverage_info.get('smooth_params', {}).get('width_smooth_sigma_hz', 200e6),
            "quantiles": coverage_info.get('chosen_quantiles', {}),
        },
        "clip_db": coverage_info.get("clip_db", 0.4),
    }
    
    with open(baseline_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)
    
    # 5.5) 保存 baseline_quality.json（质量指标，供前端和验收使用）
    # 新版 v6: 增加 half_width_max, half_width_p50, offset_p95 字段
    quality_json_path = _resolve(repo_root, "Output/baseline_quality.json")
    
    # 计算 half_width
    half_width = (bounds[0] - bounds[1]) / 2
    
    quality_dict = {
        "coverage_mean": coverage_info.get('coverage_mean'),
        "coverage_min": coverage_info.get('coverage_min'),
        "sliding_coverage_min": coverage_info.get('sliding_coverage_min'),
        # width 相关
        "width_min": float(np.min(width)),
        "width_median": float(np.median(width)),
        "width_max": float(np.max(width)),
        "width_p95": float(np.percentile(width, 95)),
        "width_smoothness": float(np.std(np.diff(width))),
        # half_width 相关 (新增)
        "half_width_max": float(np.max(half_width)),
        "half_width_p50": float(np.median(half_width)),
        # offset 相关 (新增)
        "offset_p95": offset_p95,
        # 基本信息
        "center_level_db": center_level_db,
        "n_traces": len(names),
        "k_final": coverage_info.get('k_final'),
        "rrs_mae": coverage_info.get('rrs_mae'),
        "rrs_smooth_enabled": coverage_info.get('rrs_smooth_enabled', False),
        "chosen_quantiles": coverage_info.get('chosen_quantiles', {}),
        "clip_db": coverage_info.get("clip_db", 0.4),
        # 阈值定义
        "thresholds": {
            "coverage_mean_min": 0.97,
            "coverage_min_min": 0.93,
            "sliding_coverage_min": 0.93,
            "width_p95_max": 0.4,
            "width_smoothness_max": 0.03,
        },
        # passed 规则: coverage_mean>=0.97 且 coverage_min>=0.93 且 width_p95 <= 0.4
        "passed": bool(
            coverage_info.get('coverage_mean', 0) >= 0.97 and
            coverage_info.get('coverage_min', 0) >= 0.93 and
            float(np.percentile(width, 95)) <= 0.4
        ),
    }
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_dict, f, ensure_ascii=False, indent=2)
    print(f"Baseline quality saved: {quality_json_path}")

    # 6) 保存切换点特性
    if switch_feats:
        with open(switch_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(switch_feats[0].keys()))
            writer.writeheader()
            writer.writerows(switch_feats)
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
    for i in range(aligned_traces.shape[0]):
        amp = aligned_traces[i]
        feats = extract_system_features(frequency, rrs, bounds, BAND_RANGES, amp)
        feats_list.append(feats)
    keys = sorted({key for feat in feats_list for key in feat.keys()})
    values = {key: np.array([feat.get(key, 0.0) for feat in feats_list], dtype=float) for key in keys}
    stats_rows = {
        "count": [len(feats_list)] * len(keys),
        "mean": [float(np.mean(values[key])) for key in keys],
        "std": [float(np.std(values[key], ddof=1)) if len(values[key]) > 1 else 0.0 for key in keys],
        "min": [float(np.min(values[key])) for key in keys],
        "max": [float(np.max(values[key])) for key in keys],
        "p05": [float(np.percentile(values[key], 5)) for key in keys],
        "median": [float(np.median(values[key])) for key in keys],
        "p95": [float(np.percentile(values[key], 95)) for key in keys],
        "p99": [float(np.percentile(values[key], 99)) for key in keys],
        "mad": [float(np.median(np.abs(values[key] - np.median(values[key])))) for key in keys],
    }

    with open(normal_feat_stats, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stat"] + keys)
        for stat_name, row_values in stats_rows.items():
            writer.writerow([stat_name] + row_values)

    # 8) 正常统计摘要（用于仿真约束）
    # 仅平滑 sigma（噪声幅度），不平滑 rrs
    residuals = aligned_traces - rrs
    mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
    sigma_i = 1.4826 * mad
    sigma_smooth = _smooth_series(sigma_i, window=9)

    # 整体 offset 统计（来自真实正常曲线）
    offset_stats = _summarize(offsets)

    # 低频到高频的慢变 tilt：对残差做一阶/二阶拟合
    x = (frequency - frequency[0]) / (frequency[-1] - frequency[0] + 1e-12)
    linear_slopes = []
    quad_coeffs = []
    for res in residuals:
        coeffs = np.polyfit(x, res, deg=2)
        quad_coeffs.append(coeffs[0])
        linear_slopes.append(coeffs[1])
    linear_slopes = np.array(linear_slopes, dtype=float)
    quad_coeffs = np.array(quad_coeffs, dtype=float)

    # 关键特征分布摘要（用于仿真约束）
    key_features = [
        "gain", "bias", "comp", "df", "viol_rate",
        "step_score", "res_slope", "ripple_var",
        "X11", "X12", "X13", "X16", "X17", "X18",
    ]
    feature_summary = {}
    for key in key_features:
        values_key = np.array([feat.get(key, 0.0) for feat in feats_list], dtype=float)
        feature_summary[key] = _summarize(values_key)

    normal_stats = {
        "version": "v1",
        "freq_start_hz": float(frequency[0]),
        "freq_end_hz": float(frequency[-1]),
        "sigma_window": 9,
        "offset_stats": offset_stats,
        "tilt_stats": {
            "linear_slope": _summarize(linear_slopes),
            "quadratic_coef": _summarize(quad_coeffs),
        },
        "feature_stats": feature_summary,
        "arrays_path": str(normal_stats_npz_path.relative_to(repo_root)),
    }
    with open(normal_stats_json_path, "w", encoding="utf-8") as f:
        json.dump(normal_stats, f, ensure_ascii=False, indent=2)
    np.savez(
        normal_stats_npz_path,
        frequency=frequency,
        rrs=rrs,
        sigma_i=sigma_i,
        sigma_smooth=sigma_smooth,
    )
    print(f"Normal stats saved: {normal_stats_json_path}")

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
