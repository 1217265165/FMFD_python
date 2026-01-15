import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .config import (
    BAND_RANGES,
    K_LIST,
    N_POINTS,
    SINGLE_BAND_MODE,
    COVERAGE_MEAN_MIN,
    COVERAGE_MIN_MIN,
    VENDOR_TOL_SEGMENTS,
    TARGET_COVERAGE,
    TARGET_COVERAGE_MIN_SEG,
    RRSM_SG_WINDOW,
    RRSM_SG_POLY,
    RRSM_SG_MAE_MAX,
    OUTLIER_VIOL_RATE_TH,
    PRIOR_EXPAND_FOR_OUTLIER,
    QUANTILE_TARGET,
    SEG_SMOOTH_WINDOW_RATIO,
    SEG_SMOOTH_WINDOW_MIN,
    PRIOR_MAX_FACTOR,
    TRANSITION_POINTS,
    W_FLOOR_MIN,
    SMOOTHNESS_STD_MAX,
)

def load_and_align(folder_path, use_spectrum_column=True):
    """
    加载文件夹内所有 CSV 频响，插值到统一频率网格（取频率交集，N_POINTS 均匀采样）。
    返回: frequency, traces(np.ndarray: n_traces x n_points), file_names
    """
    traces = []
    names = []
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            try:
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
                    try:
                        df = pd.read_csv(os.path.join(folder_path, f), encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    df = pd.read_csv(os.path.join(folder_path, f), encoding='utf-8', errors='ignore')
                if df.shape[1] >= 2:
                    freq = df.iloc[:, 0].values.astype(float)
                    if use_spectrum_column and df.shape[1] >= 3:
                        amp = df.iloc[:, -2].values.astype(float)
                    else:
                        amp = df.iloc[:, 1].values.astype(float)
                    traces.append((freq, amp))
                    names.append(f)
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
                continue
    if not traces:
        raise FileNotFoundError("未找到有效 CSV 频响数据")
    all_freq = [t[0] for t in traces]
    min_f = max(np.min(f) for f in all_freq)
    max_f = min(np.max(f) for f in all_freq)
    frequency = np.linspace(min_f, max_f, N_POINTS)
    aligned = []
    for freq, amp in traces:
        interp = interp1d(freq, amp, kind="linear", fill_value="extrapolate")
        aligned.append(interp(frequency))
    return frequency, np.vstack(aligned), names

def align_to_frequency(target_frequency, freq, amp):
    """
    将单条曲线插值到指定 target_frequency 网格，用于检测阶段复用基线频率。
    """
    interp = interp1d(freq, amp, kind="linear", fill_value="extrapolate")
    return interp(target_frequency)

def compute_coverage(traces, upper, lower):
    """计算包络覆盖率（逐条曲线的点覆盖比例）。"""
    n_traces = traces.shape[0]
    coverages = []
    for i in range(n_traces):
        trace = traces[i]
        in_bounds = (trace >= lower) & (trace <= upper)
        coverages.append(float(np.mean(in_bounds)))
    return {
        'coverage_mean': float(np.mean(coverages)),
        'coverage_min': float(np.min(coverages)),
        'coverage_per_trace': coverages,
    }

def _ensure_odd(n, minimum=1):
    n = int(max(n, minimum))
    if n % 2 == 0:
        n += 1
    return max(n, 3)

def vendor_tolerance_db(frequency_hz):
    """逐点返回厂商先验容差（dB）。"""
    tol = np.zeros_like(frequency_hz, dtype=float)
    for (lo, hi), val in VENDOR_TOL_SEGMENTS:
        mask = (frequency_hz >= lo) & (frequency_hz <= hi)
        tol[mask] = val
    return tol

def compute_rrs_pointwise_median(traces, apply_smoothing=True):
    """逐点中位数 + 轻度 SG 平滑（可选，限制 MAE）。"""
    rrs_raw = np.median(traces, axis=0)
    if not apply_smoothing:
        return rrs_raw, rrs_raw
    window = _ensure_odd(RRSM_SG_WINDOW)
    if window >= len(rrs_raw):
        return rrs_raw, rrs_raw
    poly = min(RRSM_SG_POLY, window - 1)
    rrs_sm = savgol_filter(rrs_raw, window_length=window, polyorder=poly, mode='interp')
    mae = np.mean(np.abs(rrs_sm - rrs_raw))
    if mae > RRSM_SG_MAE_MAX:
        print(f"[RRS] SG 平滑被回退，MAE={mae:.4f} > {RRSM_SG_MAE_MAX:.4f}")
        return rrs_raw, rrs_raw
    return rrs_sm, rrs_raw

def filter_outlier_traces_by_prior(frequency, traces, rrs_raw, tol_prior, viol_rate_th=OUTLIER_VIOL_RATE_TH, expand_factor=PRIOR_EXPAND_FOR_OUTLIER):
    """基于先验容差的异常曲线剔除。"""
    residual = traces - rrs_raw
    thresh = tol_prior * expand_factor
    viol = np.abs(residual) > thresh
    viol_rate = np.mean(viol, axis=1)
    outlier_mask = viol_rate > viol_rate_th
    inlier_mask = ~outlier_mask
    return inlier_mask, outlier_mask, viol_rate

def _smooth_segment(arr, mask):
    idx = np.where(mask)[0]
    if len(idx) < 5:
        return arr
    window = _ensure_odd(int(len(idx) * SEG_SMOOTH_WINDOW_RATIO), minimum=SEG_SMOOTH_WINDOW_MIN)
    if window > len(idx):
        window = _ensure_odd(len(idx))
    poly = min(RRSM_SG_POLY, window - 1)
    arr_sm = savgol_filter(arr[idx], window_length=window, polyorder=poly, mode='interp')
    out = arr.copy()
    out[idx] = arr_sm
    return out

def _blend_boundaries(arr, boundaries):
    """在分段边界处做线性过渡避免折角。"""
    n = len(arr)
    arr = arr.copy()
    for idx_end, idx_start in boundaries:
        if idx_end < 0 or idx_start <= idx_end or idx_start >= n:
            continue
        # gap 线性插值
        gap_idx = np.arange(idx_end, idx_start + 1)
        if len(gap_idx) > 2:
            w = np.linspace(0, 1, len(gap_idx))
            arr[gap_idx] = arr[idx_end] * (1 - w) + arr[idx_start] * w
        # 端点两侧平滑衔接
        trans = min(TRANSITION_POINTS, idx_end + 1, n - idx_start)
        for i in range(trans):
            w = (i + 1) / (trans + 1)
            arr[idx_end - i] = arr[idx_end - i] * (1 - w) + arr[idx_start] * w
            if idx_start + i < n:
                arr[idx_start + i] = arr[idx_start + i] * (1 - w) + arr[idx_end] * w
    return arr

def compute_envelope_quantile_piecewise(frequency, traces, rrs, tol_prior, target=QUANTILE_TARGET):
    """用分位数 + 分段平滑 + 软先验约束计算包络。"""
    residual = traces - rrs
    q_low = (1 - target) / 2
    q_high = 1 - q_low
    lower0 = rrs + np.quantile(residual, q_low, axis=0)
    upper0 = rrs + np.quantile(residual, q_high, axis=0)

    # 分段平滑
    upper1 = upper0.copy()
    lower1 = lower0.copy()
    boundaries = []
    for (lo, hi), _tol in VENDOR_TOL_SEGMENTS:
        mask = (frequency >= lo) & (frequency <= hi)
        if not np.any(mask):
            continue
        upper1 = _smooth_segment(upper1, mask)
        lower1 = _smooth_segment(lower1, mask)
        idx = np.where(mask)[0]
        boundaries.append((idx[-1], idx[-1] + 1))
    upper1 = _blend_boundaries(upper1, boundaries[:-1])
    lower1 = _blend_boundaries(lower1, boundaries[:-1])

    # 软先验约束与下限地板
    w_upper = upper1 - rrs
    w_lower = rrs - lower1
    w_floor = np.median(np.abs(residual), axis=0)
    w_floor = _smooth_segment(w_floor, np.ones_like(w_floor, dtype=bool))
    w_floor = np.maximum(w_floor, W_FLOOR_MIN)

    w_upper_clipped = np.clip(w_upper, w_floor, tol_prior * PRIOR_MAX_FACTOR)
    w_lower_clipped = np.clip(w_lower, w_floor, tol_prior * PRIOR_MAX_FACTOR)

    upper_final = rrs + w_upper_clipped
    lower_final = rrs - w_lower_clipped

    coverage = compute_coverage(traces, upper_final, lower_final)
    if coverage['coverage_mean'] < TARGET_COVERAGE_MIN_SEG:
        # 回退上限裁剪，保留下限地板
        w_upper_final = np.maximum(w_upper, w_floor)
        w_lower_final = np.maximum(w_lower, w_floor)
        upper_final = rrs + w_upper_final
        lower_final = rrs - w_lower_final
    return upper_final, lower_final, {
        'q_low': q_low,
        'q_high': q_high,
        'w_floor_min': float(np.min(w_floor)),
    }

def compute_segment_coverage(frequency, traces, upper, lower, segments):
    covs = []
    for (lo, hi), _tol in segments:
        mask = (frequency >= lo) & (frequency <= hi)
        if not np.any(mask):
            covs.append(1.0)
            continue
        sub_traces = traces[:, mask]
        sub_upper = upper[mask]
        sub_lower = lower[mask]
        cov = compute_coverage(sub_traces, sub_upper, sub_lower)
        covs.append(float(cov['coverage_mean']))
    return covs, float(np.min(covs)) if covs else 1.0

def compute_rrs_bounds(frequency, traces, band_ranges=BAND_RANGES, k_list=K_LIST, validate_coverage=True, names=None):
    """
    计算 RRS 与分段包络（分位数+分段平滑+先验软约束）。
    返回 (rrs, (upper, lower), coverage_info)
    """
    tol_prior = vendor_tolerance_db(frequency)

    # 初次 RRS & outlier 剔除
    _, rrs_raw0 = compute_rrs_pointwise_median(traces, apply_smoothing=False)
    inlier_mask, outlier_mask, viol_rate = filter_outlier_traces_by_prior(
        frequency, traces, rrs_raw0, tol_prior,
        viol_rate_th=OUTLIER_VIOL_RATE_TH,
        expand_factor=PRIOR_EXPAND_FOR_OUTLIER,
    )
    if not np.any(inlier_mask):
        print("[WARN] 全部曲线被标为异常，回退使用全部数据")
        inlier_mask = np.ones_like(outlier_mask, dtype=bool)

    traces_inlier = traces[inlier_mask]

    rrs, rrs_raw = compute_rrs_pointwise_median(traces_inlier, apply_smoothing=True)

    upper, lower, env_meta = compute_envelope_quantile_piecewise(
        frequency, traces_inlier, rrs, tol_prior, target=QUANTILE_TARGET
    )

    coverage = compute_coverage(traces_inlier, upper, lower)
    seg_covs, seg_min = compute_segment_coverage(frequency, traces_inlier, upper, lower, VENDOR_TOL_SEGMENTS)
    smooth_upper = float(np.std(np.diff(upper)))
    smooth_lower = float(np.std(np.diff(lower)))

    width_upper = upper - rrs
    width_lower = rrs - lower
    width_stats = []
    for (lo, hi), _ in VENDOR_TOL_SEGMENTS:
        mask = (frequency >= lo) & (frequency <= hi)
        if np.any(mask):
            wu = width_upper[mask]
            wl = width_lower[mask]
            width_stats.append({
                'freq_range': (float(lo), float(hi)),
                'upper': {
                    'median': float(np.median(wu)),
                    'p95': float(np.percentile(wu, 95)),
                    'max': float(np.max(wu)),
                },
                'lower': {
                    'median': float(np.median(wl)),
                    'p95': float(np.percentile(wl, 95)),
                    'max': float(np.max(wl)),
                }
            })

    coverage_info = {
        'coverage_mean': coverage['coverage_mean'],
        'coverage_min': coverage['coverage_min'],
        'coverage_by_segment': seg_covs,
        'coverage_min_segment': seg_min,
        'smoothness_upper': smooth_upper,
        'smoothness_lower': smooth_lower,
        'width_stats_by_segment': width_stats,
        'tol_prior': tol_prior,
        'rrs_raw': rrs_raw,
        'inlier_mask': inlier_mask,
        'outlier_mask': outlier_mask,
        'viol_rate': viol_rate,
        'target_coverage': TARGET_COVERAGE,
        'target_coverage_min_segment': TARGET_COVERAGE_MIN_SEG,
        'q_low': env_meta['q_low'],
        'q_high': env_meta['q_high'],
        'w_floor_min': env_meta['w_floor_min'],
    }
    if names is not None:
        names = list(names)
        coverage_info['inlier_names'] = [names[i] for i, m in enumerate(inlier_mask) if m]
        coverage_info['outlier_names'] = [names[i] for i, m in enumerate(outlier_mask) if m]

    # 兼容 validate_coverage 参数（单频默认走新流程，不再逐步扩张 k）
    return rrs, (upper, lower), coverage_info

def detect_switch_steps(frequency, traces, band_ranges=BAND_RANGES, tol=0.2):
    """多频段切换点台阶检测；单频模式直接返回空。"""
    if SINGLE_BAND_MODE:
        return []
    feats = []
    for i in range(len(band_ranges) - 1):
        end_f = band_ranges[i][1]
        next_f = band_ranges[i + 1][0]
        m_end = np.argmin(np.abs(frequency - end_f))
        m_next = np.argmin(np.abs(frequency - next_f))
        current_vals = traces[:, m_end]
        next_vals = traces[:, m_next]
        diffs = next_vals - current_vals
        step_mean = float(np.mean(diffs))
        step_std = float(np.std(diffs))
        is_ok = np.abs(step_mean) <= tol
        feats.append({
            "end_freq": float(frequency[m_end]),
            "start_freq": float(frequency[m_next]),
            "step_mean": step_mean,
            "step_std": step_std,
            "tolerance": tol,
            "is_within_tolerance": bool(is_ok),
        })
    return feats
