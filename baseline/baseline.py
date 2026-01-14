import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .config import BAND_RANGES, K_LIST, N_POINTS, SINGLE_BAND_MODE, COVERAGE_MEAN_MIN, COVERAGE_MIN_MIN


def load_and_align(folder_path, use_spectrum_column=True):
    """
    加载文件夹内所有 CSV 频响，插值到统一频率网格（取频率交集，N_POINTS 均匀采样）。
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing CSV files.
    use_spectrum_column : bool
        If True, uses the second-to-last column (频谱仪读数) as amplitude.
        If False, uses column 1 (功率 dBm).
        
    返回: frequency, traces(np.ndarray: n_traces x n_points), file_names
    
    Note on column selection:
    - Column 0: 频率（Hz） - Frequency in Hz
    - Column 1: 功率（dBm） - Power setting
    - Column -2 (second to last): 频谱仪读数 - Spectrum analyzer reading
    - Column -1 (last): 差值 - Difference
    """
    traces = []
    names = []
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            try:
                # Try different encodings
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
                    
                    # Use spectrum analyzer reading (second-to-last column) as per requirement
                    if use_spectrum_column and df.shape[1] >= 3:
                        amp = df.iloc[:, -2].values.astype(float)  # 频谱仪读数
                    else:
                        amp = df.iloc[:, 1].values.astype(float)  # 功率读数
                    
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
    """计算包络覆盖率。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points), amplitude traces.
    upper : np.ndarray
        Upper envelope boundary.
    lower : np.ndarray
        Lower envelope boundary.
        
    Returns
    -------
    dict
        coverage_mean: 平均覆盖率
        coverage_min: 最小覆盖率
        coverage_per_trace: 每条曲线的覆盖率
    """
    n_traces = traces.shape[0]
    coverages = []
    
    for i in range(n_traces):
        trace = traces[i]
        in_bounds = (trace >= lower) & (trace <= upper)
        coverage = np.mean(in_bounds)
        coverages.append(coverage)
    
    return {
        'coverage_mean': float(np.mean(coverages)),
        'coverage_min': float(np.min(coverages)),
        'coverage_per_trace': coverages,
    }


def auto_expand_envelope(
    frequency, traces, 
    initial_k=3.0, 
    target_coverage_mean=COVERAGE_MEAN_MIN,
    target_coverage_min=COVERAGE_MIN_MIN,
    max_iterations=20,
    smooth_envelope=True
):
    """自动扩大包络直到达到覆盖率要求。
    
    优化版（2024-01）：
    - 使用 Savitzky-Golay 滤波器平滑包络，减少高频噪声
    - 更稳健的包络计算
    
    Uses median ± k*MAD approach with adaptive k expansion.
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis.
    traces : np.ndarray
        Shape (n_traces, n_points).
    initial_k : float
        Initial k factor for MAD-based envelope.
    target_coverage_mean : float
        Minimum required mean coverage (default 0.97).
    target_coverage_min : float
        Minimum required min coverage (default 0.93).
    max_iterations : int
        Maximum iterations for envelope expansion.
    smooth_envelope : bool
        If True, apply Savitzky-Golay smoothing to envelope bounds.
        
    Returns
    -------
    tuple
        (rrs, (upper, lower), coverage_info)
    """
    # Compute baseline (median for robustness)
    rrs = np.median(traces, axis=0)
    
    # Apply Savitzky-Golay smoothing to RRS baseline
    if smooth_envelope and len(rrs) >= 101:
        rrs = savgol_filter(rrs, window_length=101, polyorder=3)
    elif smooth_envelope and len(rrs) >= 51:
        rrs = savgol_filter(rrs, window_length=51, polyorder=3)
    
    # MAD-based robust spread estimate
    mad = np.median(np.abs(traces - rrs), axis=0)
    # Convert MAD to sigma-equivalent: sigma ≈ 1.4826 * MAD
    sigma_est = 1.4826 * mad
    
    # Smooth sigma estimate to reduce envelope irregularities
    if smooth_envelope and len(sigma_est) >= 51:
        sigma_est = savgol_filter(sigma_est, window_length=51, polyorder=2)
    
    k = initial_k
    
    for iteration in range(max_iterations):
        upper = rrs + k * sigma_est
        lower = rrs - k * sigma_est
        
        # Apply smoothing to envelope bounds
        if smooth_envelope:
            if len(upper) >= 101:
                upper = savgol_filter(upper, window_length=101, polyorder=3)
                lower = savgol_filter(lower, window_length=101, polyorder=3)
            elif len(upper) >= 51:
                upper = savgol_filter(upper, window_length=51, polyorder=3)
                lower = savgol_filter(lower, window_length=51, polyorder=3)
        
        coverage = compute_coverage(traces, upper, lower)
        
        if (coverage['coverage_mean'] >= target_coverage_mean and 
            coverage['coverage_min'] >= target_coverage_min):
            print(f"Envelope converged at k={k:.2f}, "
                  f"coverage_mean={coverage['coverage_mean']:.4f}, "
                  f"coverage_min={coverage['coverage_min']:.4f}")
            break
        
        # Expand k
        k += 0.3
    else:
        print(f"Warning: Envelope did not converge after {max_iterations} iterations. "
              f"Final k={k:.2f}, coverage_mean={coverage['coverage_mean']:.4f}, "
              f"coverage_min={coverage['coverage_min']:.4f}")
    
    coverage['k_final'] = k
    coverage['n_iterations'] = iteration + 1
    coverage['smooth_envelope'] = smooth_envelope
    
    return rrs, (upper, lower), coverage


def compute_rrs_bounds(frequency, traces, band_ranges=BAND_RANGES, k_list=K_LIST, 
                       validate_coverage=True):
    """
    计算分段 RRS（均值/中位数）与包络（均值 ± k*std 或 自适应）。
    
    In single-band mode, uses auto_expand_envelope for coverage validation.
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis.
    traces : np.ndarray
        Shape (n_traces, n_points).
    band_ranges : list
        Band ranges for segmented processing.
    k_list : list
        k factors for each band.
    validate_coverage : bool
        If True and in single-band mode, automatically expand envelope
        to meet coverage requirements.
    """
    if SINGLE_BAND_MODE and validate_coverage:
        # Use adaptive envelope expansion in single-band mode
        return auto_expand_envelope(frequency, traces)
    
    # Original multi-band implementation
    assert len(band_ranges) == len(k_list)
    rrs = np.zeros_like(frequency)
    upper = np.zeros_like(frequency)
    lower = np.zeros_like(frequency)
    for (start, end), k in zip(band_ranges, k_list):
        mask = (frequency >= start) & (frequency <= end)
        if not np.any(mask):
            continue
        band = traces[:, mask]
        m = np.mean(band, axis=0)
        s = np.std(band, axis=0)
        rrs[mask] = m
        upper[mask] = m + k * s
        lower[mask] = m - k * s
    
    # Compute coverage for reporting
    coverage = compute_coverage(traces, upper, lower)
    
    return rrs, (upper, lower), coverage

def detect_switch_steps(frequency, traces, band_ranges=BAND_RANGES, tol=0.2):
    """
    检测频段切换点台阶特性，输出每个切换点的均值/标准差/是否在容差内。
    
    In single-band mode, returns empty list (no switch points).
    """
    if SINGLE_BAND_MODE:
        return []  # No switch points in single-band mode
    
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