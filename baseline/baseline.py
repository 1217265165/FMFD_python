import os
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from .config import BAND_RANGES, K_LIST, N_POINTS, F_START, F_STOP, F_STEP, N_POINTS_FIXED
from .io import auto_read_csv, load_all_real_responses


def get_fixed_frequency_grid():
    """
    获取固定频率网格（820点，10MHz步进）。
    
    频率范围：10 MHz ~ 8.2 GHz，步进 10 MHz
    
    Returns:
        np.ndarray: 固定频率数组，长度820
    """
    # 使用linspace避免浮点精度问题
    freq = np.linspace(F_START, F_STOP, N_POINTS_FIXED)
    return freq


def align_to_fixed_grid(freq: np.ndarray, amp: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    """
    将曲线对齐到固定820点频率网格。
    
    如果原始频率与固定网格存在轻微浮点误差（<= tolerance Hz），使用直接对齐。
    否则使用线性插值重采样。
    
    Args:
        freq: 原始频率数组
        amp: 原始幅度数组
        tolerance: 频率容差(Hz)，默认1Hz
        
    Returns:
        np.ndarray: 对齐后的幅度数组（820点）
    """
    fixed_freq = get_fixed_frequency_grid()
    
    # 检查是否需要插值
    if len(freq) == len(fixed_freq):
        # 检查频率是否近似相等
        max_diff = np.max(np.abs(freq - fixed_freq))
        if max_diff <= tolerance:
            # 直接返回，频率已经对齐
            return amp
    
    # 需要插值重采样
    # 先排序确保频率单调递增
    sort_idx = np.argsort(freq)
    freq_sorted = freq[sort_idx]
    amp_sorted = amp[sort_idx]
    
    # 线性插值到固定网格
    interp = interp1d(freq_sorted, amp_sorted, kind="linear", 
                      bounds_error=False, fill_value="extrapolate")
    aligned_amp = interp(fixed_freq)
    
    return aligned_amp


def load_and_align(folder_path, n_points=None, use_new_format=True, use_fixed_grid=True):
    """
    加载文件夹内所有 CSV 频响，插值到统一频率网格。
    
    支持两种数据格式：
    1. 新格式（真实频响）：多列，第1列频率，倒数第二列幅度
    2. 旧格式：两列（freq_Hz, amplitude_dB）
    
    Args:
        folder_path: 数据文件夹路径
        n_points: 插值点数（当use_fixed_grid=False时使用），默认使用 N_POINTS
        use_new_format: 是否优先使用新格式读取
        use_fixed_grid: 是否使用固定820点频率网格（推荐True）
    
    返回: frequency, traces(np.ndarray: n_traces x n_points), file_names
    """
    if n_points is None:
        n_points = N_POINTS
    
    folder_path = str(folder_path)
    
    if use_new_format:
        # 使用新的自动检测读取函数
        try:
            freqs_list, amps_list, names = load_all_real_responses(folder_path)
            traces = list(zip(freqs_list, amps_list))
        except Exception as e:
            warnings.warn(f"新格式读取失败，回退到旧格式: {e}")
            use_new_format = False
    
    if not use_new_format:
        # 旧格式读取（两列CSV）
        traces = []
        names = []
        for f in os.listdir(folder_path):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(folder_path, f))
                    if df.shape[1] >= 2:
                        freq = df.iloc[:, 0].values.astype(float)
                        amp = df.iloc[:, 1].values.astype(float)
                        traces.append((freq, amp))
                        names.append(f)
                except Exception as e:
                    warnings.warn(f"跳过文件 {f}: {e}")
                    continue
    
    if not traces:
        raise FileNotFoundError("未找到有效 CSV 频响数据")
    
    # 验证原始数据频率范围
    for i, (freq, amp) in enumerate(traces):
        freq_min, freq_max = freq.min(), freq.max()
        if freq_min < 1e6 or freq_max < 1e8:
            file_name = names[i] if names and i < len(names) else f"index_{i}"
            raise ValueError(
                f"文件 {file_name}: 频率范围异常 "
                f"({freq_min:.2e} ~ {freq_max:.2e} Hz)，请检查频率列读取是否正确"
            )
    
    if use_fixed_grid:
        # 使用固定820点频率网格
        frequency = get_fixed_frequency_grid()
        aligned = []
        for freq, amp in traces:
            aligned_amp = align_to_fixed_grid(freq, amp)
            aligned.append(aligned_amp)
        print(f"[固定网格模式] 所有曲线对齐到 {len(frequency)} 点 ({frequency[0]/1e6:.0f}MHz ~ {frequency[-1]/1e9:.1f}GHz)")
    else:
        # 原始模式：取频率交集，插值到n_points
        all_freq = [t[0] for t in traces]
        min_f = max(np.min(f) for f in all_freq)
        max_f = min(np.max(f) for f in all_freq)
        frequency = np.linspace(min_f, max_f, n_points)
        
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

def compute_rrs_bounds(frequency, traces, band_ranges=BAND_RANGES, k_list=K_LIST):
    """
    计算分段 RRS（均值）与包络（均值 ± k*std）。
    """
    assert len(band_ranges) == len(k_list)
    rrs = np.zeros_like(frequency)
    upper = np.zeros_like(frequency)
    lower = np.zeros_like(frequency)
    for (start, end), k in zip(band_ranges, k_list):
        mask = (frequency >= start) & (frequency <= end)
        band = traces[:, mask]
        m = np.mean(band, axis=0)
        s = np.std(band, axis=0)
        rrs[mask] = m
        upper[mask] = m + k * s
        lower[mask] = m - k * s
    return rrs, (upper, lower)

def detect_switch_steps(frequency, traces, band_ranges=BAND_RANGES, tol=0.2):
    """
    检测频段切换点台阶特性，输出每个切换点的均值/标准差/是否在容差内。
    
    注意：对于单频段数据，此函数返回空列表。
    """
    # 单频段情况：无切换点
    if len(band_ranges) <= 1:
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


def compute_single_band_baseline(frequency, traces, q_low=0.02, q_high=0.98, smooth_window=21):
    """
    单频段 baseline 构建（不需要切换点）。
    
    由多条正常曲线构建：
    - center(f)：中心曲线（median）
    - lower(f), upper(f)：动态包络（分位数 + 平滑）
    - mad(f)：尺度统计量（用于归一化/阈值）
    
    Args:
        frequency: 频率数组
        traces: 多条正常曲线 (n_traces x n_points)
        q_low: 下包络分位数（默认0.02）
        q_high: 上包络分位数（默认0.98）
        smooth_window: 平滑窗口大小（必须为奇数）
        
    Returns:
        dict: {
            'center': 中心曲线,
            'lower': 下包络,
            'upper': 上包络,
            'mad': 尺度统计量,
            'std': 标准差
        }
    """
    # Ensure smooth_window is odd (required by median filter for symmetric window)
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # Center curve using median (more robust than mean against outliers)
    center = np.median(traces, axis=0)
    
    # 动态包络（分位数）
    lower_raw = np.percentile(traces, q_low * 100, axis=0)
    upper_raw = np.percentile(traces, q_high * 100, axis=0)
    
    # 平滑包络（rolling median）
    lower = median_filter(lower_raw, size=smooth_window, mode='reflect')
    upper = median_filter(upper_raw, size=smooth_window, mode='reflect')
    
    # 尺度统计量
    # MAD (Median Absolute Deviation)
    residuals = traces - center[np.newaxis, :]
    mad = np.median(np.abs(residuals), axis=0) * 1.4826  # 转换为标准差估计
    
    # 标准差
    std = np.std(traces, axis=0)
    
    return {
        'center': center,
        'lower': lower,
        'upper': upper,
        'mad': mad,
        'std': std,
    }


def compute_abrupt_change_thresholds(frequency, traces, center, q_dr=0.995, q_d2r=0.995):
    """
    计算"包络内但突变异常"的检测阈值。
    
    对每条正常曲线 y_k：
    - r_k[i] = y_k[i] - center[i]  (残差)
    - dr_k[i] = r_k[i] - r_k[i-1]  (一阶差分)
    - d2r_k[i] = dr_k[i] - dr_k[i-1]  (二阶差分)
    
    计算鲁棒统计（跨所有正常曲线）：
    - thr_dr_global = quantile(|dr|, q_dr)
    - thr_d2r_global = quantile(|d2r|, q_d2r)
    
    Args:
        frequency: 频率数组
        traces: 多条正常曲线 (n_traces x n_points)
        center: 中心曲线
        q_dr: 一阶差分阈值分位数
        q_d2r: 二阶差分阈值分位数
        
    Returns:
        dict: {
            'thr_dr_global': 全局一阶差分阈值,
            'thr_d2r_global': 全局二阶差分阈值,
            'thr_dr_per_freq': 每频点一阶差分阈值 (optional),
            'thr_d2r_per_freq': 每频点二阶差分阈值 (optional),
        }
    """
    n_traces, n_points = traces.shape
    
    # 计算残差
    residuals = traces - center[np.newaxis, :]  # (n_traces, n_points)
    
    # 一阶差分
    dr = np.diff(residuals, axis=1)  # (n_traces, n_points-1)
    
    # 二阶差分
    d2r = np.diff(dr, axis=1)  # (n_traces, n_points-2)
    
    # 全局阈值（所有曲线所有频点）
    thr_dr_global = float(np.percentile(np.abs(dr).flatten(), q_dr * 100))
    thr_d2r_global = float(np.percentile(np.abs(d2r).flatten(), q_d2r * 100))
    
    # 每频点阈值（可选，更精细的检测）
    thr_dr_per_freq = np.percentile(np.abs(dr), q_dr * 100, axis=0)
    thr_d2r_per_freq = np.percentile(np.abs(d2r), q_d2r * 100, axis=0)
    
    return {
        'thr_dr_global': thr_dr_global,
        'thr_d2r_global': thr_d2r_global,
        'thr_dr_per_freq': thr_dr_per_freq,
        'thr_d2r_per_freq': thr_d2r_per_freq,
    }


def detect_abrupt_changes(curve, center, thresholds, use_global=True):
    """
    检测曲线中的突变点（包络内但突变异常）。
    
    Args:
        curve: 待检测曲线
        center: 中心曲线
        thresholds: 由 compute_abrupt_change_thresholds 返回的阈值字典
        use_global: 是否使用全局阈值（True）或每频点阈值（False）
        
    Returns:
        dict: {
            'dr_violations': 一阶差分违例位置索引,
            'd2r_violations': 二阶差分违例位置索引,
            'max_dr': 最大一阶差分绝对值,
            'max_d2r': 最大二阶差分绝对值,
            'n_dr_violations': 一阶差分违例数量,
            'n_d2r_violations': 二阶差分违例数量,
        }
    """
    residual = curve - center
    dr = np.diff(residual)
    d2r = np.diff(dr)
    
    if use_global:
        thr_dr = thresholds['thr_dr_global']
        thr_d2r = thresholds['thr_d2r_global']
        dr_violations = np.where(np.abs(dr) > thr_dr)[0]
        d2r_violations = np.where(np.abs(d2r) > thr_d2r)[0]
    else:
        thr_dr = thresholds['thr_dr_per_freq']
        thr_d2r = thresholds['thr_d2r_per_freq']
        # 确保长度匹配
        dr_violations = np.where(np.abs(dr) > thr_dr[:len(dr)])[0]
        d2r_violations = np.where(np.abs(d2r) > thr_d2r[:len(d2r)])[0]
    
    return {
        'dr_violations': dr_violations.tolist(),
        'd2r_violations': d2r_violations.tolist(),
        'max_dr': float(np.max(np.abs(dr))) if len(dr) > 0 else 0.0,
        'max_d2r': float(np.max(np.abs(d2r))) if len(d2r) > 0 else 0.0,
        'n_dr_violations': len(dr_violations),
        'n_d2r_violations': len(d2r_violations),
    }


def compute_rrs(traces, center=None):
    """
    计算归一化残差 RRS (Relative Response Spectrum)。
    
    rrs(f) = residual / (MAD + eps)
    
    Args:
        traces: 多条曲线 (n_traces, n_points)
        center: 中心曲线，如果为None则使用median
        
    Returns:
        rrs: 归一化残差 (n_traces, n_points)
    """
    if center is None:
        center = np.median(traces, axis=0)
    
    residuals = traces - center[np.newaxis, :]
    mad = np.median(np.abs(residuals), axis=0) * 1.4826
    
    eps = 1e-6
    rrs = residuals / (mad[np.newaxis, :] + eps)
    
    return rrs


def compute_envelope_coverage(traces: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """
    计算包络对正常数据的覆盖率。
    
    对每条正常曲线，计算有多少点落在 [lower, upper] 包络内。
    
    Args:
        traces: 多条正常曲线 (n_traces, n_points)
        lower: 下包络
        upper: 上包络
        
    Returns:
        dict: {
            'coverage_per_trace': 每条曲线的覆盖率,
            'coverage_mean': 平均覆盖率,
            'coverage_min': 最小覆盖率,
            'coverage_p05': 5%分位数覆盖率,
            'n_traces': 曲线数量,
        }
    """
    n_traces, n_points = traces.shape
    
    # 对每条曲线计算覆盖率
    coverages = []
    for i in range(n_traces):
        trace = traces[i]
        in_envelope = (trace >= lower) & (trace <= upper)
        coverage = np.mean(in_envelope)
        coverages.append(coverage)
    
    coverages = np.array(coverages)
    
    return {
        'coverage_per_trace': coverages.tolist(),
        'coverage_mean': float(np.mean(coverages)),
        'coverage_min': float(np.min(coverages)),
        'coverage_p05': float(np.percentile(coverages, 5)),
        'n_traces': n_traces,
    }


def auto_widen_envelope(
    traces: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mad: np.ndarray,
    target_coverage: float = 0.95,
    min_coverage: float = 0.90,
    max_iterations: int = 5,
    k_step: float = 0.5,
) -> tuple:
    """
    自动增宽包络，直到覆盖率达到目标。
    
    如果 coverage_mean < target_coverage 或 coverage_min < min_coverage，
    自动增加 padding: lower -= k * mad, upper += k * mad，直到满足条件。
    
    Args:
        traces: 多条正常曲线 (n_traces, n_points)
        lower: 下包络
        upper: 上包络
        mad: MAD尺度统计量（用于padding）
        target_coverage: 目标平均覆盖率
        min_coverage: 最小单曲线覆盖率
        max_iterations: 最大迭代次数
        k_step: 每次迭代增加的k步长
        
    Returns:
        (lower, upper, coverage_stats, k_final): 增宽后的包络、覆盖率统计、最终k值
    """
    lower = lower.copy()
    upper = upper.copy()
    k = 0.0
    
    for iteration in range(max_iterations + 1):
        coverage_stats = compute_envelope_coverage(traces, lower, upper)
        
        coverage_mean = coverage_stats['coverage_mean']
        coverage_min_val = coverage_stats['coverage_min']
        
        # 检查是否满足条件
        if coverage_mean >= target_coverage and coverage_min_val >= min_coverage:
            print(f"[包络覆盖] 迭代{iteration}: k={k:.2f}, coverage_mean={coverage_mean:.4f}, coverage_min={coverage_min_val:.4f} - 满足要求")
            break
        
        if iteration < max_iterations:
            # 增加padding
            k += k_step
            lower = lower - k_step * mad
            upper = upper + k_step * mad
            print(f"[包络覆盖] 迭代{iteration}: k={k:.2f}, coverage_mean={coverage_mean:.4f}, coverage_min={coverage_min_val:.4f} - 自动增宽包络")
    
    return lower, upper, coverage_stats, k