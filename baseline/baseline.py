import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from .config import BAND_RANGES, K_LIST, N_POINTS, SINGLE_BAND_MODE, COVERAGE_MEAN_MIN, COVERAGE_MIN_MIN

# ============ 基线/包络配置参数 (2024-01 v2 重构) ============
# 目标：
# 1. RRS 默认使用 pointwise median（不做强平滑），保留原始细节起伏
# 2. 包络平滑、宽度合理、不在局部频段突然鼓包

# RRS 平滑参数（默认关闭或很小）
RRS_SMOOTH_ENABLED = False        # 默认不对 RRS 做平滑
RRS_SMOOTH_WINDOW = 15            # 若启用，使用小窗口（≤15）
RRS_SMOOTH_POLY = 3               # Savitzky-Golay 多项式阶数

# 包络宽度参数（只平滑包络，不平滑 RRS）
SIGMA_SMOOTH_GAUSSIAN = 8         # sigma 平滑高斯核 sigma（稍大以避免毛刺）
WIDTH_MIN_DB = 0.03               # 包络最小宽度 (dB)
WIDTH_MAX_DB = 0.80               # 包络最大宽度 (dB)，放宽以避免硬切
WIDTH_POSTSMOOTH_GAUSSIAN = 12    # width 后平滑高斯核 sigma（使包络缓慢变化）

# 覆盖率搜索参数
K_SEARCH_MIN = 1.5                # k 搜索下界
K_SEARCH_MAX = 6.0                # k 搜索上界
K_SEARCH_STEP = 0.1               # k 搜索步长

# 滑窗覆盖率检查
SLIDING_WINDOW_SIZE = 41          # 滑窗大小
SLIDING_COVERAGE_MIN = 0.93       # 任意滑窗覆盖率最低要求


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
        coverage_per_point: 每个频点的覆盖率
    """
    n_traces = traces.shape[0]
    n_points = traces.shape[1]
    coverages = []
    
    # 每条曲线的覆盖率
    for i in range(n_traces):
        trace = traces[i]
        in_bounds = (trace >= lower) & (trace <= upper)
        coverage = np.mean(in_bounds)
        coverages.append(coverage)
    
    # 每个频点的覆盖率
    point_coverages = []
    for j in range(n_points):
        col = traces[:, j]
        in_bounds = (col >= lower[j]) & (col <= upper[j])
        point_coverages.append(np.mean(in_bounds))
    
    return {
        'coverage_mean': float(np.mean(coverages)),
        'coverage_min': float(np.min(coverages)),
        'coverage_per_trace': coverages,
        'coverage_per_point': point_coverages,
        'coverage_point_5th': float(np.percentile(point_coverages, 5)),
        'coverage_point_50th': float(np.percentile(point_coverages, 50)),
        'coverage_point_95th': float(np.percentile(point_coverages, 95)),
    }


def compute_rrs_robust(traces, smooth=RRS_SMOOTH_ENABLED, 
                       smooth_window=RRS_SMOOTH_WINDOW, 
                       smooth_poly=RRS_SMOOTH_POLY):
    """计算鲁棒的 RRS 基线（pointwise median，保留原始细节）。
    
    新版逻辑（2024-01 v2）：
    - 默认不做强平滑：直接使用 pointwise median
    - 可选轻微平滑：若启用，仅对聚合后的 RRS 做小窗口（≤15）平滑
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)，正常曲线数据
    smooth : bool
        是否对 RRS 做平滑（默认 False）
    smooth_window : int
        平滑窗口大小（默认 15）
    smooth_poly : int
        平滑多项式阶数
        
    Returns
    -------
    np.ndarray
        RRS 基线，shape (n_points,)
    """
    n_traces, n_points = traces.shape
    
    # 核心：直接计算 pointwise median（不对每条曲线预平滑）
    rrs = np.median(traces, axis=0)
    
    # 可选：对聚合后的 RRS 做轻微平滑（默认关闭）
    if smooth and n_points >= smooth_window:
        rrs = savgol_filter(rrs, smooth_window, smooth_poly, mode='nearest')
    
    return rrs


def compute_robust_sigma(traces, rrs, gaussian_sigma=SIGMA_SMOOTH_GAUSSIAN):
    """计算每个频点的鲁棒 sigma（用于包络宽度）。
    
    算法步骤：
    1. 计算残差 res_i = y_i - rrs
    2. 对每个频点计算 robust sigma: sigma_j = 1.4826 * MAD({res_i[j]})
    3. 对 sigma 做高斯平滑
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)
    rrs : np.ndarray
        Shape (n_points,)，RRS 基线
    gaussian_sigma : float
        高斯平滑核的 sigma
        
    Returns
    -------
    np.ndarray
        平滑后的 sigma，shape (n_points,)
    """
    # 计算残差
    residuals = traces - rrs  # Shape (n_traces, n_points)
    
    # 对每个频点计算 robust sigma (MAD-based)
    n_points = rrs.shape[0]
    sigma_raw = np.zeros(n_points)
    
    for j in range(n_points):
        res_j = residuals[:, j]
        mad = np.median(np.abs(res_j - np.median(res_j)))
        sigma_raw[j] = 1.4826 * mad
    
    # 设置最小 sigma 避免除零
    sigma_raw = np.maximum(sigma_raw, 1e-6)
    
    # 高斯平滑（使用 reflect mode 处理端点）
    sigma_smooth = gaussian_filter1d(sigma_raw, sigma=gaussian_sigma, mode='reflect')
    
    return sigma_smooth


def find_optimal_k(traces, rrs, sigma_smooth, target_coverage=COVERAGE_MEAN_MIN,
                   k_min=K_SEARCH_MIN, k_max=K_SEARCH_MAX, k_step=K_SEARCH_STEP):
    """二分搜索找到满足覆盖率要求的最小 k。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)
    rrs : np.ndarray
        Shape (n_points,)
    sigma_smooth : np.ndarray
        Shape (n_points,)
    target_coverage : float
        目标覆盖率
    k_min, k_max, k_step : float
        搜索范围和步长
        
    Returns
    -------
    float
        找到的最优 k
    """
    best_k = k_max
    
    # 网格搜索找最小 k
    for k in np.arange(k_min, k_max + k_step, k_step):
        upper = rrs + k * sigma_smooth
        lower = rrs - k * sigma_smooth
        
        coverage = compute_coverage(traces, upper, lower)
        
        if coverage['coverage_mean'] >= target_coverage:
            best_k = k
            break
    
    return best_k


def constrain_envelope_width(upper, lower, rrs, 
                             width_min=WIDTH_MIN_DB, 
                             width_max=WIDTH_MAX_DB,
                             postsmooth_gaussian=WIDTH_POSTSMOOTH_GAUSSIAN):
    """约束并平滑包络宽度，避免局部鼓包。
    
    Parameters
    ----------
    upper, lower : np.ndarray
        初始包络边界
    rrs : np.ndarray
        RRS 基线
    width_min, width_max : float
        宽度约束范围 (dB)
    postsmooth_gaussian : float
        宽度平滑高斯核 sigma
        
    Returns
    -------
    tuple
        (constrained_upper, constrained_lower, width)
    """
    # 计算宽度
    width = upper - lower
    
    # 约束宽度范围
    width = np.clip(width, width_min, width_max)
    
    # 对宽度做平滑
    width_smooth = gaussian_filter1d(width, sigma=postsmooth_gaussian, mode='reflect')
    
    # 根据平滑后的宽度重建包络
    half_width = width_smooth / 2
    constrained_upper = rrs + half_width
    constrained_lower = rrs - half_width
    
    return constrained_upper, constrained_lower, width_smooth


def compute_sliding_coverage(traces, upper, lower, window_size=SLIDING_WINDOW_SIZE):
    """计算滑窗覆盖率（检查是否有局部覆盖率过低的区域）。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)
    upper, lower : np.ndarray
        包络边界
    window_size : int
        滑窗大小
        
    Returns
    -------
    np.ndarray
        每个滑窗位置的覆盖率
    """
    n_traces, n_points = traces.shape
    n_windows = n_points - window_size + 1
    
    if n_windows <= 0:
        return np.array([1.0])
    
    window_coverages = []
    for start in range(n_windows):
        end = start + window_size
        window_in_bounds = 0
        window_total = n_traces * window_size
        
        for i in range(n_traces):
            trace_window = traces[i, start:end]
            upper_window = upper[start:end]
            lower_window = lower[start:end]
            window_in_bounds += np.sum((trace_window >= lower_window) & (trace_window <= upper_window))
        
        window_coverages.append(window_in_bounds / window_total)
    
    return np.array(window_coverages)


def auto_expand_envelope(
    frequency, traces, 
    target_coverage_mean=COVERAGE_MEAN_MIN,
    target_coverage_min=COVERAGE_MIN_MIN,
    rrs_smooth=RRS_SMOOTH_ENABLED,
):
    """新版自适应包络计算（2024-01 v2 重构）。
    
    目标：
    1. RRS 使用 pointwise median（默认不平滑），保留原始细节
    2. 包络平滑、宽度合理、不在局部频段突然鼓包
    3. 覆盖率满足全局和滑窗要求
    
    算法步骤：
    1. 使用 compute_rrs_robust() 计算 RRS（默认不平滑）
    2. 使用 compute_robust_sigma() 计算每点的鲁棒 sigma
    3. 使用 find_optimal_k() 找到满足覆盖率的最小 k
    4. 使用 constrain_envelope_width() 约束并平滑包络宽度
    5. 检查滑窗覆盖率，必要时微调
    
    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis.
    traces : np.ndarray
        Shape (n_traces, n_points), 仅正常曲线数据.
    target_coverage_mean : float
        目标平均覆盖率 (default 0.97).
    target_coverage_min : float
        目标最小覆盖率 (default 0.93).
    rrs_smooth : bool
        是否对 RRS 做轻微平滑（默认 False）
        
    Returns
    -------
    tuple
        (rrs, (upper, lower), coverage_info)
    """
    n_traces, n_points = traces.shape
    print(f"[Baseline] Computing RRS from {n_traces} normal traces, {n_points} frequency points")
    print(f"[Baseline] RRS smoothing: {'enabled' if rrs_smooth else 'disabled (pointwise median)'}")
    
    # Step 1: 计算 RRS（默认使用 pointwise median，不平滑）
    rrs = compute_rrs_robust(traces, smooth=rrs_smooth)
    
    # 验证 RRS 贴合度（不平滑时应该完全一致）
    pointwise_median = np.median(traces, axis=0)
    rrs_mae = np.mean(np.abs(rrs - pointwise_median))
    print(f"[Baseline] RRS vs pointwise median MAE: {rrs_mae:.6f} dB")
    
    # Step 2: 计算鲁棒 sigma（对 sigma 做平滑，避免包络毛刺）
    sigma_smooth = compute_robust_sigma(traces, rrs)
    
    # Step 3: 找到满足覆盖率的最小 k
    chosen_k = find_optimal_k(traces, rrs, sigma_smooth, target_coverage_mean)
    print(f"[Baseline] Chosen k={chosen_k:.2f} for target coverage {target_coverage_mean:.2f}")
    
    # 计算初始包络
    upper0 = rrs + chosen_k * sigma_smooth
    lower0 = rrs - chosen_k * sigma_smooth
    
    # Step 4: 约束并平滑包络宽度（使包络缓慢变化）
    upper, lower, width = constrain_envelope_width(upper0, lower0, rrs)
    
    # Step 5: 检查滑窗覆盖率
    sliding_cov = compute_sliding_coverage(traces, upper, lower)
    sliding_cov_min = float(np.min(sliding_cov))
    
    # 如果滑窗覆盖率不满足要求，逐步扩大
    if sliding_cov_min < SLIDING_COVERAGE_MIN:
        print(f"[Baseline] Sliding coverage min={sliding_cov_min:.4f} < {SLIDING_COVERAGE_MIN}, expanding...")
        for extra_k in np.arange(0.1, 1.5, 0.1):
            upper_exp = rrs + (chosen_k + extra_k) * sigma_smooth
            lower_exp = rrs - (chosen_k + extra_k) * sigma_smooth
            upper, lower, width = constrain_envelope_width(upper_exp, lower_exp, rrs)
            sliding_cov = compute_sliding_coverage(traces, upper, lower)
            sliding_cov_min = float(np.min(sliding_cov))
            if sliding_cov_min >= SLIDING_COVERAGE_MIN:
                chosen_k += extra_k
                print(f"[Baseline] Expanded to k={chosen_k:.2f}, sliding coverage min={sliding_cov_min:.4f}")
                break
    
    # 计算最终覆盖率
    coverage = compute_coverage(traces, upper, lower)
    
    print(f"[Baseline] Final coverage: mean={coverage['coverage_mean']:.4f}, "
          f"min={coverage['coverage_min']:.4f}")
    print(f"[Baseline] Sliding window coverage: min={sliding_cov_min:.4f}")
    print(f"[Baseline] Width stats: min={width.min():.4f}, median={np.median(width):.4f}, "
          f"max={width.max():.4f} dB")
    
    # 记录所有元数据
    coverage['k_final'] = chosen_k
    coverage['target_coverage_mean'] = target_coverage_mean
    coverage['target_coverage_min'] = target_coverage_min
    coverage['rrs_mae'] = rrs_mae
    coverage['rrs_smooth_enabled'] = rrs_smooth
    coverage['width_min'] = float(width.min())
    coverage['width_median'] = float(np.median(width))
    coverage['width_max'] = float(width.max())
    coverage['width_smoothness'] = float(np.std(np.diff(width)))
    coverage['sliding_coverage_min'] = sliding_cov_min
    coverage['n_normal_traces'] = n_traces
    coverage['smooth_params'] = {
        'rrs_smooth_enabled': RRS_SMOOTH_ENABLED,
        'rrs_smooth_window': RRS_SMOOTH_WINDOW,
        'sigma_smooth_gaussian': SIGMA_SMOOTH_GAUSSIAN,
        'width_postsmooth_gaussian': WIDTH_POSTSMOOTH_GAUSSIAN,
    }
    
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


# =============================================================================
# 新版包络构建算法 (2024-01 v6) - 用户详细规格实现
# =============================================================================
# 厂商分段容差（以 RRS 为中心的允许偏差，单位 dBm 差值）
# 10MHz~100MHz: ±0.80
# 100MHz~3.25GHz: ±0.40 (稳定频段)
# 3.25GHz~5.25GHz: ±0.60
# 5.25GHz~8.2GHz: ±0.80

# 稳定频段（用于 offset 估计）
STABLE_BAND_START_HZ = 100e6
STABLE_BAND_END_HZ = 3.25e9

# 平滑参数
SMOOTH_SIGMA_HZ_DEFAULT = 200e6  # 200MHz 高斯平滑尺度


def vendor_tolerance_dbm(frequency_hz):
    """按频段返回厂商容差 (half-width, dBm差值 = dB)。
    
    厂商分段容差:
    - 10MHz~100MHz: ±0.80 dB
    - 100MHz~3.25GHz: ±0.40 dB (最紧/稳定频段)
    - 3.25GHz~5.25GHz: ±0.60 dB
    - 5.25GHz~8.2GHz: ±0.80 dB
    
    Parameters
    ----------
    frequency_hz : np.ndarray
        频率轴 (Hz)
        
    Returns
    -------
    np.ndarray
        每个频点的容差 (dBm差值)
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    tol = np.full_like(f, 0.80, dtype=np.float64)
    
    tol[(f >= 10e6) & (f < 100e6)] = 0.80
    tol[(f >= 100e6) & (f < 3.25e9)] = 0.40
    tol[(f >= 3.25e9) & (f < 5.25e9)] = 0.60
    tol[(f >= 5.25e9) & (f <= 8.2e9)] = 0.80
    
    return tol


def compute_rrs_pointwise_median(traces):
    """计算 RRS = pointwise median，不做任何平滑。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)
        
    Returns
    -------
    np.ndarray
        RRS 基线，shape (n_points,)
    """
    return np.median(traces, axis=0)


def estimate_trace_offsets(traces, rrs, frequency_hz,
                          use_stable_band=True,
                          max_offset_db=0.4):
    """在稳定频段 (100MHz~3.25GHz) 估计每条曲线的全局偏移。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)
    rrs : np.ndarray
        RRS 基线
    frequency_hz : np.ndarray
        频率轴
    use_stable_band : bool
        是否仅使用稳定频段估计偏移
    max_offset_db : float
        最大允许偏移 (会 clip 到 ±max_offset_db)
        
    Returns
    -------
    np.ndarray
        每条曲线的偏移，shape (n_traces,)
    """
    n_traces = traces.shape[0]
    offsets = np.zeros(n_traces)
    
    if use_stable_band:
        mask = (frequency_hz >= STABLE_BAND_START_HZ) & (frequency_hz <= STABLE_BAND_END_HZ)
        if np.sum(mask) < 10:
            mask = np.ones(len(frequency_hz), dtype=bool)
    else:
        mask = np.ones(len(frequency_hz), dtype=bool)
    
    for i in range(n_traces):
        diff = traces[i, mask] - rrs[mask]
        offset = np.median(diff)
        offsets[i] = np.clip(offset, -max_offset_db, max_offset_db)
    
    return offsets


def drop_outlier_traces_by_spec(traces, rrs, frequency_hz, tol,
                                exceed_rate_threshold=0.10,
                                max_exceed_threshold=0.80,
                                p95_exceed_threshold=0.30,
                                max_drop_fraction=0.20):
    """根据厂商容差超限剔除异常曲线。
    
    Parameters
    ----------
    traces : np.ndarray
        Shape (n_traces, n_points)，已去除偏移的曲线
    rrs : np.ndarray
        RRS 基线
    frequency_hz : np.ndarray
        频率轴
    tol : np.ndarray
        每个频点的厂商容差
    exceed_rate_threshold : float
        超限点比例阈值 (默认 0.10，即超过 10% 的点超出容差)
    max_exceed_threshold : float
        单点最大超限阈值 (dB)
    p95_exceed_threshold : float
        95 分位超限阈值 (dB)
    max_drop_fraction : float
        最大剔除比例 (默认 0.20)
        
    Returns
    -------
    tuple
        (valid_mask, dropped_indices, stats)
    """
    n_traces = traces.shape[0]
    abs_resid = np.abs(traces - rrs)
    
    # exceed = max(0, |resid| - tol)
    exceed = np.maximum(0, abs_resid - tol)
    
    valid_mask = np.ones(n_traces, dtype=bool)
    stats_list = []
    
    for i in range(n_traces):
        exc_i = exceed[i]
        exceed_rate = np.mean(exc_i > 0)
        max_exceed = np.max(exc_i)
        p95_exceed = np.percentile(exc_i, 95) if len(exc_i) > 0 else 0.0
        
        stats_list.append({
            'exceed_rate': exceed_rate,
            'max_exceed': max_exceed,
            'p95_exceed': p95_exceed,
        })
        
        # 任一超过阈值则标记为异常
        if (exceed_rate > exceed_rate_threshold or
            max_exceed > max_exceed_threshold or
            p95_exceed > p95_exceed_threshold):
            valid_mask[i] = False
    
    n_outliers = n_traces - np.sum(valid_mask)
    
    # 限制最大剔除比例
    if n_outliers > n_traces * max_drop_fraction:
        print(f"[Outlier] Warning: {n_outliers} outliers ({n_outliers/n_traces:.1%}) exceeds max_drop_fraction, keeping all")
        valid_mask = np.ones(n_traces, dtype=bool)
        n_outliers = 0
    
    dropped_indices = list(np.where(~valid_mask)[0])
    
    return valid_mask, dropped_indices, {
        'n_outliers': n_outliers,
        'thresholds': {
            'exceed_rate': exceed_rate_threshold,
            'max_exceed': max_exceed_threshold,
            'p95_exceed': p95_exceed_threshold,
        },
        'trace_stats': stats_list,
    }


def compute_envelope_from_vendor_plus_quantile(
    frequency_hz,
    traces,
    *,
    # Offset 归一化参数
    normalize_offset=True,
    max_offset_db=0.4,
    # 异常剔除参数
    drop_outliers=True,
    exceed_rate_threshold=0.10,
    max_exceed_threshold=0.80,
    p95_exceed_threshold=0.30,
    # 包络参数
    quantile=0.97,
    smooth_sigma_hz=SMOOTH_SIGMA_HZ_DEFAULT,
    extra_clip_max=0.25,
    # 目标覆盖率
    target_coverage_mean=0.97,
    target_coverage_min=0.93,
):
    """基于厂商容差 + 分位数残差构建包络（用户指定算法）。
    
    算法步骤:
    1) rrs0 = median(traces)
    2) offsets = estimate_trace_offsets() 在稳定频段; traces_norm = traces - offsets
    3) rrs = median(traces_norm)
    4) tol = vendor_tolerance_dbm(f)
    5) 可选：drop_outlier_traces_by_spec 后重算 rrs
    6) abs_resid = |traces_norm - rrs|
    7) point_q = quantile(abs_resid, q=0.97, axis=0)
    8) raw_extra = max(0, point_q - tol)
    9) extra_smooth = gaussian_filter1d(raw_extra, sigma=sigma_points)
    10) extra_smooth clip 到 [0, extra_clip_max]
    11) half_width = tol + extra_smooth
    12) upper/lower = rrs ± half_width
    
    Parameters
    ----------
    frequency_hz : np.ndarray
        频率轴 (Hz)
    traces : np.ndarray
        Shape (n_traces, n_points), 正常曲线数据 (dBm)
    normalize_offset : bool
        是否做全局偏移归一化
    max_offset_db : float
        最大允许偏移
    drop_outliers : bool
        是否剔除异常曲线
    exceed_rate_threshold, max_exceed_threshold, p95_exceed_threshold : float
        异常检测阈值
    quantile : float
        残差分位数 (默认 0.97)
    smooth_sigma_hz : float
        extra 平滑的高斯 sigma (Hz)，会换算成点数
    extra_clip_max : float
        extra 的最大值 (dBm)
    target_coverage_mean, target_coverage_min : float
        目标覆盖率
        
    Returns
    -------
    tuple
        (rrs, upper, lower, info_dict)
    """
    n_traces, n_points = traces.shape
    print(f"[Envelope v6] Computing from {n_traces} traces, {n_points} points")
    print(f"[Envelope v6] Y-axis unit: dBm")
    
    # Step 1: 初始 RRS
    rrs0 = compute_rrs_pointwise_median(traces)
    
    # Step 2: 估计偏移并归一化
    if normalize_offset:
        offsets = estimate_trace_offsets(traces, rrs0, frequency_hz,
                                        use_stable_band=True,
                                        max_offset_db=max_offset_db)
        traces_norm = traces - offsets[:, np.newaxis]
        offset_stats = {
            'mean': float(np.mean(offsets)),
            'std': float(np.std(offsets)),
            'p50': float(np.median(offsets)),
            'p95': float(np.percentile(np.abs(offsets), 95)),
        }
        print(f"[Envelope v6] Offset normalization: mean={offset_stats['mean']:.4f}, "
              f"std={offset_stats['std']:.4f}, p95_abs={offset_stats['p95']:.4f} dB")
    else:
        traces_norm = traces.copy()
        offsets = np.zeros(n_traces)
        offset_stats = {}
    
    # Step 3: 重新计算 RRS (基于归一化后的曲线)
    rrs = compute_rrs_pointwise_median(traces_norm)
    
    # Step 4: 厂商容差
    tol = vendor_tolerance_dbm(frequency_hz)
    
    # Step 5: 可选异常剔除
    dropped_indices = []
    outlier_stats = {}
    if drop_outliers:
        valid_mask, dropped_indices, outlier_stats = drop_outlier_traces_by_spec(
            traces_norm, rrs, frequency_hz, tol,
            exceed_rate_threshold=exceed_rate_threshold,
            max_exceed_threshold=max_exceed_threshold,
            p95_exceed_threshold=p95_exceed_threshold,
        )
        if len(dropped_indices) > 0:
            traces_norm = traces_norm[valid_mask]
            # 重算 RRS
            rrs = compute_rrs_pointwise_median(traces_norm)
            print(f"[Envelope v6] Dropped {len(dropped_indices)} outlier traces, recomputed RRS")
    
    n_valid = traces_norm.shape[0]
    
    # Step 6: 计算残差绝对值
    abs_resid = np.abs(traces_norm - rrs)
    
    # Step 7: 逐点分位数
    point_q = np.percentile(abs_resid, quantile * 100, axis=0)
    
    # Step 8: raw_extra = max(0, point_q - tol)
    raw_extra = np.maximum(0, point_q - tol)
    
    # Step 9: 平滑 extra (使用 Hz 参数换算到点数)
    df = frequency_hz[1] - frequency_hz[0] if len(frequency_hz) > 1 else 10e6
    sigma_points = smooth_sigma_hz / df
    
    if sigma_points > 0:
        extra_smooth = gaussian_filter1d(raw_extra, sigma=sigma_points, mode='reflect')
    else:
        extra_smooth = raw_extra.copy()
    
    # Step 10: clip extra 到 [0, extra_clip_max]
    extra_smooth = np.clip(extra_smooth, 0, extra_clip_max)
    
    # Step 11: half_width = tol + extra_smooth
    half_width = tol + extra_smooth
    
    # Step 12: 构建包络
    upper = rrs + half_width
    lower = rrs - half_width
    
    # 计算覆盖率
    coverage = compute_coverage(traces_norm, upper, lower)
    
    # 如果覆盖率不满足要求，逐步扩大 extra
    expansion_iterations = 0
    while (coverage['coverage_mean'] < target_coverage_mean or
           coverage['coverage_min'] < target_coverage_min):
        expansion_iterations += 1
        if expansion_iterations > 30:
            print(f"[Envelope v6] Warning: Coverage targets not met after 30 iterations")
            break
        
        extra_smooth = extra_smooth + 0.01
        extra_smooth = np.clip(extra_smooth, 0, extra_clip_max)
        half_width = tol + extra_smooth
        upper = rrs + half_width
        lower = rrs - half_width
        coverage = compute_coverage(traces_norm, upper, lower)
    
    if expansion_iterations > 0:
        print(f"[Envelope v6] Expanded envelope in {expansion_iterations} iterations")
    
    # 输出统计
    width = upper - lower
    print(f"[Envelope v6] half_width: min={half_width.min():.4f}, "
          f"median={np.median(half_width):.4f}, max={half_width.max():.4f} dBm")
    print(f"[Envelope v6] Coverage: mean={coverage['coverage_mean']:.4f}, "
          f"min={coverage['coverage_min']:.4f}")
    
    # 构建 info 字典
    info = {
        'version': 'v6',
        'n_traces': n_traces,
        'n_valid_traces': n_valid,
        'n_points': n_points,
        'normalize_offset': normalize_offset,
        'offset_stats': offset_stats,
        'dropped_trace_ids': dropped_indices,
        'outlier_stats': outlier_stats,
        'quantile': quantile,
        'smooth_sigma_hz': smooth_sigma_hz,
        'sigma_points': sigma_points,
        'extra_clip_max': extra_clip_max,
        'coverage_mean': coverage['coverage_mean'],
        'coverage_min': coverage['coverage_min'],
        'half_width_min': float(np.min(half_width)),
        'half_width_p50': float(np.median(half_width)),
        'half_width_max': float(np.max(half_width)),
        'width_min': float(np.min(width)),
        'width_median': float(np.median(width)),
        'width_max': float(np.max(width)),
        'rrs_median_level': float(np.median(rrs)),
        'offsets': offsets.tolist() if normalize_offset else [],
        'vendor_tolerance_db': tol.tolist(),
        'extra_smooth': extra_smooth.tolist(),
    }
    
    return rrs, upper, lower, info