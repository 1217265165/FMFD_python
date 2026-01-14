import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Single-band mode flag: When True, preamp is disabled and switch-point step injection is disabled
SINGLE_BAND_MODE = True

# ============ 故障严重度配置 (2024-01 新增) ============
# 每种故障注入支持三档严重度: light, mid, severe

SEVERITY_LEVELS = ['light', 'mid', 'severe']

# 幅度失准参数（按严重度分档）
AMP_MISCAL_PARAMS = {
    'light':  {'gain_sigma': 0.005, 'bias_sigma': 0.15, 'comp_mean': 0.005, 'comp_std': 0.002},
    'mid':    {'gain_sigma': 0.010, 'bias_sigma': 0.25, 'comp_mean': 0.010, 'comp_std': 0.004},
    'severe': {'gain_sigma': 0.020, 'bias_sigma': 0.40, 'comp_mean': 0.015, 'comp_std': 0.006},
}

# 参考电平失准参数（按严重度分档）
REFLEVEL_PARAMS = {
    'light':  {'offset_sigma': 0.2, 'offset_clip': 0.6, 'comp_mean': 0.06, 'comp_std': 0.03},
    'mid':    {'offset_sigma': 0.4, 'offset_clip': 0.8, 'comp_mean': 0.10, 'comp_std': 0.05},
    'severe': {'offset_sigma': 0.7, 'offset_clip': 1.0, 'comp_mean': 0.15, 'comp_std': 0.07},
}


# 辅助：估计局部/全局 sigma，用于自适应幅度/噪声
def _estimate_sigma(amp, window_frac=0.02, min_window=21):
    x = np.asarray(amp, dtype=float)
    n = len(x)
    w = max(min_window, int(round(n * window_frac)))
    if w % 2 == 0:
        w += 1
    half = w // 2
    pad = np.pad(x, (half, half), mode="edge")
    sig = np.zeros(n)
    for i in range(n):
        seg = pad[i:i + w]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sig[i] = max(1e-6, 1.4826 * mad)
    return sig, float(np.median(sig))

# -------------------------
# 系统级故障/畸变注入（自适应幅度）
# -------------------------
def inject_amplitude_miscal(amp, gain=None, bias=None, comp=None, rng=None, 
                            severity='mid', return_params=False):
    """
    幅度失准：A' = gain*A + bias + comp*A^2
    若 gain/bias/comp 未给出，则基于严重度和当前曲线的 σ 自适应随机生成。
    
    优化版（2024-01）：
    - 支持 severity 参数：light/mid/severe 三档
    - 避免与参考电平失准重叠太多
    
    Parameters
    ----------
    severity : str
        故障严重度: 'light', 'mid', 'severe'
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
    """
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    
    # 获取对应严重度的参数
    params_cfg = AMP_MISCAL_PARAMS.get(severity, AMP_MISCAL_PARAMS['mid'])
    
    if gain is None:
        gain = 1.0 + rng.normal(0, params_cfg['gain_sigma'] * max(1.0, sig_med))
    if bias is None:
        bias = rng.normal(0, params_cfg['bias_sigma'] * max(0.2, sig_med))
    if comp is None:
        comp = rng.normal(params_cfg['comp_mean'], params_cfg['comp_std'])
        # 限制 comp 范围，避免二次项把低频段放大成类似 ref 的整体偏移
        comp = np.clip(comp, -0.03, 0.03)
    
    result = gain * amp + bias + comp * (amp ** 2)
    
    if return_params:
        return result, {
            'severity': severity,
            'gain': float(gain),
            'bias': float(bias),
            'comp': float(comp),
        }
    return result

def inject_freq_miscal(frequency, amp, delta_f=None, rng=None, return_params=False):
    """
    频率失准：频率轴整体平移后重采样；delta_f 未给出时按带宽 ppm 生成。
    
    优化版（2024-01）：
    - 使用 cubic 插值替代 linear，实现更平滑的频率响应过渡
    - 添加多次平滑操作减小插值误差
    
    Parameters
    ----------
    frequency : array
        Frequency axis
    amp : array
        Amplitude data
    delta_f : float, optional
        Frequency shift in Hz. If None, generated from ppm.
    rng : Generator, optional
        Random number generator
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
        
    Returns
    -------
    array or (array, dict)
        Modified amplitude, optionally with injection parameters
    """
    rng = rng or np.random.default_rng()
    bw = frequency[-1] - frequency[0]
    step_hz = frequency[1] - frequency[0] if len(frequency) > 1 else 1e7
    
    if delta_f is None:
        # Increased ppm range to produce >= 1 bin shift (10MHz)
        # With 820 points and 8.19GHz bandwidth, 1 bin = 10MHz
        # ppm = delta_f / bw, so for 1 bin: ppm = 10MHz / 8.19GHz ≈ 0.0012 (1200ppm)
        # Previous: ±150ppm was too small to cause meaningful shift
        # New: ±500-3000ppm to ensure >= 1 bin shift for detectability
        ppm = rng.uniform(-3000e-6, 3000e-6)
        # Ensure minimum absolute ppm for detectability
        if abs(ppm) < 500e-6:
            # If ppm is too small, set it to ±500ppm with random sign
            ppm = rng.choice([-1, 1]) * 500e-6
        delta_f = ppm * bw
    else:
        ppm = delta_f / bw if bw > 0 else 0
    
    f_shift = frequency + delta_f
    # Optimized: use cubic interpolation for smoother frequency response
    interp = interp1d(f_shift, amp, kind="cubic", bounds_error=False, fill_value="extrapolate")
    result = interp(frequency)
    
    # Additional smoothing pass to reduce interpolation artifacts
    try:
        from scipy.signal import savgol_filter
        # Gentle smoothing: window=11, polyorder=3
        if len(result) >= 11:
            result = savgol_filter(result, window_length=11, polyorder=3)
    except ImportError:
        pass
    
    # Calculate effective shift in bins
    shift_bins = delta_f / step_hz if step_hz > 0 else 0
    
    params = {
        'delta_f_hz': float(delta_f),
        'ppm': float(ppm * 1e6),  # Convert to actual ppm
        'shift_bins': float(shift_bins),
        'bandwidth_hz': float(bw),
    }
    
    if return_params:
        return result, params
    return result

def inject_reflevel_miscal(frequency, amp, band_ranges, step_biases=None, compression_coef=None,
                           compression_start_percent=0.8, rng=None, single_band_mode=None,
                           return_params=False, severity='mid'):
    """
    参考电平失准：在切换点施加错误步进；高幅度区压缩。
    step_biases/compression_coef 未给出时按严重度和 σ 自适应随机生成。
    
    优化版（2024-01）：
    - 支持 severity 参数：light/mid/severe 三档
    - Type-A 全局偏移限制在 [-1.0, +1.0] dB
    - Type-B 压缩可限制在特定频段（高频段）
    
    In single-band mode (single_band_mode=True or SINGLE_BAND_MODE global):
    - Switch-point step injection is DISABLED
    - Type-A: Global offset (reference level shift)
    - Type-B: High-amplitude compression/saturation
    
    Parameters
    ----------
    severity : str
        故障严重度: 'light', 'mid', 'severe'
    return_params : bool
        If True, return (curve, params_dict) instead of just curve
    """
    rng = rng or np.random.default_rng()
    out = amp.copy()
    _, sig_med = _estimate_sigma(amp)
    
    # 获取对应严重度的参数
    params_cfg = REFLEVEL_PARAMS.get(severity, REFLEVEL_PARAMS['mid'])
    
    # Determine if single-band mode is active
    if single_band_mode is None:
        single_band_mode = SINGLE_BAND_MODE
    
    params = {
        'single_band_mode': single_band_mode,
        'severity': severity,
        'ref_type': 'none',
        'global_offset_db': 0.0,
        'compression_coef': 0.0,
        'compression_threshold_db': 0.0,
    }
    
    # Step injection at switch points (DISABLED in single-band mode)
    if not single_band_mode and len(band_ranges) > 1:
        if step_biases is None:
            step_biases = [rng.normal(0.6 * sig_med, 0.2 * max(0.2, sig_med))
                           for _ in range(len(band_ranges) - 1)]
        for i in range(len(band_ranges) - 1):
            end_f = band_ranges[i][1]
            m_end = np.argmin(np.abs(frequency - end_f))
            out[m_end:] += step_biases[i]
        params['ref_type'] = 'step'
        params['step_biases'] = [float(b) for b in step_biases]
    else:
        # Type-A global offset with severity-based parameters
        global_offset = rng.normal(0, params_cfg['offset_sigma'] * max(0.5, sig_med))
        # Clip to physical limits
        global_offset = np.clip(global_offset, -params_cfg['offset_clip'], params_cfg['offset_clip'])
        # Ensure minimum offset for detectability
        if abs(global_offset) < 0.10:
            global_offset = 0.10 * np.sign(global_offset) if global_offset != 0 else rng.choice([-1, 1]) * 0.10
        out = out + global_offset
        params['ref_type'] = 'global_offset'
        params['global_offset_db'] = float(global_offset)
    
    # Type-B compression with severity-based parameters
    if compression_coef is None:
        compression_coef = abs(rng.normal(params_cfg['comp_mean'], params_cfg['comp_std']))
        # Ensure minimum compression for detectability
        compression_coef = max(0.03, compression_coef)
    
    thr = np.percentile(out, 100 * compression_start_percent)
    mask = out >= thr
    out[mask] = out[mask] - compression_coef * (out[mask] - thr)
    
    params['compression_coef'] = float(compression_coef)
    params['compression_threshold_db'] = float(thr)
    params['compression_start_percent'] = float(compression_start_percent)
    
    if return_params:
        return out, params
    return out

# -------------------------
# 模块级示例畸变（自适应幅度）
# -------------------------

# PREAMP DISABLED: inject_preamp_degradation is kept for backward compatibility
# but will raise an error in single-band mode and should not be called
def inject_preamp_degradation(frequency, amp, hf_drop_db=None, rng=None):
    """
    前置放大器衰减：随频率线性下滑，高频端下降 hf_drop_db。
    
    **DISABLED IN SINGLE-BAND MODE**
    
    In single-band mode (10MHz-8.2GHz with preamp OFF), this function
    should NOT be called. It is preserved for backward compatibility only.
    """
    if SINGLE_BAND_MODE:
        raise ValueError(
            "inject_preamp_degradation is DISABLED in single-band mode. "
            "Preamp is OFF for 10MHz-8.2GHz frequency range."
        )
    
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if hf_drop_db is None:
        hf_drop_db = rng.uniform(0.5, 2.0) * max(1.0, sig_med)
    slope = hf_drop_db / (frequency[-1] - frequency[0])
    return amp - slope * (frequency - frequency[0])

def inject_lpf_shift(frequency, amp, cutoff_shift=None, rng=None):
    """低频 LPF 拐点漂移：sigmoid 模拟滚降过渡。"""
    rng = rng or np.random.default_rng()
    bw = frequency[-1] - frequency[0]
    if cutoff_shift is None:
        cutoff_shift = rng.uniform(-0.01 * bw, 0.01 * bw)
    f = frequency
    center = f[0] + 0.1 * bw + cutoff_shift
    width = 0.02 * bw
    trans = 1 / (1 + np.exp((f - center) / width))
    return amp * (0.9 + 0.1 * trans)

def inject_mixer_ripple(frequency, amp, ripple_db=None, period=None, rng=None):
    """混频器带内纹波：正弦微纹波叠加。"""
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if ripple_db is None:
        ripple_db = rng.uniform(0.2, 0.6) * max(1.0, sig_med)
    if period is None:
        bw = frequency[-1] - frequency[0]
        period = rng.uniform(0.05 * bw, 0.2 * bw)
    ripple = ripple_db * np.sin(2 * np.pi * frequency / period)
    return amp + ripple

def inject_ytf_variation(frequency, amp, notch_depth_db=None, notch_center=None, rng=None):
    """YTF 滤波器：高频端陷波/带宽变化。"""
    rng = rng or np.random.default_rng()
    f = frequency
    if notch_depth_db is None:
        _, sig_med = _estimate_sigma(amp)
        notch_depth_db = rng.uniform(0.5, 2.0) * max(1.0, sig_med)
    if notch_center is None:
        notch_center = f[0] + 0.8 * (f[-1] - f[0])
    width = 0.01 * (f[-1] - f[0])
    notch = -notch_depth_db * np.exp(-0.5 * ((f - notch_center) / width) ** 2)
    return amp + notch

def inject_clock_drift(frequency, amp, delta_f=None, rng=None):
    """时钟系统：全局 Δf。"""
    return inject_freq_miscal(frequency, amp, delta_f=delta_f, rng=rng)

def inject_lo_path_error(frequency, amp, band_ranges, band_shifts=None, rng=None):
    """本振/路径相关：分段 Δf，分频段重采样。"""
    rng = rng or np.random.default_rng()
    out = amp.copy()
    if band_shifts is None:
        bw = frequency[-1] - frequency[0]
        band_shifts = [rng.uniform(-0.02 * bw, 0.02 * bw) for _ in band_ranges]
    for (start, end), df in zip(band_ranges, band_shifts):
        mask = (frequency >= start) & (frequency <= end)
        if np.any(mask):
            f_seg = frequency[mask]
            a_seg = out[mask]
            out[mask] = inject_freq_miscal(f_seg, a_seg, delta_f=df, rng=rng)
    return out

def inject_adc_bias(amp, gain=None, bias=None, comp=None, rng=None):
    """数字 IF/ADC 偏置或非线性。"""
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if gain is None:
        gain = 1.0 + rng.normal(0, 0.05 * max(1.0, sig_med))
    if bias is None:
        bias = rng.normal(0, 0.2 * max(0.2, sig_med))
    if comp is None:
        comp = rng.normal(0.05, 0.02)
    return inject_amplitude_miscal(amp, gain, bias, comp, rng=rng)

def inject_vbw_smoothing(amp, window=None, rng=None):
    """数字 IF/VBW：滑动平均模拟平滑。"""
    rng = rng or np.random.default_rng()
    if window is None:
        window = int(max(50, min(len(amp) // 10, rng.integers(200, 800))))
    return pd.Series(amp).rolling(window=window, min_periods=1, center=True).mean().values

def inject_power_noise(amp, noise_std=None, rng=None):
    """电源噪声：全频随机噪声提升。"""
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if noise_std is None:
        noise_std = rng.uniform(0.1, 0.3) * max(1.0, sig_med)
    return amp + rng.normal(0, noise_std, size=len(amp))