import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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
def inject_amplitude_miscal(amp, gain=None, bias=None, comp=None, rng=None):
    """
    幅度失准：A' = gain*A + bias + comp*A^2
    若 gain/bias/comp 未给出，则基于当前曲线的 σ 自适应随机生成。
    """
    rng = rng or np.random.default_rng()
    _, sig_med = _estimate_sigma(amp)
    if gain is None:
        gain = 1.0 + rng.normal(0, 0.05 * max(1.0, sig_med))
    if bias is None:
        bias = rng.normal(0, 0.5 * max(0.2, sig_med))
    if comp is None:
        comp = rng.normal(0.02, 0.01)
    return gain * amp + bias + comp * (amp ** 2)

def inject_freq_miscal(frequency, amp, delta_f=None, rng=None):
    """
    频率失准：频率轴整体平移后重采样；delta_f 未给出时按带宽 ppm 生成。
    """
    rng = rng or np.random.default_rng()
    bw = frequency[-1] - frequency[0]
    if delta_f is None:
        ppm = rng.uniform(-150e-6, 150e-6)
        delta_f = ppm * bw
    f_shift = frequency + delta_f
    interp = interp1d(f_shift, amp, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp(frequency)

def inject_reflevel_miscal(frequency, amp, band_ranges, step_biases=None, compression_coef=None,
                           compression_start_percent=0.8, rng=None):
    """
    参考电平失准：在切换点施加错误步进；高幅度区压缩。
    step_biases/compression_coef 未给出时按 σ 自适应随机生成。
    """
    rng = rng or np.random.default_rng()
    out = amp.copy()
    _, sig_med = _estimate_sigma(amp)
    if step_biases is None:
        step_biases = [rng.normal(0.6 * sig_med, 0.2 * max(0.2, sig_med))
                       for _ in range(len(band_ranges) - 1)]
    for i in range(len(band_ranges) - 1):
        end_f = band_ranges[i][1]
        m_end = np.argmin(np.abs(frequency - end_f))
        out[m_end:] += step_biases[i]
    if compression_coef is None:
        compression_coef = abs(rng.normal(0.15, 0.05))
    thr = np.percentile(out, 100 * compression_start_percent)
    mask = out >= thr
    out[mask] = out[mask] - compression_coef * (out[mask] - thr)
    return out

# -------------------------
# 模块级示例畸变（自适应幅度）
# -------------------------
def inject_preamp_degradation(frequency, amp, hf_drop_db=None, rng=None):
    """前置放大器衰减：随频率线性下滑，高频端下降 hf_drop_db。"""
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