import numpy as np
import pandas as pd
from .faults import (
    inject_amplitude_miscal, inject_freq_miscal, inject_reflevel_miscal,
    inject_preamp_degradation, inject_lpf_shift, inject_mixer_ripple, inject_ytf_variation,
    inject_clock_drift, inject_lo_path_error, inject_adc_bias, inject_vbw_smoothing,
    inject_power_noise
)
from FMFD.features.extract import extract_system_features

def simulate_fault_dataset(baseline, n_samples=120, seed=0):
    """
    基于基线生成带标签的仿真故障数据集，返回 DataFrame。
    baseline 需包含: frequency, rrs, bounds, band_ranges
    """
    rng = np.random.default_rng(seed)
    freq = baseline["frequency"]
    rrs = baseline["rrs"]
    band_ranges = baseline["band_ranges"]
    data = []

    kinds = ["amp", "freq", "rl", "att", "preamp", "lpf", "mixer", "ytf",
             "clock", "lo", "adc", "vbw", "power", "normal"]
    probs = [0.12, 0.12, 0.12, 0.08, 0.08, 0.06, 0.06, 0.06,
             0.06, 0.06, 0.06, 0.06, 0.06, 0.1]

    # 归一化概率，避免 numpy 报 probabilities do not sum to 1
    probs = np.array(probs, dtype=float)
    s = probs.sum()
    if s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s

    for _ in range(n_samples):
        amp0 = rrs + rng.normal(0, 0.05, size=len(rrs))
        kind = rng.choice(kinds, p=probs)
        amp = amp0.copy()
        label_sys = "normal"
        label_mod = "none"

        if kind == "amp":
            amp = inject_amplitude_miscal(amp, gain=float(rng.normal(1.05, 0.02)),
                                          bias=float(rng.normal(0.3, 0.1)),
                                          comp=float(rng.normal(0.02, 0.01)))
            label_sys = "幅度失准"; label_mod = "校准系统"
        elif kind == "freq":
            df = float(rng.normal(2e7, 1e7))
            amp = inject_freq_miscal(freq, amp, df)
            label_sys = "频率失准"; label_mod = "时钟系统"
        elif kind == "rl":
            steps = list(rng.normal(0.5, 0.2, size=len(band_ranges) - 1))
            amp = inject_reflevel_miscal(freq, amp, band_ranges, steps, compression_coef=0.2, compression_start_percent=0.8)
            label_sys = "参考电平失准"; label_mod = "衰减器"
        elif kind == "att":
            steps = list(rng.normal(0.7, 0.2, size=len(band_ranges) - 1))
            amp = inject_reflevel_miscal(freq, amp, band_ranges, steps, compression_coef=0.1, compression_start_percent=0.85)
            label_sys = "参考电平失准"; label_mod = "衰减器"
        elif kind == "preamp":
            amp = inject_preamp_degradation(freq, amp, hf_drop_db=float(rng.uniform(0.5, 2.0)))
            label_sys = "幅度失准"; label_mod = "前置放大器"
        elif kind == "lpf":
            amp = inject_lpf_shift(freq, amp, cutoff_shift=float(rng.uniform(-5e8, 5e8)))
            label_mod = "低频段LPF"
        elif kind == "mixer":
            amp = inject_mixer_ripple(freq, amp, ripple_db=float(rng.uniform(0.2, 0.6)), period=float(rng.uniform(5e8, 2e9)))
            label_mod = "低频段第一混频器"
        elif kind == "ytf":
            amp = inject_ytf_variation(freq, amp, notch_depth_db=float(rng.uniform(0.5, 2.0)))
            label_mod = "高频段YTF滤波器"
        elif kind == "clock":
            df = float(rng.normal(3e7, 1e7))
            amp = inject_clock_drift(freq, amp, df)
            label_sys = "频率失准"; label_mod = "时钟系统"
        elif kind == "lo":
            shifts = list(rng.normal(1e7, 5e6, size=len(band_ranges)))
            amp = inject_lo_path_error(freq, amp, band_ranges, shifts)
            label_sys = "频率失准"; label_mod = "本振系统"
        elif kind == "adc":
            amp = inject_adc_bias(amp, gain=float(rng.normal(0.95, 0.02)), bias=float(rng.normal(0.0, 0.1)), comp=float(rng.normal(0.05, 0.02)))
            label_mod = "数字IF板"
        elif kind == "vbw":
            amp = inject_vbw_smoothing(amp, window=int(rng.integers(200, 800)))
            label_mod = "数字IF板"
        elif kind == "power":
            amp = inject_power_noise(amp, noise_std=float(rng.uniform(0.1, 0.3)))
            label_mod = "电源模块"
        # normal: no change

        feats = extract_system_features(freq, rrs, baseline["bounds"], band_ranges, amp)
        row = {"kind": kind, "label_sys": label_sys, "label_mod": label_mod, **feats}
        data.append(row)
    return pd.DataFrame(data)