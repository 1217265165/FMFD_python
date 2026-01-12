import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .config import FONT_FAMILY

# 设置中文字体与负号
matplotlib.rcParams["font.sans-serif"] = FONT_FAMILY
matplotlib.rcParams["axes.unicode_minus"] = False


def validate_frequency_axis(frequency: np.ndarray) -> None:
    """
    验证频率轴是否为真实频率值（非索引）。
    
    如果 freq.min() < 1e6 或 freq.max() < 1e8，说明频率列读取失败。
    """
    freq_min = frequency.min()
    freq_max = frequency.max()
    
    if freq_min < 1e6:
        raise ValueError(f"频率轴异常：freq.min()={freq_min:.2e} < 1e6 Hz，可能读取了索引而非真实频率")
    if freq_max < 1e8:
        raise ValueError(f"频率轴异常：freq.max()={freq_max:.2e} < 1e8 Hz，可能读取了索引而非真实频率")


def plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, out_path):
    """
    可视化：所有曲线、RRS、包络、切换点台阶标注。
    
    修复：x轴使用真实频率（Hz或MHz），从实际最小频率开始，不从0开始。
    """
    # 验证频率轴
    validate_frequency_axis(frequency)
    
    upper, lower = bounds
    
    # 转换为MHz以便可读性更好
    freq_mhz = frequency / 1e6
    
    plt.figure(figsize=(14, 8))
    for trace in traces:
        plt.plot(freq_mhz, trace, color="gray", alpha=0.3,
                 label="正常曲线" if "正常曲线" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(freq_mhz, rrs, color="blue", linewidth=2.0, label="RRS/Center")
    plt.fill_between(freq_mhz, lower, upper, color="blue", alpha=0.2, label="动态包络")

    for feat in switch_feats:
        end_freq = feat["end_freq"] / 1e6  # 转换为MHz
        step_mean = feat["step_mean"]
        step_std = feat["step_std"]
        ok = feat["is_within_tolerance"]
        plt.axvline(x=end_freq, color="green" if ok else "red", linestyle="--",
                    label="切换点" if "切换点" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(end_freq, np.max(upper), f"{step_mean:.2f} ± {step_std:.2f} dB",
                 color="green" if ok else "red", fontsize=10, rotation=45)
    
    # 使用真实频率范围，不从0开始
    plt.xlim(left=freq_mhz.min(), right=freq_mhz.max())
    plt.xlabel("频率 (MHz)")
    plt.ylabel("幅度 (dB)")
    plt.title("RRS 与分段动态包络及切换点特性")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"图像已保存: {out_path}")