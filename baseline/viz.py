import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .config import FONT_FAMILY

# 设置中文字体与负号
matplotlib.rcParams["font.sans-serif"] = FONT_FAMILY
matplotlib.rcParams["axes.unicode_minus"] = False

def plot_rrs_envelope_switch(frequency, traces, rrs, bounds, switch_feats, out_path):
    """
    可视化：所有曲线、RRS、包络、切换点台阶标注。
    """
    upper, lower = bounds
    plt.figure(figsize=(14, 8))
    for trace in traces:
        plt.plot(frequency, trace, color="gray", alpha=0.3,
                 label="正常曲线" if "正常曲线" not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(frequency, rrs, color="blue", linewidth=2.0, label="RRS")
    plt.fill_between(frequency, lower, upper, color="blue", alpha=0.2, label="动态包络")

    for feat in switch_feats:
        end_freq = feat["end_freq"]
        step_mean = feat["step_mean"]
        step_std = feat["step_std"]
        ok = feat["is_within_tolerance"]
        plt.axvline(x=end_freq, color="green" if ok else "red", linestyle="--",
                    label="切换点" if "切换点" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(end_freq, np.max(upper), f"{step_mean:.2f} ± {step_std:.2f} dB",
                 color="green" if ok else "red", fontsize=10, rotation=45)
    plt.xlim(left=0)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度 (dB)")
    plt.title("RRS 与分段动态包络及切换点特性")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"图像已保存: {out_path}")