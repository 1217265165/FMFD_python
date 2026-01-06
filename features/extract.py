import numpy as np
from scipy.signal import correlate
from sklearn.linear_model import HuberRegressor

def estimate_gain_bias(rrs, amp):
    """鲁棒拟合 A ≈ g*RRS + b，返回(g, b)。"""
    X = rrs.reshape(-1, 1)
    model = HuberRegressor().fit(X, amp)
    return float(model.coef_[0]), float(model.intercept_)

def estimate_quadratic(rrs, amp):
    """拟合 A ≈ a*RRS + b + c*RRS^2，返回二次项 c（非线性压缩/扩展）。"""
    X = np.vstack([rrs, np.ones_like(rrs), rrs**2]).T
    coef, _, _, _ = np.linalg.lstsq(X, amp, rcond=None)
    return float(coef[2])

def estimate_freq_shift(frequency, rrs, amp, max_bins=500):
    """
    频率失准：频率轴整体平移的估计（可视为缩放前的近似）。
    使用归一化互相关求最佳滞后，换算为 Δf。
    """
    r1 = (rrs - np.mean(rrs)) / (np.std(rrs) + 1e-9)
    r2 = (amp - np.mean(amp)) / (np.std(amp) + 1e-9)
    corr = correlate(r2, r1, mode="full")
    lags = np.arange(-len(rrs) + 1, len(rrs))
    center = np.argmax(corr)
    start = max(0, center - max_bins)
    end = min(len(corr), center + max_bins)
    idx = start + np.argmax(corr[start:end])
    best_lag = lags[idx]
    df = np.mean(np.diff(frequency))
    return float(best_lag * df)

def envelope_violation_rate(amp, bounds):
    """包络越界率：超出上下包络的点比例。"""
    upper, lower = bounds
    viol = (amp > upper) | (amp < lower)
    return float(np.mean(viol))

def switching_step_score(frequency, amp, band_ranges):
    """切换点步进评分：各切换点幅度差绝对值的总和。"""
    score = 0.0
    for i in range(len(band_ranges) - 1):
        end_f = band_ranges[i][1]
        next_f = band_ranges[i + 1][0]
        i_end = np.argmin(np.abs(frequency - end_f))
        i_next = np.argmin(np.abs(frequency - next_f))
        step = amp[i_next] - amp[i_end]
        score += abs(float(step))
    return score

def residual_slope(frequency, rrs, amp):
    """残差随频率的线性斜率，反映高频下滑等趋势。"""
    res = amp - rrs
    A = np.vstack([frequency - frequency[0], np.ones_like(frequency)]).T
    coef, _, _, _ = np.linalg.lstsq(A, res, rcond=None)
    return float(coef[0])

def ripple_variance(rrs, amp, window=200):
    """残差局部纹波方差的均值（滑窗）。"""
    res = amp - rrs
    if window < 5:
        window = 5
    pad = window // 2
    res_p = np.pad(res, (pad, pad), mode='edge')
    vals = []
    for i in range(len(res)):
        seg = res_p[i:i + window]
        vals.append(np.var(seg))
    return float(np.mean(vals))

# ---------------- 新增：切换点异常与非切换台阶异常 ----------------
def compute_switch_step_anomalies(frequency, amp, band_ranges, expected_step=0.0, tol=0.2, win=5):
    """
    计算切换点处台阶异常：与期望步进 (expected_step) 的偏差。
    返回: err_max, err_ratio, err_count, total_switch
    """
    errs = []
    total = 0
    for i in range(len(band_ranges) - 1):
        f_sw = band_ranges[i][1]
        idx = np.argmin(np.abs(frequency - f_sw))
        l0 = max(0, idx - win)
        r0 = min(len(amp), idx + win)
        left = np.mean(amp[max(0, l0 - win):l0]) if l0 > 0 else np.mean(amp[:idx])
        right = np.mean(amp[r0:min(len(amp), r0 + win)]) if r0 < len(amp) else np.mean(amp[idx:])
        step = right - left
        delta = step - expected_step
        if np.abs(delta) > tol:
            errs.append(delta)
        total += 1
    if total == 0:
        return 0.0, 0.0, 0, 0
    err_max = float(np.max(np.abs(errs))) if errs else 0.0
    err_count = len(errs)
    err_ratio = err_count / total
    return err_max, err_ratio, err_count, total

def compute_nonswitch_steps(frequency, amp, band_ranges, tol=0.3, block=200, margin=50):
    """
    检测非切换区域的台阶（相邻块均值差超阈）。
    返回: max_step_abs, ratio, count, total_blocks
    """
    # 标记切换点附近区域
    sw_indices = [np.argmin(np.abs(frequency - br[1])) for br in band_ranges[:-1]]
    mask = np.ones(len(frequency), dtype=bool)
    for idx in sw_indices:
        l = max(0, idx - margin)
        r = min(len(frequency), idx + margin)
        mask[l:r] = False

    steps = []
    total_blocks = 0
    i = 0
    while i + 2 * block <= len(frequency):
        if not mask[i:i + 2 * block].all():
            i += block  # 跳过切换附近
            continue
        b1 = amp[i:i + block]
        b2 = amp[i + block:i + 2 * block]
        step = float(np.mean(b2) - np.mean(b1))
        steps.append(step)
        total_blocks += 1
        i += block
    if total_blocks == 0:
        return 0.0, 0.0, 0, 0
    steps_abs = np.abs(steps)
    count = int(np.sum(steps_abs > tol))
    ratio = count / total_blocks
    # FIX: numpy 数组不可直接做布尔判断
    max_step = float(np.max(steps_abs)) if steps_abs.size > 0 else 0.0
    return max_step, ratio, count, total_blocks

# ---------------- 汇总系统特征 ----------------
def extract_system_features(frequency, rrs, bounds, band_ranges, amp):
    """
    汇总系统级特征：增益/偏置、非线性、频率平移、越界率、步进评分、残差斜率、纹波，
    外加：切换点异常与非切换台阶异常。
    """
    g, b = estimate_gain_bias(rrs, amp)
    c = estimate_quadratic(rrs, amp)
    df = estimate_freq_shift(frequency, rrs, amp)
    viol = envelope_violation_rate(amp, bounds)
    step = switching_step_score(frequency, amp, band_ranges)
    slope = residual_slope(frequency, rrs, amp)
    ripple = ripple_variance(rrs, amp)

    sw_err_max, sw_err_ratio, sw_err_count, sw_total = compute_switch_step_anomalies(
        frequency, amp, band_ranges, expected_step=0.0, tol=0.2, win=5
    )
    ns_max, ns_ratio, ns_count, ns_total = compute_nonswitch_steps(
        frequency, amp, band_ranges, tol=0.3, block=200, margin=50
    )

    return {
        "gain": g,
        "bias": b,
        "comp": c,
        "df": df,
        "viol_rate": viol,
        "step_score": step,
        "res_slope": slope,
        "ripple_var": ripple,
        "switch_step_err_max": sw_err_max,
        "switch_step_err_ratio": sw_err_ratio,
        "nonswitch_step_max": ns_max,
        "nonswitch_step_ratio": ns_ratio,
    }