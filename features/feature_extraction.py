#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集数据特征增强脚本（症状提取 / 聚类 / 模块元信息）
--------------------------------------------------
用途：
- 针对采集的 measurement CSV，提取症状特征（幅度/频率/相位噪声/参考电平偏差）
- 可选解析 trace 列，计算噪声底、杂散、SNR
- 计算异常分数（IsolationForest）与聚类标签（DBSCAN / KMeans）
- 按 RULES_SIMPLE 将症状激活映射到模块级 belief，输出 module_meta
- 输出：
    {prefix}_features_enhanced.csv
    {prefix}_module_meta.csv
    {prefix}_feature_summary.csv
    {prefix}_feature_importances.csv（若存在标签列则生成）
用法：
    python FMFD/features/feature_extraction.py --input acquired_measurements.csv --prefix run_enh --out_dir ./Output
依赖：pandas, numpy, scipy, scikit-learn
可选依赖：pywt, statsmodels（若需小波/时序特征，可自行扩展）
"""

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")

# ---------- CONFIG（可根据实验调整） ----------
ATTEN_GROUP_COLS = ["freq_Hz_set", "power_dBm_set"]
MIN_REPEATS_FOR_STATS = 2
DBSCAN_EPS = 0.9
DBSCAN_MIN_SAMPLES = 5
ISO_CONTAMINATION = 0.05

# ---------- BRB 映射（症状 -> belief_vector），需与 brb_rules.yaml 的 modules_order 对齐 ----------
RULES_SIMPLE = {
    "幅度测量准确度": [0.12, 0.12, 0.06, 0.06, 0.06, 0.06, 0.00, 0.00, 0.03, 0.04, 0.05, 0.02, 0.03, 0.10, 0.10, 0.03, 0.06, 0.04, 0.02, 0.00, 0.00],
    "频率读数准确度": [0.00, 0.00, 0.00, 0.05, 0.02, 0.05, 0.25, 0.25, 0.20, 0.08, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.06],
    "参考电平准确度": [0.20, 0.12, 0.03, 0.03, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.08, 0.05, 0.08, 0.10, 0.10, 0.03, 0.05, 0.05, 0.02, 0.02, 0.00],
}
MODULES_ORDER = [
    "衰减器", "前置放大器", "低频段前置低通滤波器", "低频段第一混频器",
    "高频段YTF滤波器", "高频段混频器",
    "时钟振荡器", "时钟合成与同步网络", "本振源（谐波发生器）", "本振混频组件",
    "校准源", "存储器", "校准信号开关",
    "中频放大器", "ADC", "数字RBW", "数字放大器", "数字检波器", "VBW滤波器",
    "电源模块", "未定义/其他",
]


# ---------- HELPERS ----------
def normalize_belief(vec):
    arr = np.array(vec, dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def robust_stats(arr):
    a = np.array(arr)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return dict(mean=np.nan, std=np.nan, mad=np.nan, skew=np.nan, kurt=np.nan, min=np.nan, max=np.nan, count=0)
    return dict(
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=1) if finite.size > 1 else 0.0),
        mad=float(np.median(np.abs(finite - np.median(finite)))),
        skew=float(skew(finite)) if finite.size > 2 else np.nan,
        kurt=float(kurtosis(finite)) if finite.size > 3 else np.nan,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        count=int(finite.size),
    )


def linear_trend(x, y):
    try:
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        if len(y) < 2:
            return np.nan, np.nan
        reg = LinearRegression().fit(x, y)
        return float(reg.coef_[0]), float(reg.intercept_)
    except Exception:
        return np.nan, np.nan


def parse_trace_cell(cell):
    """支持 list/ndarray 或 '[1,2,3]' 字符串"""
    try:
        if isinstance(cell, str):
            txt = cell.strip().strip("[]")
            if not txt:
                return None
            arr = np.fromstring(txt, sep=",")
            return arr if arr.size > 0 else None
        if isinstance(cell, (list, np.ndarray)):
            arr = np.array(cell, dtype=float)
            return arr if arr.size > 0 else None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 面向 BRB 推理的核心特征提取接口（对应小论文式 (1)-(2)）
# ---------------------------------------------------------------------------

def extract_system_features(response_curve) -> dict:
    """提取系统级特征 X1~X5（对应小论文式 (1)-(2)）。

    参数
    ----
    response_curve : array-like
        频响幅度序列，默认为等间隔采样的 dB 值。

    返回
    ----
    dict
        ``{"X1", "X2", "X3", "X4", "X5"}`` 五个特征，分别表示整体幅度偏移、
        带内平坦度、高频段衰减斜率、频率标度非线性度、幅度缩放一致性。
    """

    arr = np.asarray(response_curve, dtype=float)
    if arr.size == 0:
        return {"X1": 0.0, "X2": 0.0, "X3": 0.0, "X4": 0.0, "X5": 0.0}

    x1 = float(np.mean(arr))  # 整体幅度偏移（式 1 的常数项）
    inband = arr[: int(len(arr) * 0.6)] if len(arr) > 5 else arr
    x2 = float(np.var(inband))  # 带内平坦度

    # 高频段衰减斜率：末尾 20% 区间线性拟合斜率
    tail = arr[int(len(arr) * 0.8) :] if len(arr) > 5 else arr
    if tail.size >= 2:
        idx = np.arange(len(tail))
        coef = np.polyfit(idx, tail, 1)[0]
        x3 = float(coef)
    else:
        x3 = 0.0

    # 频率标度非线性度：归一化残差标准差
    x_axis = np.linspace(0, 1, len(arr))
    try:
        coef = np.polyfit(x_axis, arr, 1)
        fit = np.polyval(coef, x_axis)
        residual = arr - fit
        x4 = float(np.std(residual))
    except Exception:
        x4 = 0.0

    # 幅度缩放一致性（X5）：去除偏置后按最大幅度归一化，标准差越小表示缩放一致
    centered = arr - np.mean(arr)
    denom = np.max(np.abs(centered)) + 1e-12
    normalized = centered / denom
    x5 = float(np.std(normalized))

    return {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5}


def compute_dynamic_threshold_features(
    response_curve: Sequence[float],
    rrs: Optional[Sequence[float]] = None,
    envelope: Optional[Sequence[Sequence[float]]] = None,
    switch_features: Optional[Sequence[Dict[str, float]]] = None,
) -> Dict[str, float]:
    """保留原始动态阈值 / 频段切换跳变点症状（对应大论文基准构建部分）。

    说明
    ----
    - 论文中的系统级 X1~X5 是基础特征，但工程上原有的动态阈值症状（包络越界、
      频段切换台阶）依然有参考价值；本函数用于补充这些“遗留”特征，方便在
      pipeline 中按需启用或与 BRB 特征拼接。
    - 若未提供 `rrs` / `envelope` / `switch_features`，会返回对应字段的零值。

    参数
    ----
    response_curve : Sequence[float]
        单条频响幅度序列。
    rrs : Optional[Sequence[float]]
        基准 RRS 曲线（均值），通常来自 :func:`baseline.compute_rrs_bounds`。
    envelope : Optional[Sequence[Sequence[float]]]
        上下包络 (upper, lower)，用于计算越界比例与最大越界量。
    switch_features : Optional[Sequence[Dict[str, float]]]
        频段切换点特征列表，通常来自 :func:`baseline.detect_switch_steps`。

    返回
    ----
    dict
        兼容旧版动态阈值逻辑的特征，包括：
        ``env_overrun_rate``（包络越界比例）、``env_overrun_max``（最大越界量）、
        ``env_overrun_mean``（平均越界量）、``switch_step_mean_abs``（切换点步进
        绝对值均值）、``switch_step_std``（切换点步进标准差）。
    """

    arr = np.asarray(response_curve, dtype=float)
    if arr.size == 0:
        return {
            "env_overrun_rate": 0.0,
            "env_overrun_max": 0.0,
            "env_overrun_mean": 0.0,
            "switch_step_mean_abs": 0.0,
            "switch_step_std": 0.0,
        }

    overrun_rate = 0.0
    overrun_max = 0.0
    overrun_mean = 0.0
    if envelope is not None and len(envelope) == 2:
        upper, lower = np.asarray(envelope[0], dtype=float), np.asarray(envelope[1], dtype=float)
        if upper.shape == arr.shape and lower.shape == arr.shape:
            above = np.maximum(arr - upper, 0)
            below = np.maximum(lower - arr, 0)
            total_violation = above + below
            overrun_rate = float(np.mean(total_violation > 0))
            overrun_max = float(np.max(total_violation))
            overrun_mean = float(np.mean(total_violation))

    step_abs = []
    step_std = 0.0
    if switch_features:
        for feat in switch_features:
            step_abs.append(abs(float(feat.get("step_mean", 0.0))))
        if step_abs:
            step_std = float(np.std(step_abs))
    step_mean_abs = float(np.mean(step_abs)) if step_abs else 0.0

    return {
        "env_overrun_rate": overrun_rate,
        "env_overrun_max": overrun_max,
        "env_overrun_mean": overrun_mean,
        "switch_step_mean_abs": step_mean_abs,
        "switch_step_std": step_std,
    }


def extract_module_features(response_curve, module_id: int) -> dict:
    """提取模块层症状特征（对应小论文式 (3)）。

    参数
    ----
    response_curve : array-like
        频响幅度序列。
    module_id : int
        模块编号（0-20），用于在日志中标识来源。

    返回
    ----
    dict
        包含 ``step_score``、``res_slope``、``ripple_var``、``df``、
        ``viol_rate``、``bias`` 等字段，便于直接传入
        :func:`BRB.module_brb.module_level_infer`。
    """

    arr = np.asarray(response_curve, dtype=float)
    if arr.size == 0:
        return {
            "step_score": 0.0,
            "res_slope": 0.0,
            "ripple_var": 0.0,
            "df": 0.0,
            "viol_rate": 0.0,
            "bias": 0.0,
            "module_id": module_id,
        }

    diff = np.diff(arr)
    step_score = float(np.max(np.abs(diff)) if diff.size else 0.0)
    res_slope = float(np.mean(diff)) if diff.size else 0.0
    ripple_var = float(np.var(arr))
    df = float(np.std(np.fft.rfft(arr).real)) if arr.size > 2 else 0.0
    viol_rate = float(np.mean(np.abs(arr - np.mean(arr)) > 3 * np.std(arr)))
    bias = float(np.mean(arr))

    return {
        "step_score": step_score,
        "res_slope": res_slope,
        "ripple_var": ripple_var,
        "df": df,
        "viol_rate": viol_rate,
        "bias": bias,
        "module_id": module_id,
    }


# ---------- FEATURE ENGINEERING ----------
def extract_enhanced_features(df_raw: pd.DataFrame, prefix="run_enh",
                              iso_contamination=ISO_CONTAMINATION,
                              dbscan_eps=DBSCAN_EPS,
                              dbscan_min_samples=DBSCAN_MIN_SAMPLES):
    df = df_raw.copy()

    # 基础症状特征
    if "peak_dBm" in df.columns and "power_dBm_set" in df.columns:
        df["expected_dBm"] = df["power_dBm_set"] - df.get("atten_dB", 0)
        df["amplitude_error_dB"] = df["peak_dBm"] - df["expected_dBm"]
    else:
        df["amplitude_error_dB"] = np.nan
        df["expected_dBm"] = np.nan

    if "peak_freq_Hz" in df.columns and "freq_Hz_set" in df.columns:
        df["frequency_error_Hz"] = df["peak_freq_Hz"] - df["freq_Hz_set"]
        df["frequency_error_ppm"] = df["frequency_error_Hz"] / (df["freq_Hz_set"] + 1e-12) * 1e6
    else:
        df["frequency_error_Hz"] = np.nan
        df["frequency_error_ppm"] = np.nan

    df["phase_noise_raw"] = df["phase_noise_dBc_perHz"] if "phase_noise_dBc_perHz" in df.columns else np.nan
    df["ref_level_raw"] = df["ref_level_dBm"] if "ref_level_dBm" in df.columns else np.nan

    # 分组统计
    group_cols = [c for c in ATTEN_GROUP_COLS if c in df.columns]
    if len(group_cols) == 0:
        group_cols = ["seq_index"] if "seq_index" in df.columns else []
    if group_cols:
        grouped = df.groupby(group_cols)
        agg = grouped.agg({
            "amplitude_error_dB": ["mean", "std"],
            "frequency_error_Hz": ["mean", "std"],
            "phase_noise_raw": ["mean", "std"],
            "ref_level_raw": ["mean", "std"],
        })
        agg.columns = ["_".join(x).strip() for x in agg.columns.values]
        agg = agg.reset_index()
        df = df.merge(agg, on=group_cols, how="left")
    else:
        df["amplitude_error_dB_mean"] = df["amplitude_error_dB"].mean()
        df["amplitude_error_dB_std"] = df["amplitude_error_dB"].std()
        df["frequency_error_Hz_mean"] = df["frequency_error_Hz"].mean()
        df["frequency_error_Hz_std"] = df["frequency_error_Hz"].std()
        df["phase_noise_raw_mean"] = df["phase_noise_raw"].mean()
        df["phase_noise_raw_std"] = df["phase_noise_raw"].std()
        df["ref_level_raw_mean"] = df["ref_level_raw"].mean()
        df["ref_level_raw_std"] = df["ref_level_raw"].std()

    # 归一化/衍生
    for col in ["amplitude_error_dB", "frequency_error_Hz", "phase_noise_raw", "ref_level_raw"]:
        global_mean = df[col].mean()
        global_std = df[col].std() if not math.isnan(df[col].std()) else 1.0
        df[f"{col}_z"] = (df[col] - global_mean) / (global_std + 1e-12)
        df[f"{col}_abs"] = np.abs(df[col])
        df[f"{col}_rel_pct"] = (df[col] - global_mean) / (global_mean + 1e-12) * 100.0

    # 衰减-幅度斜率（若存在衰减设置）
    if "atten_dB" in df.columns and group_cols:
        slopes = []
        keys = []
        for name, g in df.groupby(group_cols):
            if g["atten_dB"].nunique() >= 2:
                slope, intercept = linear_trend(g["atten_dB"].values, g["amplitude_error_dB"].fillna(0.0).values)
            else:
                slope, intercept = np.nan, np.nan
            keys.append(name)
            slopes.append((slope, intercept))
        slope_map = {k: v[0] for k, v in zip(keys, slopes)}

        def get_slope(row):
            key = tuple(row[c] for c in group_cols) if len(group_cols) > 1 else row[group_cols[0]]
            return slope_map.get(key, np.nan)

        df["atten_amp_slope"] = df.apply(get_slope, axis=1)
    else:
        df["atten_amp_slope"] = np.nan

    # trace-based 特征（可选）
    trace_cols = [c for c in df.columns if "trace" in c.lower()]
    if trace_cols:
        for tc in trace_cols:
            spurs, noise_floors, max_spurs, snrs = [], [], [], []
            for val in df[tc].values:
                arr = parse_trace_cell(val)
                if arr is None or arr.size == 0:
                    spurs.append(0)
                    noise_floors.append(np.nan)
                    max_spurs.append(np.nan)
                    snrs.append(np.nan)
                    continue
                nf = np.percentile(arr, 20)
                noise_floors.append(float(nf))
                spur_thresh = nf + 6.0
                peaks = arr[arr > spur_thresh]
                spurs.append(int(peaks.size))
                max_spurs.append(float(np.max(peaks) if peaks.size > 0 else np.nan))
                primary = np.max(arr)
                snrs.append(float(primary - nf))
            df[f"{tc}_spur_count"] = spurs
            df[f"{tc}_noise_floor"] = noise_floors
            df[f"{tc}_max_spur"] = max_spurs
            df[f"{tc}_snr"] = snrs

    # 异常检测（IsolationForest）
    indicators = ["amplitude_error_dB", "frequency_error_Hz", "phase_noise_raw", "ref_level_raw"]
    avail_inds = [c for c in indicators if c in df.columns]
    if len(avail_inds) >= 1:
        X = df[avail_inds].fillna(0.0).values
        try:
            iso = IsolationForest(n_estimators=200, contamination=iso_contamination, random_state=42)
            iso.fit(X)
            scores = iso.decision_function(X)
            smin, smax = scores.min(), scores.max()
            norm = (scores - smin) / (smax - smin + 1e-12)
            df["if_anom_score"] = 1.0 - norm
            df["if_is_outlier"] = (df["if_anom_score"] > 0.9).astype(int)
        except Exception:
            df["if_anom_score"] = np.nan
            df["if_is_outlier"] = 0
    else:
        df["if_anom_score"] = np.nan
        df["if_is_outlier"] = 0

    # 聚类（DBSCAN + KMeans，可选）
    cluster_feats = []
    for f in ["amplitude_error_dB_z", "frequency_error_Hz_z", "phase_noise_raw_z", "ref_level_raw_z", "if_anom_score"]:
        if f in df.columns:
            cluster_feats.append(f)
    if cluster_feats and len(df) > 5:
        Xc = df[cluster_feats].fillna(0.0).values
        try:
            scaler = StandardScaler()
            Xc_s = scaler.fit_transform(Xc)
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(Xc_s)
            df["cluster_dbscan"] = db.labels_
            k = min(7, max(2, int(math.sqrt(len(df)))))
            km = KMeans(n_clusters=k, random_state=42).fit_predict(Xc_s)
            df["cluster_kmeans"] = km
        except Exception:
            df["cluster_dbscan"] = -1
            df["cluster_kmeans"] = -1
    else:
        df["cluster_dbscan"] = -1
        df["cluster_kmeans"] = -1

    # 症状 activation（0..1），映射到模块 belief
    activations = {}
    amp_ref = 0.5
    activations["幅度测量准确度"] = np.clip(np.abs(df["amplitude_error_dB"].fillna(0.0)) / amp_ref, 0.0, 1.0)
    freq_ref = 5.0
    activations["频率读数准确度"] = np.clip(np.abs(df["frequency_error_Hz"].fillna(0.0)) / freq_ref, 0.0, 1.0)
    if "phase_noise_raw" in df.columns:
        med = np.nanmedian(df["phase_noise_raw"].dropna()) if df["phase_noise_raw"].dropna().size > 0 else 0.0
        deg = df["phase_noise_raw"].fillna(med) - med
        phase_ref = 3.0
        activations["相位噪声"] = np.clip(deg / phase_ref, 0.0, 1.0)
    else:
        activations["相位噪声"] = np.zeros(len(df))
    ref_ref = 1.0
    activations["参考电平准确度"] = np.clip(np.abs(df["ref_level_raw"].fillna(0.0)) / ref_ref, 0.0, 1.0)

    for sym, acts in activations.items():
        df[f"{sym}__activation"] = acts

    bv_list = [normalize_belief(RULES_SIMPLE[sym]) for sym in RULES_SIMPLE.keys()]
    sym_order = list(RULES_SIMPLE.keys())
    module_meta = np.zeros((len(df), len(bv_list[0])), dtype=float)
    for i, sym in enumerate(sym_order):
        acts = df[f"{sym}__activation"].fillna(0.0).values
        bv = bv_list[i]
        module_meta += np.outer(acts, bv)
    row_sums = module_meta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    module_meta = module_meta / row_sums
    module_cols = [f"module_{m}" for m in MODULES_ORDER]
    module_meta_df = pd.DataFrame(module_meta, columns=module_cols)

    # 特征汇总
    feat_summary = []
    for c in df.columns:
        s = df[c]
        feat_summary.append({
            "features": c,
            "n_nonnull": int(s.notnull().sum()),
            "pct_nonnull": float(s.notnull().mean()),
            "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) else np.nan,
        })
    feat_summary_df = pd.DataFrame(feat_summary)

    # 有监督特征重要性（如果存在标签列）
    feature_importances = None
    label_col = None
    for possible in ["true_module", "真实模块", "label", "ground_truth"]:
        if possible in df.columns:
            label_col = possible
            break
    if label_col is not None:
        try:
            Xcols = [
                c for c in df.columns
                if any(k in c for k in [
                    "amplitude_error", "frequency_error", "phase_noise", "ref_level",
                    "__activation", "_anom", "_z", "_snr", "_noise_floor", "atten_amp_slope"
                ])
            ]
            Xcols = [c for c in Xcols if pd.api.types.is_numeric_dtype(df[c])]
            X = df[Xcols].fillna(0.0).values
            y = df[label_col].values
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = list(pd.factorize(y)[1])
                mapping = {v: i for i, v in enumerate(uniq)}
                y_idx = np.array([mapping.get(v, -1) for v in y])
                valid_mask = (y_idx >= 0)
                y_train = y_idx[valid_mask]
                X_train = X[valid_mask]
            else:
                y_train = y
                X_train = X
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            imps = rf.feature_importances_
            feature_importances = pd.DataFrame({"features": Xcols, "importance": imps}).sort_values("importance", ascending=False)
        except Exception:
            feature_importances = None

    # 保存输出
    prefix = prefix if prefix else "run_enh"
    feat_out_path = f"{prefix}_features_enhanced.csv"
    module_meta_path = f"{prefix}_module_meta.csv"
    feat_summary_path = f"{prefix}_feature_summary.csv"
    df_out = pd.concat([df.reset_index(drop=True), module_meta_df.reset_index(drop=True)], axis=1)
    df_out.to_csv(feat_out_path, index=False, encoding="utf-8-sig")
    module_meta_df.to_csv(module_meta_path, index=False, encoding="utf-8-sig")
    feat_summary_df.to_csv(feat_summary_path, index=False, encoding="utf-8-sig")
    if feature_importances is not None:
        feature_importances.to_csv(f"{prefix}_feature_importances.csv", index=False, encoding="utf-8-sig")
    return df_out, module_meta_df, feat_summary_df, feature_importances


# 兼容接口，供 main_pipeline 或外部调用
def compute_feature_matrix(raw_input, prefix="run_enh", **kwargs):
    if isinstance(raw_input, str):
        raw_df = pd.read_csv(raw_input, encoding="utf-8")
    elif isinstance(raw_input, pd.DataFrame):
        raw_df = raw_input.copy()
    else:
        raise ValueError("raw_input must be a pandas.DataFrame or path to CSV")
    return extract_enhanced_features(raw_df, prefix=prefix, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="采集的 measurement CSV 路径")
    parser.add_argument("--prefix", default="run_enh", help="输出文件名前缀")
    parser.add_argument("--out_dir", default=".", help="输出目录")
    parser.add_argument("--iso_contamination", type=float, default=ISO_CONTAMINATION)
    parser.add_argument("--dbscan_eps", type=float, default=DBSCAN_EPS)
    parser.add_argument("--dbscan_min_samples", type=int, default=DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / args.prefix

    df_raw = pd.read_csv(args.input, encoding="utf-8")
    df_out, module_meta_df, feat_summary_df, feature_importances = extract_enhanced_features(
        df_raw,
        prefix=str(prefix_path),
        iso_contamination=args.iso_contamination,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    print(f"[INFO] Enhanced features: {prefix_path}_features_enhanced.csv")
    print(f"[INFO] Module meta: {prefix_path}_module_meta.csv")
    print(f"[INFO] Feature summary: {prefix_path}_feature_summary.csv")
    if feature_importances is not None:
        print(f"[INFO] Feature importances: {prefix_path}_feature_importances.csv")


if __name__ == "__main__":
    main()