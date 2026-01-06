#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_system_level.py

用途：
- 读取 run_sinulation_brb.py 生成的仿真 BRB 结果：
    FMFD/Output/sim_spectrum/features_brb.csv
    FMFD/Output/sim_spectrum/labels.json
- 按系统级异常类型评估诊断性能：
    * 构造真实标签（正常 / 幅度失准 / 频率失准 / 参考电平失准）
    * 从 sys_幅度失准 / sys_频率失准 / sys_参考电平失准 中取 argmax 作为预测
    * 打印混淆矩阵、准确率
    * 绘制混淆矩阵图到 FMFD/Output/sim_spectrum/cm_system_level.png

运行示例（在仓库根目录）：
    python -m FMFD.pipelines.eval_system_level
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 为了中文标签显示正常
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_data(base_dir: Path):
    """
    从 base_dir 读取 features_brb.csv 和 labels.json
    """
    feats_path = base_dir / "features_brb.csv"
    labels_path = base_dir / "labels.json"

    if not feats_path.exists():
        raise FileNotFoundError(f"未找到 features_brb.csv: {feats_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"未找到 labels.json: {labels_path}")

    df = pd.read_csv(feats_path, encoding="utf-8-sig")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    return df, labels


def map_system_label(sample_id: str, labels_json: dict) -> str:
    """
    将 labels.json 中的结构映射为系统级标签：
      - normal      → "正常"
      - amp_error   → "幅度失准"
      - freq_error  → "频率失准"
      - ref_error   → "参考电平失准"

    如果找不到或结构异常，则返回 "未知"。
    """
    info = labels_json.get(sample_id, None)
    if info is None or not isinstance(info, dict):
        return "未知"

    t = info.get("type", None)
    if t == "normal":
        return "正常"

    sys_fault = info.get("system_fault_class", None)
    if sys_fault == "amp_error":
        return "幅度失准"
    if sys_fault == "freq_error":
        return "频率失准"
    if sys_fault == "ref_error":
        return "参考电平失准"

    return "未知"


def eval_system_level(base_dir=None, draw_cm: bool = True):
    """
    评估系统级诊断性能。

    Parameters
    ----------
    base_dir : str 或 Path 或 None
        - 若为 None，则默认使用包内路径：
          FMFD/Output/sim_spectrum
        - 否则使用指定目录。
    draw_cm : bool
        是否绘制并保存混淆矩阵图。
    """
    if base_dir is None:
        # 本文件位于 FMFD/pipelines/ 下，父级一层是 FMFD 包根
        pkg_root = Path(__file__).resolve().parents[1]
        base_dir = pkg_root / "Output" / "sim_spectrum"
    else:
        base_dir = Path(base_dir)

    print("[INFO] 使用结果目录:", base_dir)
    df, labels_json = load_data(base_dir)

    # 只保留 normal_* / fault_* 这些有明确标签的样本
    sample_ids = df["sample_id"].astype(str).tolist()
    true_labels = [map_system_label(sid, labels_json) for sid in sample_ids]

    df["label_sys_true"] = true_labels

    # 剔除 "未知" 样本（如果有的话）
    mask_known = df["label_sys_true"] != "未知"
    df_known = df[mask_known].copy()
    if df_known.empty:
        raise RuntimeError("没有找到可用的系统级标签（全是未知）。")

    # 系统级预测：从 sys_* 概率中取 argmax
    prob_cols = ["sys_幅度失准", "sys_频率失准", "sys_参考电平失准"]
    for c in prob_cols:
        if c not in df_known.columns:
            raise KeyError(f"在 features_brb.csv 中未找到列 {c}")

    probs = df_known[prob_cols].values
    sys_labels = ["幅度失准", "频率失准", "参考电平失准"]
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [sys_labels[i] for i in pred_idx]

    df_known["label_sys_pred"] = pred_labels

    # 标签顺序：包含正常和三种异常
    label_order = ["正常", "幅度失准", "频率失准", "参考电平失准"]

    y_true = df_known["label_sys_true"].values
    y_pred = df_known["label_sys_pred"].values

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    acc = accuracy_score(y_true, y_pred)

    print("=== 系统级诊断评估 ===")
    print("样本数:", len(df_known))
    print("总体准确率: {:.4f}".format(acc))
    print("混淆矩阵（行=真值, 列=预测）:")
    print("labels 顺序:", label_order)
    print(cm)

    if draw_cm:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title("系统级异常类型混淆矩阵")
        plt.tight_layout()
        out_path = base_dir / "cm_system_level.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("混淆矩阵图已保存:", out_path)


def main():
    eval_system_level()


if __name__ == "__main__":
    main()