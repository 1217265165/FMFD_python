#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRB 结果评估与可视化脚本（自动使用仓库根下的默认路径）
------------------------------------------------------------
默认路径（相对仓库根 = 当前文件的上一级目录）：
- sim_csv:   Output/sim_spectrum/features_brb.csv
- detect_csv: Output/detection_results.csv
- out_dir:  Output/reports

依赖：pandas, numpy；若需画图，安装 matplotlib
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_df(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return None
    return pd.read_csv(path)


def top_module_from_labels(label_faults: Any) -> Optional[str]:
    if isinstance(label_faults, str):
        try:
            val = json.loads(label_faults)
        except Exception:
            return None
    else:
        val = label_faults
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
        return val[0].get("module", None)
    return None


def norm_sys_label(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).lower().strip()
    if s.startswith("sys_"):
        s = s[4:]
    mapping = {
        "amp": "amp_error",
        "amplitude": "amp_error",
        "amplitude_error": "amp_error",
        "amp_error": "amp_error",
        "freq": "freq_error",
        "frequency": "freq_error",
        "frequency_error": "freq_error",
        "freq_error": "freq_error",
        "ref": "ref_error",
        "reflevel": "ref_error",
        "reference": "ref_error",
        "reference_error": "ref_error",
        "ref_level": "ref_error",
        "ref_error": "ref_error",
        "幅度失准": "amp_error",
        "幅度": "amp_error",
        "频率失准": "freq_error",
        "频率": "freq_error",
        "参考电平失准": "ref_error",
        "参考电平": "ref_error",
    }
    return mapping.get(s, s)


def eval_system_level(df: pd.DataFrame) -> Dict[str, Any]:
    sys_cols = [c for c in df.columns if c.startswith("sys_")]
    if not sys_cols or "label_system_fault_class" not in df.columns:
        return {}
    df = df.copy()
    sys_pred_raw = df[sys_cols].fillna(-np.inf).idxmax(axis=1).str.replace("sys_", "")
    df["sys_pred"] = sys_pred_raw.apply(norm_sys_label)
    df["sys_true"] = df["label_system_fault_class"].apply(norm_sys_label)
    df = df.dropna(subset=["sys_true", "sys_pred"])
    if df.empty:
        return {}
    overall = (df["sys_pred"] == df["sys_true"]).mean()
    per_class = {cls: (g["sys_pred"] == g["sys_true"]).mean() for cls, g in df.groupby("sys_true", dropna=False)}
    conf = pd.crosstab(df["sys_true"], df["sys_pred"])
    return {"overall": overall, "per_class": per_class, "confusion": conf}


def eval_module_level(df: pd.DataFrame) -> Dict[str, Any]:
    mod_cols = [c for c in df.columns if c.startswith("mod_")]
    if not mod_cols or "label_faults" not in df.columns:
        return {}
    df = df.copy()
    df["mod_pred"] = df[mod_cols].idxmax(axis=1).str.replace("mod_", "")
    df["mod_true"] = df["label_faults"].apply(top_module_from_labels)
    df = df.dropna(subset=["mod_true", "mod_pred"])
    if df.empty:
        return {}
    overall = (df["mod_pred"] == df["mod_true"]).mean()
    per_mod = {cls: (g["mod_pred"] == g["mod_true"]).mean() for cls, g in df.groupby("mod_true", dropna=False)}
    return {"overall": overall, "per_module": per_mod}


def save_confusion(conf: pd.DataFrame, out_path: Path):
    conf.to_csv(out_path, encoding="utf-8-sig")
    print(f"[INFO] confusion matrix saved: {out_path}")


def plot_bars(data: Dict[str, float], title: str, out_path: Path):
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plot: {e}")
        return
    if not data:
        print(f"[WARN] empty data for plot: {title}")
        return
    keys = list(data.keys())
    vals = [data[k] for k in keys]
    plt.figure(figsize=(6, 3))
    plt.bar(keys, vals)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] plot saved: {out_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_csv", default=None, help="仿真+BRB 结果 (features_brb.csv)")
    ap.add_argument("--detect_csv", default=None, help="检测结果 (detection_results.csv)")
    ap.add_argument("--out_dir", default=None, help="报告输出目录")
    args = ap.parse_args()

    sim_csv = Path(args.sim_csv) if args.sim_csv else repo_root / "Output" / "sim_spectrum" / "features_brb.csv"
    detect_csv = Path(args.detect_csv) if args.detect_csv else repo_root / "Output" / "detection_results.csv"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "Output" / "reports"
    safe_mkdir(out_dir)

    print(f"[INFO] repo_root   = {repo_root}")
    print(f"[INFO] sim_csv     = {sim_csv}")
    print(f"[INFO] detect_csv  = {detect_csv}")
    print(f"[INFO] out_dir     = {out_dir}")

    sim_df = load_df(sim_csv)
    if sim_df is not None:
        sys_res = eval_system_level(sim_df)
        mod_res = eval_module_level(sim_df)

        if sys_res.get("confusion") is not None:
            save_confusion(sys_res["confusion"], out_dir / "confusion_system.csv")

        summary = {
            "system_overall_acc": sys_res.get("overall"),
            "system_per_class": sys_res.get("per_class"),
            "module_overall_acc": mod_res.get("overall"),
            "module_per_module": mod_res.get("per_module"),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] summary saved: {out_dir / 'summary.json'}")

        if sys_res.get("per_class"):
            plot_bars(sys_res["per_class"], "System-level accuracy by class", out_dir / "sys_per_class.png")
        if mod_res.get("per_module"):
            plot_bars(mod_res["per_module"], "Module-level accuracy (label_faults)", out_dir / "mod_per_module.png")

    detect_df = load_df(detect_csv)
    if detect_df is not None:
        out_detect = out_dir / "detect_summary.csv"
        detect_df.describe(include="all").to_csv(out_detect, encoding="utf-8-sig")
        print(f"[INFO] detect summary saved: {out_detect}")


if __name__ == "__main__":
    main()