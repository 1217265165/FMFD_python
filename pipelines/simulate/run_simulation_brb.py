#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从基线包络生成仿真样本并串联系统/模块 BRB 推理。

对应要求.md 的 4.2：基于 `run_baseline.py` 生成的产物，自动完成
“基线 → 仿真 → 特征提取 → 系统/模块 BRB 诊断” 一键流程。

使用说明（仓库根目录执行）::

    python pipelines/simulate/run_simulation_brb.py \
        --baseline_npz Output/baseline_artifacts.npz \
        --baseline_meta Output/baseline_meta.json \
        --switch_json Output/switching_features.json \
        --out_dir Output/sim_brb \
        --n_samples 200

输出::
    - Output/sim_spectrum/raw_curves/*.csv       # 每个样本一份频率-幅度 CSV，可直接给其他方法/CLI
    - Output/sim_spectrum/raw_manifest.csv       # raw_curves 下文件列表与标签
    - Output/sim_spectrum/features_brb.csv       # 对比脚本直接读取的特征+概率
    - Output/sim_spectrum/labels.json            # 与 features_brb.csv 对应的标签
    - Output/sim_spectrum/simulated_features.csv # X1~X5 + 旧版动态阈值特征 + 标签
    - Output/sim_spectrum/system_predictions.csv # 系统级概率与正常判定
    - Output/sim_spectrum/module_predictions.csv # 21 模块概率分布
    - Output/sim_spectrum/simulated_curves.csv   # 频率-幅度矩阵（可自行提取特征）
    - Output/sim_spectrum/simulated_curves.npz   # 便于复现的仿真曲线
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from baseline.baseline import compute_rrs_bounds
from baseline.config import (
    BASELINE_ARTIFACTS,
    BASELINE_META,
    BAND_RANGES,
    OUTPUT_DIR,
    SWITCH_JSON,
)
from BRB.module_brb import MODULE_LABELS, module_level_infer
from BRB.system_brb import system_level_infer
from features.feature_extraction import (
    compute_dynamic_threshold_features,
    extract_module_features,
    extract_system_features,
)
from pipelines.simulate.faults import (
    inject_adc_bias,
    inject_amplitude_miscal,
    inject_clock_drift,
    inject_freq_miscal,
    inject_lpf_shift,
    inject_lo_path_error,
    inject_mixer_ripple,
    inject_power_noise,
    inject_preamp_degradation,
    inject_reflevel_miscal,
    inject_vbw_smoothing,
    inject_ytf_variation,
)


def _resolve(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_baseline(
    repo_root: Path, npz_path: Path, meta_path: Path, switch_path: Path
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], List[Tuple[float, float]], list]:
    npz_path = _resolve(repo_root, npz_path)
    meta_path = _resolve(repo_root, meta_path)
    switch_path = _resolve(repo_root, switch_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到基线产物 {npz_path}，请先运行 pipelines/run_baseline.py")

    data = np.load(npz_path, allow_pickle=True)
    frequency = data["frequency"]
    if "rrs" in data and "upper" in data and "lower" in data:
        rrs = data["rrs"]
        bounds = (data["upper"], data["lower"])
    else:
        traces = data["traces"]
        rrs, bounds = compute_rrs_bounds(frequency, traces)

    band_ranges = BAND_RANGES
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            band_ranges = meta.get("band_ranges", BAND_RANGES)
        except Exception:
            band_ranges = BAND_RANGES

    switch_feats = []
    if switch_path.exists():
        try:
            with open(switch_path, "r", encoding="utf-8") as f:
                switch_feats = json.load(f)
        except Exception:
            switch_feats = []

    return frequency, rrs, bounds, band_ranges, switch_feats


def _write_csv(path: Path, rows: List[Dict[str, object]], encoding: str = "utf-8") -> None:
    if not rows:
        path.write_text("", encoding=encoding)
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_curves(path: Path, frequency: np.ndarray, curves: List[np.ndarray]) -> None:
    if not curves:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = ["frequency"] + [f"sim_{idx:05d}" for idx in range(len(curves))]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, freq in enumerate(frequency):
            row = [freq]
            for curve in curves:
                row.append(curve[i] if i < len(curve) else "")
            writer.writerow(row)


def _write_raw_csvs(base_dir: Path, frequency: np.ndarray, curves: List[np.ndarray], labels: List[str], modules: List[str]) -> None:
    raw_dir = base_dir / "raw_curves"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    for idx, curve in enumerate(curves):
        sample_id = f"sim_{idx:05d}"
        csv_path = raw_dir / f"{sample_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["freq_Hz", "amplitude_dB"])
            for freq, amp in zip(frequency, curve):
                writer.writerow([freq, amp])

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "label": labels[idx] if idx < len(labels) else "",
                "module": modules[idx] if idx < len(modules) else "",
                "path": str(csv_path.relative_to(base_dir)),
            }
        )

    _write_csv(base_dir / "raw_manifest.csv", manifest_rows)


def _pick_base_trace(rrs: np.ndarray, traces: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if traces is None or traces.size == 0:
        noise = rng.normal(0, 0.05, size=len(rrs))
        return rrs + noise
    idx = rng.integers(0, traces.shape[0])
    return traces[idx]


def simulate_curve(
    frequency: np.ndarray,
    rrs: np.ndarray,
    band_ranges: List[Tuple[float, float]],
    traces: np.ndarray | None,
    rng: np.random.Generator,
    target_class: str | None = None,
) -> Tuple[np.ndarray, str, str]:
    """Generate simulated curve with optional target fault class.
    
    Args:
        frequency: Frequency array
        rrs: Reference response spectrum
        band_ranges: Frequency band ranges
        traces: Optional baseline traces
        rng: Random number generator
        target_class: If specified, force generation of this class
                     Options: 'amp_error', 'freq_error', 'ref_error', 'normal'
    
    Returns:
        (curve, label_sys, label_mod)
    """
    curve = _pick_base_trace(rrs, traces, rng).copy()
    
    # Define fault kinds grouped by system-level class
    amp_faults = ["amp", "preamp", "lpf", "mixer", "ytf", "adc", "vbw", "power"]
    freq_faults = ["freq", "clock", "lo"]
    ref_faults = ["rl", "att"]
    
    # If target class specified, select from that class only
    if target_class == "amp_error":
        fault_kind = rng.choice(amp_faults)
    elif target_class == "freq_error":
        fault_kind = rng.choice(freq_faults)
    elif target_class == "ref_error":
        fault_kind = rng.choice(ref_faults)
    elif target_class == "normal":
        fault_kind = "normal"
    else:
        # Random selection with balanced probabilities
        kind_probs = {
            "amp": 0.12,
            "freq": 0.12,
            "rl": 0.12,
            "att": 0.08,
            "preamp": 0.08,
            "lpf": 0.06,
            "mixer": 0.06,
            "ytf": 0.06,
            "clock": 0.06,
            "lo": 0.06,
            "adc": 0.06,
            "vbw": 0.06,
            "power": 0.06,
            "normal": 0.1,
        }
        kinds = list(kind_probs.keys())
        probs = np.array(list(kind_probs.values()), dtype=float)
        probs = probs / probs.sum()
        fault_kind = rng.choice(kinds, p=probs)
    
    label_sys = "normal"
    label_mod = "none"

    if fault_kind == "amp":
        curve = inject_amplitude_miscal(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "校准源"
    elif fault_kind == "freq":
        curve = inject_freq_miscal(frequency, curve, rng=rng)
        label_sys, label_mod = "频率失准", "时钟振荡器"
    elif fault_kind in ("rl", "att"):
        steps = list(rng.normal(0.6, 0.2, size=len(band_ranges) - 1))
        curve = inject_reflevel_miscal(frequency, curve, band_ranges, steps, rng=rng)
        label_sys, label_mod = "参考电平失准", "衰减器"
    elif fault_kind == "preamp":
        curve = inject_preamp_degradation(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "前置放大器"
    elif fault_kind == "lpf":
        curve = inject_lpf_shift(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "低频段前置低通滤波器"
    elif fault_kind == "mixer":
        curve = inject_mixer_ripple(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "低频段第一混频器"
    elif fault_kind == "ytf":
        curve = inject_ytf_variation(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "高频段YTF滤波器"
    elif fault_kind == "clock":
        curve = inject_clock_drift(frequency, curve, rng=rng)
        label_sys, label_mod = "频率失准", "时钟合成与同步网络"
    elif fault_kind == "lo":
        curve = inject_lo_path_error(frequency, curve, band_ranges, rng=rng)
        label_sys, label_mod = "频率失准", "本振混频组件"
    elif fault_kind == "adc":
        curve = inject_adc_bias(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "ADC"
    elif fault_kind == "vbw":
        curve = inject_vbw_smoothing(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "数字检波器"
    elif fault_kind == "power":
        curve = inject_power_noise(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "电源模块"

    return curve, label_sys, label_mod


def run_simulation(args: argparse.Namespace):
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = _resolve(repo_root, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    freq, rrs, bounds, band_ranges, switch_feats = load_baseline(
        repo_root,
        Path(args.baseline_npz),
        Path(args.baseline_meta),
        Path(args.switch_json),
    )
    traces = None
    npz_data = np.load(_resolve(repo_root, Path(args.baseline_npz)), allow_pickle=True)
    if "traces" in npz_data:
        traces = npz_data["traces"]

    rng = np.random.default_rng(args.seed)

    curves: List[np.ndarray] = []
    feature_rows: List[Dict[str, object]] = []
    system_rows: List[Dict[str, object]] = []
    module_rows: List[Dict[str, object]] = []
    brb_rows: List[Dict[str, object]] = []
    labels: dict = {}
    sys_labels: List[str] = []
    mod_labels: List[str] = []

    # Generate balanced samples across 4 system classes
    if args.balanced:
        # Ensure n_samples is divisible by 4 for perfect balance
        n_per_class = args.n_samples // 4
        remaining = args.n_samples % 4
        
        class_counts = {
            'amp_error': n_per_class + (1 if remaining > 0 else 0),
            'freq_error': n_per_class + (1 if remaining > 1 else 0),
            'ref_error': n_per_class + (1 if remaining > 2 else 0),
            'normal': n_per_class,
        }
        
        print(f"Generating balanced dataset with {args.n_samples} samples:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
        
        # Generate samples for each class
        idx = 0
        for target_class in ['amp_error', 'freq_error', 'ref_error', 'normal']:
            for _ in range(class_counts[target_class]):
                sample_id = f"sim_{idx:05d}"
                curve, label_sys, label_mod = simulate_curve(freq, rrs, band_ranges, traces, rng, target_class=target_class)
                curves.append(curve)
                sys_labels.append(label_sys)
                mod_labels.append(label_mod)

                sys_feats = extract_system_features(curve)
                dyn_feats = compute_dynamic_threshold_features(curve, rrs, bounds, switch_feats)
                sys_result = system_level_infer(sys_feats)

                module_feats = extract_module_features(curve, module_id=idx)
                module_probs = module_level_infer({**module_feats, **sys_feats, **dyn_feats}, sys_result)

                sys_probs = sys_result.get("probabilities", sys_result)
                fault_class = "normal"
                if label_sys == "幅度失准":
                    fault_class = "amp_error"
                elif label_sys == "频率失准":
                    fault_class = "freq_error"
                elif label_sys == "参考电平失准":
                    fault_class = "ref_error"

                labels[sample_id] = {
                    "type": "normal" if fault_class == "normal" else "fault",
                    "system_fault_class": fault_class if fault_class != "normal" else None,
                    "module": None if fault_class == "normal" else label_mod,
                }

                feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
                system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
                module_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **dict(zip(MODULE_LABELS, module_probs.values()))})

                brb_rows.append(
                    {
                        "sample_id": sample_id,
                        **sys_feats,
                        **dyn_feats,
                        **{f"sys_{k}": v for k, v in sys_probs.items()},
                        **{f"mod_{k}": v for k, v in module_probs.items()},
                    }
                )
                idx += 1
    else:
        # Original random generation
        for idx in range(args.n_samples):
            sample_id = f"sim_{idx:05d}"
            curve, label_sys, label_mod = simulate_curve(freq, rrs, band_ranges, traces, rng)
            curves.append(curve)
            sys_labels.append(label_sys)
            mod_labels.append(label_mod)

            sys_feats = extract_system_features(curve)
            dyn_feats = compute_dynamic_threshold_features(curve, rrs, bounds, switch_feats)
            sys_result = system_level_infer(sys_feats)

            module_feats = extract_module_features(curve, module_id=idx)
            module_probs = module_level_infer({**module_feats, **sys_feats, **dyn_feats}, sys_result)

            sys_probs = sys_result.get("probabilities", sys_result)
            fault_class = "normal"
            if label_sys == "幅度失准":
                fault_class = "amp_error"
            elif label_sys == "频率失准":
                fault_class = "freq_error"
            elif label_sys == "参考电平失准":
                fault_class = "ref_error"

            labels[sample_id] = {
                "type": "normal" if fault_class == "normal" else "fault",
                "system_fault_class": fault_class if fault_class != "normal" else None,
                "module": None if fault_class == "normal" else label_mod,
            }

            feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
            system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
            module_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **dict(zip(MODULE_LABELS, module_probs.values()))})

            brb_rows.append(
                {
                    "sample_id": sample_id,
                    **sys_feats,
                    **dyn_feats,
                    **{f"sys_{k}": v for k, v in sys_probs.items()},
                    **{f"mod_{k}": v for k, v in module_probs.items()},
                }
            )

    _write_raw_csvs(out_dir, freq, curves, sys_labels, mod_labels)
    _write_csv(out_dir / "simulated_features.csv", feature_rows)
    _write_csv(out_dir / "system_predictions.csv", system_rows)
    _write_csv(out_dir / "module_predictions.csv", module_rows)
    _write_csv(out_dir / "features_brb.csv", brb_rows, encoding="utf-8-sig")
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_curves(out_dir / "simulated_curves.csv", freq, curves)
    np.savez(out_dir / "simulated_curves.npz", frequency=freq, curves=np.array(curves))

    print(f"已保存特征/预测至 {out_dir}，comparison/评估脚本可直接使用 features_brb.csv + labels.json")


def build_argparser():
    parser = argparse.ArgumentParser(description="仿真频响并执行 BRB 诊断")
    parser.add_argument("--baseline_npz", default=BASELINE_ARTIFACTS)
    parser.add_argument("--baseline_meta", default=BASELINE_META)
    parser.add_argument("--switch_json", default=SWITCH_JSON)
    parser.add_argument("--out_dir", default=f"{OUTPUT_DIR}/sim_spectrum")
    parser.add_argument("--n_samples", type=int, default=200, 
                       help="总样本数（建议4的倍数以便完美平衡）")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--balanced", action="store_true", default=True,
                       help="生成各类故障均衡的样本（默认开启）")
    parser.add_argument("--no-balanced", dest="balanced", action="store_false",
                       help="使用原始随机概率生成样本")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    run_simulation(parser.parse_args())
