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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.baseline import compute_rrs_bounds
from baseline.config import (
    BASELINE_ARTIFACTS,
    BASELINE_META,
    BAND_RANGES,
    OUTPUT_DIR,
    SWITCH_JSON,
)
from BRB.module_brb import DISABLED_MODULES, MODULE_LABELS, module_level_infer
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
    inject_reflevel_miscal,
    inject_vbw_smoothing,
    inject_ytf_variation,
    SINGLE_BAND_MODE,
)
from pipelines.default_paths import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    BASELINE_NPZ,
    BASELINE_META,
    SIM_DIR,
    SEED,
    SINGLE_BAND,
    DISABLE_PREAMP,
    DEFAULT_N_SAMPLES,
    DEFAULT_BALANCED,
    build_run_snapshot,
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

    # Collect all unique fieldnames from all rows
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    fieldnames = sorted(all_fieldnames)
    
    with path.open("w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
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
            writer.writerow(["freq_hz", "spec_reading_dbm"])
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


KIND_TO_MODULE = {
    "amp": "校准源",
    "freq": "时钟振荡器",
    "rl": "衰减器",
    "att": "衰减器",
    "lpf": "低频段前置低通滤波器",
    "mixer": "低频段第一混频器",
    "ytf": "高频段YTF滤波器",
    "clock": "时钟合成与同步网络",
    "lo": "本振混频组件",
    "adc": "ADC",
    "vbw": "数字检波器",
    "power": "电源模块",
}


def _filter_kind_probs(kind_probs: Dict[str, float]) -> Dict[str, float]:
    if not DISABLED_MODULES:
        return kind_probs
    filtered = {}
    for kind, prob in kind_probs.items():
        module = KIND_TO_MODULE.get(kind)
        if module and module in DISABLED_MODULES:
            continue
        filtered[kind] = prob
    return filtered or kind_probs


def _choose_ref_module(rng: np.random.Generator) -> str:
    ref_modules = ["衰减器", "校准源", "存储器", "校准信号开关"]
    enabled = [module for module in ref_modules if module not in DISABLED_MODULES]
    if not enabled:
        return "校准源"
    return rng.choice(enabled)


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
    
    # More realistic probability distribution based on actual fault complexity
    # Amplitude faults have more module types (7) so naturally more common
    # Frequency faults have fewer modules (3) so less common
    # Reference level faults have specific modules (2)
    # NOTE: Preamp is DISABLED in single-band mode (10MHz-8.2GHz)
    if target_class is None:
        # Realistic distribution reflecting module diversity (NO PREAMP)
        kind_probs = {
            # Amplitude-related (多种模块, 概率较高) - NO PREAMP
            "amp": 0.12,      # Calibration source
            "lpf": 0.09,      # Low-pass filter
            "mixer": 0.09,    # Mixer
            "ytf": 0.09,      # YTF filter
            "adc": 0.09,      # ADC
            "vbw": 0.08,      # Digital detector
            "power": 0.08,    # Power supply
            # Frequency-related (少量模块, 概率较低)
            "freq": 0.08,     # Frequency calibration
            "clock": 0.06,    # Clock synthesis
            "lo": 0.06,       # Local oscillator
            # Reference level (特定模块)
            "rl": 0.08,       # Reference level
            "att": 0.06,      # Attenuator
            # Normal
            "normal": 0.10,   # Normal state (increased slightly)
        }
    elif target_class == "amp_error":
        # Select from amplitude fault modules with realistic weights (NO PREAMP)
        kind_probs = {
            "amp": 0.18, "lpf": 0.15, "mixer": 0.15,
            "ytf": 0.15, "adc": 0.15, "vbw": 0.11, "power": 0.11
        }
    elif target_class == "freq_error":
        # Select from frequency fault modules
        kind_probs = {"freq": 0.40, "clock": 0.30, "lo": 0.30}
    elif target_class == "ref_error":
        # Select from reference level modules
        kind_probs = {"rl": 0.60, "att": 0.40}
    elif target_class == "normal":
        kind_probs = {"normal": 1.0}
    else:
        # Fallback to distribution without preamp
        kind_probs = {
            "amp": 0.14, "freq": 0.12, "rl": 0.12, "att": 0.08,
            "lpf": 0.08, "mixer": 0.08, "ytf": 0.08,
            "clock": 0.06, "lo": 0.06, "adc": 0.06, "vbw": 0.06,
            "power": 0.06, "normal": 0.10,
        }
    
    kind_probs = _filter_kind_probs(kind_probs)
    kinds = list(kind_probs.keys())
    probs = np.array(list(kind_probs.values()), dtype=float)
    probs = probs / probs.sum()
    fault_kind = rng.choice(kinds, p=probs)
    
    label_sys = "normal"
    label_mod = "none"
    fault_params = {}  # Track injection parameters

    if fault_kind == "amp":
        curve = inject_amplitude_miscal(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "校准源"
        fault_params['type'] = 'amp_miscal'
    elif fault_kind == "freq":
        curve, freq_params = inject_freq_miscal(frequency, curve, rng=rng, return_params=True)
        label_sys, label_mod = "频率失准", "时钟振荡器"
        fault_params.update(freq_params)
        fault_params['type'] = 'freq_miscal'
    elif fault_kind in ("rl", "att"):
        # Use single_band_mode=True for reflevel injection (no step injection)
        curve, ref_params = inject_reflevel_miscal(frequency, curve, band_ranges, rng=rng, 
                                                    single_band_mode=True, return_params=True)
        label_sys, label_mod = "参考电平失准", _choose_ref_module(rng)
        fault_params.update(ref_params)
        fault_params['type'] = 'ref_miscal'
    # NOTE: preamp case is REMOVED - it's disabled in single-band mode
    elif fault_kind == "lpf":
        curve = inject_lpf_shift(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "低频段前置低通滤波器"
        fault_params['type'] = 'lpf_shift'
    elif fault_kind == "mixer":
        curve = inject_mixer_ripple(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "低频段第一混频器"
        fault_params['type'] = 'mixer_ripple'
    elif fault_kind == "ytf":
        curve = inject_ytf_variation(frequency, curve, rng=rng)
        label_sys, label_mod = "幅度失准", "高频段YTF滤波器"
        fault_params['type'] = 'ytf_variation'
    elif fault_kind == "clock":
        # Clock drift also uses freq_miscal internally
        curve, freq_params = inject_freq_miscal(frequency, curve, rng=rng, return_params=True)
        label_sys, label_mod = "频率失准", "时钟合成与同步网络"
        fault_params.update(freq_params)
        fault_params['type'] = 'clock_drift'
    elif fault_kind == "lo":
        curve = inject_lo_path_error(frequency, curve, band_ranges, rng=rng)
        label_sys, label_mod = "频率失准", "本振混频组件"
        fault_params['type'] = 'lo_path_error'
    elif fault_kind == "adc":
        curve = inject_adc_bias(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "ADC"
        fault_params['type'] = 'adc_bias'
    elif fault_kind == "vbw":
        curve = inject_vbw_smoothing(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "数字检波器"
        fault_params['type'] = 'vbw_smoothing'
    elif fault_kind == "power":
        curve = inject_power_noise(curve, rng=rng)
        label_sys, label_mod = "幅度失准", "电源模块"
        fault_params['type'] = 'power_noise'
    else:
        fault_params['type'] = 'normal'

    return curve, label_sys, label_mod, fault_params


def run_simulation(args: argparse.Namespace):
    repo_root = PROJECT_ROOT
    out_dir = _resolve(repo_root, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    build_run_snapshot(out_dir)

    print(f"[INFO] project_root={repo_root}")
    print(f"[INFO] single_band={SINGLE_BAND}")
    print(f"[INFO] disable_preamp={DISABLE_PREAMP}")
    print(f"[INFO] seed={args.seed}")
    print(f"[INFO] output_dir={out_dir}")

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
    fault_params_list: List[Dict] = []  # Track fault injection parameters

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
                curve, label_sys, label_mod, fault_params = simulate_curve(freq, rrs, band_ranges, traces, rng, target_class=target_class)
                curves.append(curve)
                sys_labels.append(label_sys)
                mod_labels.append(label_mod)
                fault_params_list.append({'sample_id': sample_id, **fault_params})

                # Pass baseline_curve=rrs and envelope=bounds to extract X16-X18 features properly
                sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
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
                    "system_fault_class": fault_class,  # Always include, "normal" for normal samples
                    "module": None if fault_class == "normal" else label_mod,
                    "fault_params": fault_params,  # Include injection parameters
                }

                feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
                system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
                module_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **dict(zip(MODULE_LABELS, module_probs.values()))})

                brb_rows.append(
                    {
                        "sample_id": sample_id,
                        **sys_feats,
                        **dyn_feats,
                        **{f"mod_{k}": v for k, v in module_probs.items()},
                    }
                )
                idx += 1
    else:
        # Realistic distribution generation (default)
        print(f"Generating realistic distribution with {args.n_samples} samples")
        print("Expected distribution (based on module diversity):")
        print("  Amplitude faults (8 modules): ~58%")
        print("  Frequency faults (3 modules): ~20%")
        print("  Reference faults (2 modules): ~14%")
        print("  Normal state: ~8%")
        print()
        
        for idx in range(args.n_samples):
            sample_id = f"sim_{idx:05d}"
            curve, label_sys, label_mod, fault_params = simulate_curve(freq, rrs, band_ranges, traces, rng)
            curves.append(curve)
            sys_labels.append(label_sys)
            mod_labels.append(label_mod)
            fault_params_list.append({'sample_id': sample_id, **fault_params})

            # Pass baseline_curve=rrs and envelope=bounds to extract X16-X18 features properly
            sys_feats = extract_system_features(curve, baseline_curve=rrs, envelope=bounds)
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
                "system_fault_class": fault_class,  # Always include, "normal" for normal samples
                "module": None if fault_class == "normal" else label_mod,
                "fault_params": fault_params,
            }

            feature_rows.append({"sample_id": sample_id, "fault_kind": label_sys, "module_label": label_mod, **sys_feats, **dyn_feats})
            system_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **sys_probs})
            module_rows.append({"sample_id": sample_id, "fault_kind": label_sys, **dict(zip(MODULE_LABELS, module_probs.values()))})

            brb_rows.append(
                {
                    "sample_id": sample_id,
                    **sys_feats,
                    **dyn_feats,
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

    # Validate output counts
    raw_count = len(list((out_dir / "raw_curves").glob("*.csv")))
    label_count = len(labels)
    features_path = out_dir / "features_brb.csv"
    features_count = 0
    if features_path.exists():
        with features_path.open("r", encoding="utf-8-sig") as f:
            features_count = max(0, sum(1 for _ in f) - 1)

    expected = args.n_samples
    if raw_count != expected or label_count != expected or features_count != expected:
        print(
            "[ERROR] Output counts mismatch: "
            f"raw_curves={raw_count}, labels={label_count}, features={features_count}, expected={expected}"
        )
        raise SystemExit(1)
    
    # Save fault params CSV for effect check
    if fault_params_list:
        _write_csv(out_dir / "fault_params.csv", fault_params_list, encoding="utf-8-sig")
        print(f"Saved fault_params.csv with injection parameters")
    
    # Generate freq_ref_effect_check.csv
    _generate_effect_check(out_dir, feature_rows, labels)

    # Print summary statistics
    print()
    print("=" * 60)
    print("仿真完成摘要")
    print("=" * 60)
    print(f"  raw_curves 路径: {out_dir / 'raw_curves'}")
    print(f"  labels.json 路径: {out_dir / 'labels.json'}")
    print(f"  生成样本数: {len(curves)}")
    print()
    print("  系统级分布:")
    sys_class_dist = {}
    for sample_id, lbl in labels.items():
        cls = lbl.get('system_fault_class', 'normal')
        sys_class_dist[cls] = sys_class_dist.get(cls, 0) + 1
    for cls in ['normal', 'amp_error', 'freq_error', 'ref_error']:
        count = sys_class_dist.get(cls, 0)
        pct = count / len(labels) * 100 if labels else 0
        print(f"    {cls}: {count} ({pct:.1f}%)")
    print("=" * 60)


def _generate_effect_check(out_dir: Path, feature_rows: List[Dict], labels: dict):
    """Generate freq_ref_effect_check.csv to verify injection → feature correlation."""
    freq_features = ['X16', 'X17', 'X18', 'X23', 'X24', 'X25']
    ref_features = ['X3', 'X5', 'X26', 'X27', 'X28']

    stats = []
    for cls in ['normal', 'amp_error', 'freq_error', 'ref_error']:
        cls_rows = [
            row for row in feature_rows
            if labels.get(row.get('sample_id', ''), {}).get('system_fault_class', 'normal') == cls
        ]
        if not cls_rows:
            continue

        row_stats = {'class': cls, 'n': len(cls_rows)}
        for f in freq_features + ref_features:
            vals = [float(r.get(f, 0.0)) for r in cls_rows if f in r]
            if vals:
                arr = np.array(vals, dtype=float)
                row_stats[f'{f}_mean'] = float(np.mean(arr))
                row_stats[f'{f}_std'] = float(np.std(arr))
                row_stats[f'{f}_p90'] = float(np.percentile(arr, 90))
        stats.append(row_stats)

    if stats:
        output_path = out_dir / 'freq_ref_effect_check.csv'
        keys = sorted({k for row in stats for k in row.keys()})
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats)
        print("Saved freq_ref_effect_check.csv")
        
        stats_by_class = {row.get("class"): row for row in stats}
        print("\n=== Freq/Ref Feature Effect Check ===")
        print("Freq features (should be high for freq_error):")
        for f in ['X16', 'X23', 'X24']:
            for cls, row in stats_by_class.items():
                key = f"{f}_mean"
                if key in row:
                    print(f"  {cls:12s} {f}_mean={row.get(key, 0):.4f}")

        print("\nRef features (should be high for ref_error):")
        for f in ['X26', 'X27', 'X28']:
            for cls, row in stats_by_class.items():
                key = f"{f}_mean"
                if key in row:
                    print(f"  {cls:12s} {f}_mean={row.get(key, 0):.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="仿真频响并执行 BRB 诊断")
    parser.add_argument("--baseline_npz", default=BASELINE_NPZ)
    parser.add_argument("--baseline_meta", default=BASELINE_META)
    parser.add_argument("--switch_json", default=SWITCH_JSON)
    parser.add_argument("--out_dir", default=SIM_DIR)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="总样本数（默认400，建议4的倍数以便完美平衡）",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=DEFAULT_BALANCED,
        help="生成系统级均衡的样本（每类相同数量，默认开启）",
    )
    parser.add_argument("--realistic", dest="balanced", action="store_false",
                       help="使用真实概率分布（反映模块多样性：幅度58%%,频率20%%,参考14%%,正常8%%）")
    return parser


if __name__ == "__main__":
    import sys
    import os
    
    # Change to repository root for relative paths to work
    # This enables Windows double-click execution
    script_dir = Path(__file__).resolve().parent
    repo_root = PROJECT_ROOT
    os.chdir(repo_root)
    
    # Build parser and run
    parser = build_argparser()
    args = parser.parse_args()
    
    # Show banner for interactive use (Windows double-click)
    print("=" * 60)
    print("FMFD Simulation Pipeline (系统级均衡仿真)")
    print("=" * 60)
    print(f"  Samples: {args.n_samples} (balanced={args.balanced})")
    print(f"  Output:  {args.out_dir}")
    print("=" * 60)
    print()
    
    # Run simulation
    run_simulation(args)
    
    # Windows: pause if double-clicked (no parent console)
    if sys.platform == 'win32':
        try:
            # Check if we're in an interactive session
            if sys.stdin.isatty():
                print()
                print("=" * 60)
                print("Simulation complete! Press Enter to exit...")
                input()
        except Exception:
            pass
