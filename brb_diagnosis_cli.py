#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRB 诊断命令行接口
用于接收频响数据并返回BRB诊断结果

该脚本可以通过命令行调用，也可以打包为exe供QT程序调用
调用方式:
  python brb_diagnosis_cli.py --input <input_csv> --output <output_json> [--baseline <baseline_dir>]
  
输入CSV格式: frequency,amplitude (两列，频率和幅度)
输出JSON格式: 包含系统级和模块级诊断结果

新增功能:
  --labels: 提供 labels.json 文件路径，可在输出中附加 ground_truth 字段
"""

import argparse
import json
import re
import sys
import os
import warnings
from pathlib import Path

# 设置环境变量抑制所有警告（在导入numpy/pandas之前）
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::DeprecationWarning'

# 抑制numpy/pandas兼容性警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"[错误] 导入依赖库失败: {e}", file=sys.stderr)
    print(f"[提示] 请安装必需的Python包: pip install numpy pandas scipy scikit-learn", file=sys.stderr)
    sys.exit(1)


def resolve_import_path():
    """解决导入路径问题 - 添加FMFD父目录到sys.path"""
    current_file = Path(__file__).resolve()
    fmfd_root = current_file.parent  # Python/FMFD目录
    python_root = fmfd_root.parent    # Python目录
    
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    
    return fmfd_root


def parse_sample_id(input_path: Path) -> str:
    """从输入文件名中解析 sample_id。
    
    解析规则：
    1. 尝试正则匹配 sim_XXXXX 格式（5位数字）
    2. 如果无法匹配，使用文件名（不含扩展名）
    
    Parameters
    ----------
    input_path : Path
        输入文件路径
        
    Returns
    -------
    str
        解析出的 sample_id
    """
    filename = input_path.stem  # 不含扩展名的文件名
    
    # 尝试匹配 sim_XXXXX 格式
    match = re.search(r'(sim_\d{5})', filename)
    if match:
        return match.group(1)
    
    # 回退到使用文件名
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='BRB诊断命令行工具 - 频响异常诊断',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python brb_diagnosis_cli.py --input test_data.csv --output result.json
  python brb_diagnosis_cli.py --input test_data.csv --output result.json --baseline ./baseline_data
  python brb_diagnosis_cli.py --input sim_00009.csv --output result.json --labels labels.json
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='输入CSV文件路径 (格式: frequency,amplitude)')
    parser.add_argument('--output', '-o', required=True,
                        help='输出JSON文件路径')
    parser.add_argument('--baseline', '-b', default=None,
                        help='基线数据目录 (可选，默认使用程序内置路径)')
    parser.add_argument('--mode', '-m', default='sub_brb',
                        choices=['er', 'simple', 'sub_brb'],
                        help='BRB推理模式: sub_brb(推荐,子BRB架构), er(增强版), simple(简化版)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    parser.add_argument('--run_name', default=None,
                        help='运行名称 (用于组织输出目录，默认使用时间戳)')
    parser.add_argument('--out_dir', default='Output',
                        help='输出目录根路径 (默认: Output)')
    parser.add_argument('--labels', '-l', default=None,
                        help='labels.json 文件路径 (可选，用于回填 ground_truth)')
    parser.add_argument('--topk', type=int, default=3,
                        help='输出 TopK 模块数量 (默认: 3)')
    parser.add_argument('--include_baseline', action='store_true',
                        help='在输出 JSON 中包含基线信息 (用于前端绘图)')
    parser.add_argument('--downsample_baseline', type=int, default=1,
                        help='基线下采样率 (默认: 1 不下采样, 可选 4 降为 205 点)')
    
    args = parser.parse_args()
    
    # 解决导入路径
    fmfd_root = resolve_import_path()
    
    try:
        # 导入FMFD模块
        from baseline.baseline import align_to_frequency
        from baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES
        from features.extract import extract_system_features
        from BRB.system_brb import system_level_infer
        from BRB.module_brb import module_level_infer, DISABLED_MODULES
        from tools.label_mapping import (
            SYS_CLASS_TO_CN, CN_TO_SYS_CLASS, 
            get_topk_modules, normalize_module_name
        )
        
        if args.verbose:
            print(f"[INFO] FMFD模块导入成功", file=sys.stderr)
            print(f"[INFO] 工作目录: {fmfd_root}", file=sys.stderr)
        
        # 1. 读取输入数据
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[错误] 输入文件不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        
        # 解析 sample_id
        sample_id = parse_sample_id(input_path)
        if args.verbose:
            print(f"[INFO] 解析 sample_id: {sample_id}", file=sys.stderr)
        
        df = pd.read_csv(input_path)
        if df.shape[1] < 2:
            print(f"[错误] 输入CSV文件至少需要2列 (frequency, amplitude)", file=sys.stderr)
            sys.exit(1)
        
        freq_raw = df.iloc[:, 0].values
        amp_raw = df.iloc[:, 1].values
        
        if args.verbose:
            print(f"[INFO] 读取数据点数: {len(freq_raw)}", file=sys.stderr)
            print(f"[INFO] 频率范围: {freq_raw.min():.2e} - {freq_raw.max():.2e} Hz", file=sys.stderr)
            print(f"[INFO] 幅度范围: {amp_raw.min():.2f} - {amp_raw.max():.2f} dB", file=sys.stderr)
        
        # 2. 加载基线数据
        if args.baseline:
            baseline_dir = Path(args.baseline)
            baseline_artifacts = baseline_dir / "baseline_artifacts.npz"
            baseline_meta = baseline_dir / "baseline_meta.json"
        else:
            # 使用默认路径
            baseline_artifacts = fmfd_root / BASELINE_ARTIFACTS
            baseline_meta = fmfd_root / BASELINE_META
        
        if not baseline_artifacts.exists():
            print(f"[错误] 基线数据文件不存在: {baseline_artifacts}", file=sys.stderr)
            sys.exit(1)
        
        art = np.load(baseline_artifacts)
        frequency = art["frequency"]
        rrs = art["rrs"]
        bounds = (art["upper"], art["lower"])
        
        with open(baseline_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        band_ranges = meta.get("band_ranges", BAND_RANGES)
        
        if args.verbose:
            print(f"[INFO] 基线频率点数: {len(frequency)}", file=sys.stderr)
            print(f"[INFO] 基线频段数: {len(band_ranges)}", file=sys.stderr)
        
        # 3. 对齐频率并提取特征
        amp = align_to_frequency(frequency, freq_raw, amp_raw)
        features = extract_system_features(frequency, rrs, bounds, band_ranges, amp)
        
        if args.verbose:
            print(f"[INFO] 提取特征数: {len(features)}", file=sys.stderr)
            print(f"[INFO] 特征: {list(features.keys())}", file=sys.stderr)
        
        # 4. 执行BRB推理
        sys_probs = system_level_infer(features, mode=args.mode)
        mod_probs = module_level_infer(features, sys_probs)
        
        if args.verbose:
            print(f"[INFO] 系统级诊断完成", file=sys.stderr)
            print(f"[INFO] 模块级诊断完成 ({len(mod_probs)}个模块)", file=sys.stderr)
        
        # 5. 构造输出结果
        # 计算证据字段（直接从 features 获取，保证一致性）
        sys_probs_dict = sys_probs.get('probabilities', sys_probs) if isinstance(sys_probs, dict) else sys_probs
        is_normal = sys_probs.get('is_normal', False) if isinstance(sys_probs, dict) else False
        max_prob = max(sys_probs_dict.values()) if sys_probs_dict else 0.0
        predicted_class = max(sys_probs_dict, key=sys_probs_dict.get) if sys_probs_dict else "未知"
        
        # 获取 TopK 模块（跳过禁用模块）
        topk_modules = get_topk_modules(mod_probs, k=args.topk, skip_disabled=True, disabled_modules=list(DISABLED_MODULES))
        topk_list = [{"module": name, "probability": float(prob)} for name, prob in topk_modules]
        
        # 证据字段使用 features 中的真实值，保证一致性
        viol_rate = float(features.get('viol_rate', 0.0))
        
        result = {
            "status": "success",
            "meta": {
                "sample_id": sample_id,
                "input_file": str(input_path.absolute()),
            },
            "data_points": len(freq_raw),
            "frequency_range": {
                "min": float(freq_raw.min()),
                "max": float(freq_raw.max())
            },
            "features": {k: float(v) if isinstance(v, (int, float)) else v for k, v in features.items()},
            "system_diagnosis": {
                "probabilities": {k: float(v) for k, v in sys_probs_dict.items()},
                "predicted_class": predicted_class,
                "max_prob": float(max_prob),
                "is_normal": is_normal,
            },
            "module_diagnosis": {
                "probabilities": {k: float(v) for k, v in mod_probs.items()},
                "topk": topk_list,
                "disabled_modules": list(DISABLED_MODULES),
            },
            "evidence": {
                "viol_rate": viol_rate,
                "envelope_violation": viol_rate > 0.1,
                "violation_max_db": float(features.get('X12', features.get('env_overrun_max', 0.0))),
                "violation_energy": float(features.get('X13', features.get('env_violation_energy', 0.0))),
                "baseline_coverage": 1.0 - viol_rate,
            },
            "config": {
                "mode": args.mode,
                "run_name": args.run_name,
                "topk": args.topk,
            }
        }
        
        # 5.5 如果需要包含基线信息（用于前端绘图）
        if args.include_baseline:
            ds = args.downsample_baseline
            if ds > 1:
                # 下采样
                freq_ds = frequency[::ds].tolist()
                rrs_ds = rrs[::ds].tolist()
                upper_ds = bounds[0][::ds].tolist()
                lower_ds = bounds[1][::ds].tolist()
            else:
                freq_ds = frequency.tolist()
                rrs_ds = rrs.tolist()
                upper_ds = bounds[0].tolist()
                lower_ds = bounds[1].tolist()
            
            # 计算基线整体电平中心（RRS 的中位数）
            center_level_db = float(np.median(rrs))
            
            # 厂商规格容差（系统级：-10 ± 0.4 dB）
            spec_center_db = -10.0
            spec_tol_db = 0.4
            spec_upper_db = spec_center_db + spec_tol_db  # -9.6
            spec_lower_db = spec_center_db - spec_tol_db  # -10.4
            
            result["baseline"] = {
                "frequency_hz": freq_ds,
                "rrs_db": rrs_ds,
                "upper_db": upper_ds,
                "lower_db": lower_ds,
                "center_level_db": center_level_db,
                "spec_center_db": spec_center_db,
                "spec_tol_db": spec_tol_db,
                "spec_upper_db": spec_upper_db,
                "spec_lower_db": spec_lower_db,
                "chosen_k": meta.get("k_final", 3.5),
                "coverage_target": meta.get("coverage_mean", 0.97),
                "n_points": len(freq_ds),
                "downsample_factor": ds,
            }
        
        # 6. 如果提供了 labels.json，加载 ground_truth
        if args.labels:
            labels_path = Path(args.labels)
            if labels_path.exists():
                try:
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        labels_data = json.load(f)
                    
                    if sample_id in labels_data:
                        gt = labels_data[sample_id]
                        gt_sys_class = gt.get("system_fault_class")
                        gt_sys_cn = SYS_CLASS_TO_CN.get(gt_sys_class, "正常") if gt_sys_class else "正常"
                        
                        result["ground_truth"] = {
                            "type": gt.get("type", "unknown"),
                            "system_class_en": gt_sys_class or "normal",
                            "system_class_cn": gt_sys_cn,
                            "module": gt.get("module"),
                            "fault_params": gt.get("fault_params", {}),
                        }
                        
                        if args.verbose:
                            print(f"[INFO] 已加载 ground_truth: {gt_sys_cn}", file=sys.stderr)
                    else:
                        if args.verbose:
                            print(f"[WARN] sample_id '{sample_id}' 不在 labels.json 中", file=sys.stderr)
                except Exception as e:
                    if args.verbose:
                        print(f"[WARN] 加载 labels.json 失败: {e}", file=sys.stderr)
        
        # 7. 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if args.verbose:
            print(f"[INFO] 结果已保存到: {output_path}", file=sys.stderr)
            print("\n" + "="*50, file=sys.stderr)
            print("[系统级诊断结果]", file=sys.stderr)
            print("="*50, file=sys.stderr)
            print(f"  样本ID: {sample_id}", file=sys.stderr)
            print(f"  预测类别: {predicted_class}", file=sys.stderr)
            print(f"  最大概率: {max_prob:.4f} ({max_prob*100:.2f}%)", file=sys.stderr)
            print(f"  是否正常: {is_normal}", file=sys.stderr)
            print("\n  概率分布:", file=sys.stderr)
            for k, v in sys_probs_dict.items():
                print(f"    {k}: {v:.4f} ({v*100:.2f}%)", file=sys.stderr)
            print("\n" + "="*50, file=sys.stderr)
            print(f"[模块级诊断 TOP{args.topk}]（跳过禁用: {DISABLED_MODULES}）", file=sys.stderr)
            print("="*50, file=sys.stderr)
            for i, item in enumerate(topk_list, 1):
                print(f"  {i}. {item['module']}: {item['probability']:.4f} ({item['probability']*100:.2f}%)", file=sys.stderr)
            print("\n" + "="*50, file=sys.stderr)
            print("[证据特征]", file=sys.stderr)
            print("="*50, file=sys.stderr)
            print(f"  viol_rate: {viol_rate:.4f}", file=sys.stderr)
            print(f"  violation_max_db: {result['evidence']['violation_max_db']:.4f}", file=sys.stderr)
            print(f"  violation_energy: {result['evidence']['violation_energy']:.4f}", file=sys.stderr)
            print(f"  baseline_coverage: {result['evidence']['baseline_coverage']:.4f}", file=sys.stderr)
            if "ground_truth" in result:
                print("\n" + "="*50, file=sys.stderr)
                print("[Ground Truth]", file=sys.stderr)
                print("="*50, file=sys.stderr)
                gt = result["ground_truth"]
                print(f"  类型: {gt['type']}", file=sys.stderr)
                print(f"  系统级: {gt['system_class_cn']}", file=sys.stderr)
                print(f"  模块: {gt['module']}", file=sys.stderr)
            print("\n" + "="*50, file=sys.stderr)
            print(f"[输出文件] {output_path}", file=sys.stderr)
            print("="*50, file=sys.stderr)
        
        print(json.dumps(result, ensure_ascii=False))
        return 0
        
    except ImportError as e:
        print(f"[错误] 模块导入失败: {e}", file=sys.stderr)
        print(f"[提示] 请确保已安装所需依赖: numpy, pandas", file=sys.stderr)
        print(f"[提示] 请确保FMFD模块在Python路径中", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False), file=sys.stdout)
        print(f"[错误] {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
