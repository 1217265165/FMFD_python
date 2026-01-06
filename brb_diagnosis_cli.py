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
"""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(
        description='BRB诊断命令行工具 - 频响异常诊断',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python brb_diagnosis_cli.py --input test_data.csv --output result.json
  python brb_diagnosis_cli.py --input test_data.csv --output result.json --baseline ./baseline_data
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='输入CSV文件路径 (格式: frequency,amplitude)')
    parser.add_argument('--output', '-o', required=True,
                        help='输出JSON文件路径')
    parser.add_argument('--baseline', '-b', default=None,
                        help='基线数据目录 (可选，默认使用程序内置路径)')
    parser.add_argument('--mode', '-m', default='er',
                        choices=['er', 'simple'],
                        help='BRB推理模式: er(默认,增强版) 或 simple(简化版)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    # 解决导入路径
    fmfd_root = resolve_import_path()
    
    try:
        # 导入FMFD模块
        from FMFD.baseline.baseline import align_to_frequency
        from FMFD.baseline.config import BASELINE_ARTIFACTS, BASELINE_META, BAND_RANGES
        from FMFD.features.extract import extract_system_features
        from FMFD.BRB.system_brb import system_level_infer
        from FMFD.BRB.module_brb import module_level_infer
        
        if args.verbose:
            print(f"[INFO] FMFD模块导入成功", file=sys.stderr)
            print(f"[INFO] 工作目录: {fmfd_root}", file=sys.stderr)
        
        # 1. 读取输入数据
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[错误] 输入文件不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        
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
        result = {
            "status": "success",
            "input_file": str(input_path.absolute()),
            "data_points": len(freq_raw),
            "frequency_range": {
                "min": float(freq_raw.min()),
                "max": float(freq_raw.max())
            },
            "features": {k: float(v) for k, v in features.items()},
            "system_diagnosis": {k: float(v) for k, v in sys_probs.items()},
            "module_diagnosis": {k: float(v) for k, v in mod_probs.items()},
            "mode": args.mode
        }
        
        # 6. 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if args.verbose:
            print(f"[INFO] 结果已保存到: {output_path}", file=sys.stderr)
            print("\n[系统级诊断结果]", file=sys.stderr)
            for k, v in sys_probs.items():
                print(f"  {k}: {v:.4f}", file=sys.stderr)
            print("\n[模块级诊断TOP5]", file=sys.stderr)
            sorted_mods = sorted(mod_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            for k, v in sorted_mods:
                print(f"  {k}: {v:.4f}", file=sys.stderr)
        
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
