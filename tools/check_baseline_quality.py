#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线质量验收脚本

用于验证基线/包络计算质量，防止回归。

输入：
- Output/baseline_artifacts.npz
- normal_response_data/

输出（控制台+CSV）：
- coverage_total（>=0.97）
- coverage_per_curve 的均值/最小值（最小值>=0.93）
- width_min/median/max（max<=0.60）
- width_smoothness：np.std(np.diff(width)) 要小

脚本 exit code：不达标返回 1（CI/本地可用）
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 质量阈值
COVERAGE_TOTAL_MIN = 0.97      # 总覆盖率必须 >= 97%
COVERAGE_PER_CURVE_MIN = 0.93  # 每条曲线覆盖率最小值必须 >= 93%
WIDTH_MAX_THRESHOLD = 0.60     # 宽度最大值必须 <= 0.60 dB
WIDTH_SMOOTHNESS_MAX = 0.02    # 宽度平滑度（diff 的 std）必须 < 0.02


def check_baseline_quality(baseline_artifacts_path: Path, 
                           normal_data_path: Path,
                           output_csv: Path = None,
                           verbose: bool = True) -> dict:
    """检查基线质量。
    
    Parameters
    ----------
    baseline_artifacts_path : Path
        baseline_artifacts.npz 文件路径
    normal_data_path : Path
        normal_response_data 目录路径
    output_csv : Path, optional
        输出 CSV 文件路径
    verbose : bool
        是否打印详细信息
        
    Returns
    -------
    dict
        质量指标和是否通过
    """
    # 加载基线产物
    if not baseline_artifacts_path.exists():
        raise FileNotFoundError(f"基线文件不存在: {baseline_artifacts_path}")
    
    art = np.load(baseline_artifacts_path)
    frequency = art["frequency"]
    traces = art["traces"]
    rrs = art["rrs"]
    upper = art["upper"]
    lower = art["lower"]
    
    n_traces, n_points = traces.shape
    
    if verbose:
        print(f"[CheckBaseline] 加载基线: {n_traces} 条曲线, {n_points} 个频点")
    
    # 1. 计算总覆盖率
    all_in_bounds = 0
    total_points = 0
    coverage_per_curve = []
    
    for i in range(n_traces):
        trace = traces[i]
        in_bounds = (trace >= lower) & (trace <= upper)
        curve_coverage = np.mean(in_bounds)
        coverage_per_curve.append(curve_coverage)
        all_in_bounds += np.sum(in_bounds)
        total_points += len(trace)
    
    coverage_total = all_in_bounds / total_points
    coverage_mean = np.mean(coverage_per_curve)
    coverage_min = np.min(coverage_per_curve)
    
    # 2. 计算宽度统计
    width = upper - lower
    width_min = float(np.min(width))
    width_median = float(np.median(width))
    width_max = float(np.max(width))
    width_smoothness = float(np.std(np.diff(width)))
    
    # 3. 验证通过/失败
    checks = {
        'coverage_total': {
            'value': coverage_total,
            'threshold': COVERAGE_TOTAL_MIN,
            'passed': coverage_total >= COVERAGE_TOTAL_MIN,
            'description': '总覆盖率 >= 97%',
        },
        'coverage_min': {
            'value': coverage_min,
            'threshold': COVERAGE_PER_CURVE_MIN,
            'passed': coverage_min >= COVERAGE_PER_CURVE_MIN,
            'description': '每条曲线覆盖率最小值 >= 93%',
        },
        'width_max': {
            'value': width_max,
            'threshold': WIDTH_MAX_THRESHOLD,
            'passed': width_max <= WIDTH_MAX_THRESHOLD,
            'description': '包络宽度最大值 <= 0.60 dB',
        },
        'width_smoothness': {
            'value': width_smoothness,
            'threshold': WIDTH_SMOOTHNESS_MAX,
            'passed': width_smoothness < WIDTH_SMOOTHNESS_MAX,
            'description': '包络宽度平滑度（diff std）< 0.02',
        },
    }
    
    all_passed = all(c['passed'] for c in checks.values())
    
    results = {
        'n_traces': n_traces,
        'n_points': n_points,
        'coverage_total': coverage_total,
        'coverage_mean': coverage_mean,
        'coverage_min': coverage_min,
        'width_min': width_min,
        'width_median': width_median,
        'width_max': width_max,
        'width_smoothness': width_smoothness,
        'checks': checks,
        'all_passed': all_passed,
    }
    
    # 打印结果
    if verbose:
        print("\n" + "="*60)
        print("基线质量检查结果")
        print("="*60)
        print(f"  曲线数: {n_traces}")
        print(f"  频点数: {n_points}")
        print(f"\n  覆盖率:")
        print(f"    总覆盖率: {coverage_total:.4f} (阈值 >= {COVERAGE_TOTAL_MIN})")
        print(f"    均值: {coverage_mean:.4f}")
        print(f"    最小值: {coverage_min:.4f} (阈值 >= {COVERAGE_PER_CURVE_MIN})")
        print(f"\n  包络宽度:")
        print(f"    最小: {width_min:.4f} dB")
        print(f"    中位数: {width_median:.4f} dB")
        print(f"    最大: {width_max:.4f} dB (阈值 <= {WIDTH_MAX_THRESHOLD})")
        print(f"    平滑度: {width_smoothness:.6f} (阈值 < {WIDTH_SMOOTHNESS_MAX})")
        print("\n  检查项:")
        for name, check in checks.items():
            status = "✓ 通过" if check['passed'] else "✗ 失败"
            print(f"    {name}: {status}")
        print("\n" + "="*60)
        if all_passed:
            print("✓ 基线质量检查通过")
        else:
            print("✗ 基线质量检查失败")
        print("="*60)
    
    # 保存 CSV
    if output_csv:
        df = pd.DataFrame([{
            'n_traces': n_traces,
            'n_points': n_points,
            'coverage_total': coverage_total,
            'coverage_mean': coverage_mean,
            'coverage_min': coverage_min,
            'width_min': width_min,
            'width_median': width_median,
            'width_max': width_max,
            'width_smoothness': width_smoothness,
            'all_passed': all_passed,
        }])
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n结果已保存到: {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='基线质量验收脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--baseline', '-b', default='Output/baseline_artifacts.npz',
                        help='基线产物文件路径 (default: Output/baseline_artifacts.npz)')
    parser.add_argument('--normal_data', '-n', default='normal_response_data',
                        help='正常数据目录路径 (default: normal_response_data)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出 CSV 文件路径 (可选)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，只输出最终结果')
    
    args = parser.parse_args()
    
    # 解析路径
    repo_root = Path(__file__).resolve().parents[1]
    baseline_path = repo_root / args.baseline
    normal_path = repo_root / args.normal_data
    output_path = Path(args.output) if args.output else None
    
    try:
        results = check_baseline_quality(
            baseline_path,
            normal_path,
            output_path,
            verbose=not args.quiet
        )
        
        # 返回 exit code
        if results['all_passed']:
            if args.quiet:
                print("PASS")
            sys.exit(0)
        else:
            if args.quiet:
                print("FAIL")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
