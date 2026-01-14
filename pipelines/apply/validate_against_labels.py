#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用核对脚本：批量跑诊断并对照 labels.json

该脚本用于验证诊断流程的正确性：
1. 遍历 raw_curves 目录下所有 sim_*.csv 文件
2. 对每个文件运行 BRB 诊断
3. 与 labels.json 中的真值对比
4. 生成统计报告和混淆矩阵

使用方式：
    python pipelines/apply/validate_against_labels.py \
        --raw_dir Output/sim_spectrum/raw_curves \
        --labels Output/sim_spectrum/labels.json \
        --out_dir Output/apply_check \
        --topk 3

输出文件：
    - system_check.csv: 每条样本的 GT vs Pred + 是否命中
    - module_check.csv: Top1/Top3 是否命中
    - mismatch_cases.csv: 只存错的样本
    - confusion_system_apply.png: 系统级混淆矩阵
    - summary.json: 总统计
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def parse_sample_id(filepath: Path) -> str:
    """从文件名解析 sample_id。"""
    match = re.search(r'(sim_\d{5})', filepath.stem)
    return match.group(1) if match else filepath.stem


def run_diagnosis(
    csv_path: Path, 
    frequency: np.ndarray,
    rrs: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    band_ranges: List[Tuple[float, float]],
    mode: str = 'sub_brb'
) -> Dict:
    """对单个样本运行诊断。
    
    直接调用诊断函数而非 subprocess，提高稳定性。
    """
    import pandas as pd
    from baseline.baseline import align_to_frequency
    from features.extract import extract_system_features
    from BRB.system_brb import system_level_infer
    from BRB.module_brb import module_level_infer
    from tools.label_mapping import get_topk_modules, SYS_CLASS_TO_CN
    
    # 读取数据
    df = pd.read_csv(csv_path)
    freq_raw = df.iloc[:, 0].values
    amp_raw = df.iloc[:, 1].values
    
    # 对齐频率
    amp = align_to_frequency(frequency, freq_raw, amp_raw)
    
    # 提取特征
    features = extract_system_features(frequency, rrs, bounds, band_ranges, amp)
    
    # 系统级推理
    sys_result = system_level_infer(features, mode=mode)
    sys_probs = sys_result.get('probabilities', sys_result)
    
    # 模块级推理
    mod_probs = module_level_infer(features, sys_result)
    
    # 获取预测结果
    predicted_class = max(sys_probs, key=sys_probs.get)
    max_prob = max(sys_probs.values())
    is_normal = sys_result.get('is_normal', False)
    
    # TopK 模块（跳过禁用模块）
    topk_modules = get_topk_modules(mod_probs, k=10, skip_disabled=True)
    
    return {
        'system_diagnosis': {
            'probabilities': sys_probs,
            'predicted_class': predicted_class,
            'max_prob': max_prob,
            'is_normal': is_normal,
        },
        'module_diagnosis': {
            'probabilities': mod_probs,
            'topk': topk_modules,
        },
        'features': features,
    }


def load_labels(labels_path: Path) -> Dict:
    """加载 labels.json。"""
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_module_hit(pred_topk: List[Tuple[str, float]], gt_module: Optional[str], k: int = 3) -> Dict:
    """检查模块命中情况。
    
    Parameters
    ----------
    pred_topk : List[Tuple[str, float]]
        预测的 TopK 模块列表
    gt_module : Optional[str]
        真实模块名（normal 类型为 None）
    k : int
        检查前 k 个
        
    Returns
    -------
    Dict
        命中结果
    """
    from tools.label_mapping import normalize_module_name, is_module_disabled
    
    if gt_module is None:
        return {'top1_hit': 'NA', 'topk_hit': 'NA', 'skip_reason': 'normal_type'}
    
    # 规范化真实模块名
    gt_normalized = normalize_module_name(gt_module)
    
    # 检查是否为禁用模块
    if is_module_disabled(gt_module):
        return {'top1_hit': 'NA', 'topk_hit': 'NA', 'skip_reason': 'disabled_module'}
    
    # 获取前 k 个预测
    topk = pred_topk[:k]
    pred_names = [normalize_module_name(name) for name, _ in topk]
    
    top1_hit = pred_names[0] == gt_normalized if pred_names else False
    topk_hit = gt_normalized in pred_names
    
    return {
        'top1_hit': top1_hit,
        'topk_hit': topk_hit,
        'pred_top1': topk[0][0] if topk else None,
        'pred_topk': [name for name, _ in topk],
    }


def generate_confusion_matrix(
    system_results: List[Dict],
    labels_order: List[str],
    out_path: Path
):
    """生成系统级混淆矩阵图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 构建混淆矩阵
        n_classes = len(labels_order)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        label_to_idx = {label: i for i, label in enumerate(labels_order)}
        
        for result in system_results:
            gt = result['gt_sys_cn']
            pred = result['pred_sys_cn']
            
            gt_idx = label_to_idx.get(gt, -1)
            pred_idx = label_to_idx.get(pred, -1)
            
            if gt_idx >= 0 and pred_idx >= 0:
                confusion[gt_idx, pred_idx] += 1
        
        # 绘制混淆矩阵
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=labels_order,
            yticklabels=labels_order,
            ylabel='真实类别 (Ground Truth)',
            xlabel='预测类别 (Predicted)',
            title='系统级诊断混淆矩阵 (Application Check)'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 在每个格子中显示数值
        thresh = confusion.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(confusion[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if confusion[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return confusion
        
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过混淆矩阵图生成", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='批量诊断核对脚本 - 对照 labels.json 验证诊断结果'
    )
    parser.add_argument('--raw_dir', required=True,
                        help='raw_curves 目录路径')
    parser.add_argument('--labels', required=True,
                        help='labels.json 文件路径')
    parser.add_argument('--out_dir', required=True,
                        help='输出目录路径')
    parser.add_argument('--topk', type=int, default=3,
                        help='TopK 模块命中检查数量 (默认: 3)')
    parser.add_argument('--mode', default='sub_brb',
                        choices=['er', 'simple', 'sub_brb'],
                        help='BRB推理模式 (默认: sub_brb)')
    parser.add_argument('--baseline_npz', default='Output/baseline_artifacts.npz',
                        help='基线 NPZ 文件路径')
    parser.add_argument('--baseline_meta', default='Output/baseline_meta.json',
                        help='基线 meta JSON 文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    # 导入标签映射
    from tools.label_mapping import (
        SYS_CLASS_TO_CN, CN_TO_SYS_CLASS, 
        SYS_LABEL_ORDER_CN, DISABLED_MODULES,
        normalize_module_name
    )
    from baseline.config import BAND_RANGES
    
    # 路径处理
    raw_dir = Path(args.raw_dir)
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入
    if not raw_dir.exists():
        print(f"[错误] raw_dir 不存在: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_path.exists():
        print(f"[错误] labels.json 不存在: {labels_path}", file=sys.stderr)
        sys.exit(1)
    
    # 加载基线数据
    baseline_npz = Path(args.baseline_npz)
    baseline_meta = Path(args.baseline_meta)
    
    if not baseline_npz.exists():
        # 尝试相对于 repo_root
        baseline_npz = repo_root / args.baseline_npz
        baseline_meta = repo_root / args.baseline_meta
    
    if not baseline_npz.exists():
        print(f"[错误] 基线文件不存在: {baseline_npz}", file=sys.stderr)
        sys.exit(1)
    
    art = np.load(baseline_npz)
    frequency = art["frequency"]
    rrs = art["rrs"]
    bounds = (art["upper"], art["lower"])
    
    with open(baseline_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    band_ranges = meta.get("band_ranges", BAND_RANGES)
    
    # 加载标签
    labels = load_labels(labels_path)
    print(f"[INFO] 加载标签: {len(labels)} 条", file=sys.stderr)
    
    # 获取 CSV 文件列表
    csv_files = sorted(raw_dir.glob("sim_*.csv"))
    print(f"[INFO] 发现 CSV 文件: {len(csv_files)} 个", file=sys.stderr)
    
    # 统计变量
    system_results = []
    module_results = []
    mismatch_cases = []
    
    skip_reasons = {
        'no_label': 0,
        'disabled_module': 0,
        'normal_type': 0,
        'error': 0,
    }
    
    # 遍历处理
    for i, csv_path in enumerate(csv_files):
        sample_id = parse_sample_id(csv_path)
        
        if args.verbose and (i + 1) % 50 == 0:
            print(f"[INFO] 处理进度: {i + 1}/{len(csv_files)}", file=sys.stderr)
        
        # 检查是否有标签
        if sample_id not in labels:
            skip_reasons['no_label'] += 1
            continue
        
        gt = labels[sample_id]
        gt_type = gt.get("type", "unknown")
        gt_sys_class = gt.get("system_fault_class")
        gt_module = gt.get("module")
        
        # 获取系统级中文标签
        if gt_sys_class is None or gt_type == "normal":
            gt_sys_cn = "正常"
        else:
            gt_sys_cn = SYS_CLASS_TO_CN.get(gt_sys_class, gt_sys_class)
        
        # 运行诊断
        try:
            result = run_diagnosis(
                csv_path, frequency, rrs, bounds, band_ranges, mode=args.mode
            )
        except Exception as e:
            skip_reasons['error'] += 1
            if args.verbose:
                print(f"[WARN] {sample_id} 诊断失败: {e}", file=sys.stderr)
            continue
        
        pred_sys_cn = result['system_diagnosis']['predicted_class']
        pred_probs = result['system_diagnosis']['probabilities']
        pred_topk = result['module_diagnosis']['topk']
        
        # 系统级命中检查
        sys_hit = (pred_sys_cn == gt_sys_cn)
        
        system_results.append({
            'sample_id': sample_id,
            'gt_type': gt_type,
            'gt_sys_class': gt_sys_class,
            'gt_sys_cn': gt_sys_cn,
            'pred_sys_cn': pred_sys_cn,
            'pred_max_prob': result['system_diagnosis']['max_prob'],
            'sys_hit': sys_hit,
            'pred_probs': pred_probs,
        })
        
        # 模块级命中检查
        mod_hit_result = check_module_hit(pred_topk, gt_module, k=args.topk)
        
        if mod_hit_result['top1_hit'] == 'NA':
            skip_reasons[mod_hit_result.get('skip_reason', 'normal_type')] += 1
        
        module_results.append({
            'sample_id': sample_id,
            'gt_module': gt_module,
            **mod_hit_result,
        })
        
        # 记录不匹配的样本
        if not sys_hit or (mod_hit_result['topk_hit'] not in [True, 'NA']):
            mismatch_cases.append({
                'sample_id': sample_id,
                'csv_path': str(csv_path),
                'gt_sys_cn': gt_sys_cn,
                'pred_sys_cn': pred_sys_cn,
                'sys_hit': sys_hit,
                'gt_module': gt_module,
                'top1_hit': mod_hit_result['top1_hit'],
                'topk_hit': mod_hit_result['topk_hit'],
                'pred_top1': mod_hit_result.get('pred_top1'),
            })
    
    # 计算统计数据
    n_files = len(csv_files)
    n_labels = len(labels)
    n_evaluated = len(system_results)
    
    sys_correct = sum(1 for r in system_results if r['sys_hit'])
    sys_accuracy = sys_correct / n_evaluated if n_evaluated > 0 else 0.0
    
    # 模块统计（排除 NA）
    mod_valid = [r for r in module_results if r['top1_hit'] not in ['NA', 'NA']]
    n_mod_valid = len(mod_valid)
    mod_top1_correct = sum(1 for r in mod_valid if r['top1_hit'] == True)
    mod_topk_correct = sum(1 for r in mod_valid if r['topk_hit'] == True)
    
    mod_top1_acc = mod_top1_correct / n_mod_valid if n_mod_valid > 0 else 0.0
    mod_topk_acc = mod_topk_correct / n_mod_valid if n_mod_valid > 0 else 0.0
    
    # 输出统计
    print("\n" + "="*60, file=sys.stderr)
    print("验证结果摘要", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"  CSV 文件数: {n_files}", file=sys.stderr)
    print(f"  Labels 数量: {n_labels}", file=sys.stderr)
    print(f"  成功诊断数: {n_evaluated}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  系统级准确率: {sys_accuracy:.4f} ({sys_correct}/{n_evaluated})", file=sys.stderr)
    print(f"  模块 Top1 准确率: {mod_top1_acc:.4f} ({mod_top1_correct}/{n_mod_valid})", file=sys.stderr)
    print(f"  模块 Top{args.topk} 准确率: {mod_topk_acc:.4f} ({mod_topk_correct}/{n_mod_valid})", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  跳过原因统计:", file=sys.stderr)
    for reason, count in skip_reasons.items():
        print(f"    {reason}: {count}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  不匹配样本数: {len(mismatch_cases)}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    # 保存 system_check.csv
    system_check_path = out_dir / "system_check.csv"
    with open(system_check_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['sample_id', 'gt_type', 'gt_sys_class', 'gt_sys_cn', 
                      'pred_sys_cn', 'pred_max_prob', 'sys_hit']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(system_results)
    print(f"[INFO] 已保存: {system_check_path}", file=sys.stderr)
    
    # 保存 module_check.csv
    module_check_path = out_dir / "module_check.csv"
    with open(module_check_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['sample_id', 'gt_module', 'top1_hit', 'topk_hit', 
                      'pred_top1', 'skip_reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(module_results)
    print(f"[INFO] 已保存: {module_check_path}", file=sys.stderr)
    
    # 保存 mismatch_cases.csv
    mismatch_path = out_dir / "mismatch_cases.csv"
    with open(mismatch_path, 'w', newline='', encoding='utf-8-sig') as f:
        if mismatch_cases:
            fieldnames = list(mismatch_cases[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatch_cases)
    print(f"[INFO] 已保存: {mismatch_path}", file=sys.stderr)
    
    # 生成混淆矩阵
    confusion_path = out_dir / "confusion_system_apply.png"
    confusion = generate_confusion_matrix(
        system_results, SYS_LABEL_ORDER_CN, confusion_path
    )
    if confusion is not None:
        print(f"[INFO] 已保存: {confusion_path}", file=sys.stderr)
    
    # 保存 summary.json
    summary = {
        'n_files': n_files,
        'n_labels': n_labels,
        'n_evaluated': n_evaluated,
        'system_accuracy': sys_accuracy,
        'system_correct': sys_correct,
        'module_valid_samples': n_mod_valid,
        'module_top1_accuracy': mod_top1_acc,
        'module_top1_correct': mod_top1_correct,
        f'module_top{args.topk}_accuracy': mod_topk_acc,
        f'module_top{args.topk}_correct': mod_topk_correct,
        'skip_reasons': skip_reasons,
        'n_mismatch': len(mismatch_cases),
        'disabled_modules': list(DISABLED_MODULES),
        'config': {
            'topk': args.topk,
            'mode': args.mode,
            'raw_dir': str(raw_dir),
            'labels_path': str(labels_path),
        },
    }
    
    if confusion is not None:
        summary['confusion_matrix'] = confusion.tolist()
    
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存: {summary_path}", file=sys.stderr)
    
    print(f"\n[完成] 所有输出文件已保存到: {out_dir}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
