#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成特征使用情况调试文件 feature_usage_debug.json

用于验证"Ours"方法的特征使用情况，确保X1-X22特征正确分流到系统层和模块层。
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from BRB.system_brb import SystemBRBConfig
from methods.ours_adapter import OursAdapter


def generate_feature_usage_debug():
    """生成特征使用调试信息"""
    
    # 创建Ours方法实例
    ours = OursAdapter()
    config = ours.config
    
    # 系统层使用的特征
    system_used = []
    
    # 基础特征(X1-X5)总是被使用
    system_used.extend(['X1', 'X2', 'X3', 'X4', 'X5'])
    
    # 如果启用扩展特征
    if config.use_extended_features:
        # 包络/残差特征(X11-X15)用于系统层
        system_used.extend(['X11', 'X12', 'X13', 'X14', 'X15'])
        # 频率对齐特征(X16-X18)用于系统层频率识别
        system_used.extend(['X16', 'X17', 'X18'])
        # 幅度细粒度特征(X19-X22)用于系统层幅度/参考识别
        system_used.extend(['X19', 'X20', 'X21', 'X22'])
    
    # 模块层按异常类型分流的特征
    module_used_by_anomaly = {
        "amp_error": [
            # 幅度模块使用的特征
            "X1",  # 幅度偏移
            "X2",  # 带内平坦度
            "X5",  # 缩放一致性
            "X6",  # 纹波
            "X7",  # 增益非线性
            "X11",  # 包络超出率
            "X12",  # 最大违规
            "X13",  # 违规能量
            "X19",  # 低频斜率
            "X20",  # 峰度
            "X21",  # 峰值数
            "X22",  # 主频占比
            # 传统特征
            "step_score", "ripple_var", "bias", "gain"
        ],
        "freq_error": [
            # 频率模块使用的特征
            "X4",  # 频率标度非线性
            "X14",  # 低频残差
            "X15",  # 高频残差
            "X16",  # 频移
            "X17",  # 频率缩放
            "X18",  # 频率平移
            # 传统特征
            "df", "step_score"
        ],
        "ref_error": [
            # 参考电平模块使用的特征
            "X1",  # 幅度偏移
            "X3",  # 高频衰减
            "X5",  # 缩放一致性
            "X11",  # 包络超出率
            "X12",  # 最大违规
            "X13",  # 违规能量
            # 传统特征
            "res_slope", "bias", "gain"
        ]
    }
    
    # 模块层传统特征（所有异常类型共用）
    module_used_common = [
        "step_score", "res_slope", "ripple_var", "df", "viol_rate", "bias", "gain"
    ]
    
    # 生成调试信息
    debug_info = {
        "summary": {
            "total_features_available": 22,
            "system_layer_features": len(system_used),
            "module_layer_amp_features": len(module_used_by_anomaly["amp_error"]),
            "module_layer_freq_features": len(module_used_by_anomaly["freq_error"]),
            "module_layer_ref_features": len(module_used_by_anomaly["ref_error"]),
            "extended_features_enabled": config.use_extended_features,
        },
        "system_used": system_used,
        "module_used_by_anomaly": module_used_by_anomaly,
        "module_used_common": module_used_common,
        "feature_descriptions": {
            "X1": "整体幅度偏移 (amplitude offset)",
            "X2": "带内平坦度 (inband flatness / ripple variance)",
            "X3": "高频段衰减斜率 (HF attenuation slope)",
            "X4": "频率标度非线性度 (frequency scale nonlinearity)",
            "X5": "幅度缩放一致性 (amplitude scale consistency)",
            "X6": "纹波方差 (ripple variance)",
            "X7": "增益非线性 (gain nonlinearity / max step)",
            "X8": "本振泄漏 (LO leakage)",
            "X9": "调谐线性度残差 (tuning linearity residual)",
            "X10": "频段幅度一致性 (band amplitude consistency)",
            "X11": "超出动态包络点占比 (envelope overrun rate)",
            "X12": "最大包络违规幅度 (max envelope violation)",
            "X13": "包络违规能量 (envelope violation energy)",
            "X14": "低频段残差均值 (low band residual mean)",
            "X15": "高频段残差方差 (high band residual std)",
            "X16": "互相关滞后 (correlation shift / frequency shift proxy)",
            "X17": "频轴缩放因子 (frequency axis warp scale)",
            "X18": "频轴平移因子 (frequency axis warp bias)",
            "X19": "低频段斜率 (low frequency slope)",
            "X20": "去趋势残差峰度 (detrended residual kurtosis)",
            "X21": "残差峰值数量 (residual peak count)",
            "X22": "残差主频能量占比 (dominant frequency energy ratio)",
        },
        "configuration": {
            "alpha": config.alpha,
            "overall_threshold": config.overall_threshold,
            "max_prob_threshold": config.max_prob_threshold,
            "attribute_weights_length": len(config.attribute_weights),
            "rule_weights": list(config.rule_weights),
        },
        "notes": [
            "系统层使用X1-X5基础特征 + X11-X22扩展特征（共17个）",
            "模块层根据系统层检测到的异常类型进行特征分流：",
            "  - 幅度异常模块：使用幅度相关特征（X1,X2,X5,X6,X7,X11-X13,X19-X22）",
            "  - 频率异常模块：使用频率相关特征（X4,X14-X18）",
            "  - 参考异常模块：使用参考相关特征（X1,X3,X5,X11-X13）",
            "所有模块层还使用传统特征：step_score, res_slope, ripple_var, df, viol_rate, bias, gain"
        ]
    }
    
    # 保存到Output目录
    output_dir = repo_root / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_usage_debug.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Feature usage debug file generated: {output_path}")
    print(f"\n总结:")
    print(f"  - 可用特征总数: {debug_info['summary']['total_features_available']}")
    print(f"  - 系统层使用: {debug_info['summary']['system_layer_features']} 个特征")
    print(f"  - 模块层（幅度）: {debug_info['summary']['module_layer_amp_features']} 个特征")
    print(f"  - 模块层（频率）: {debug_info['summary']['module_layer_freq_features']} 个特征")
    print(f"  - 模块层（参考）: {debug_info['summary']['module_layer_ref_features']} 个特征")
    print(f"  - 扩展特征已启用: {debug_info['summary']['extended_features_enabled']}")
    
    return debug_info


if __name__ == "__main__":
    generate_feature_usage_debug()
