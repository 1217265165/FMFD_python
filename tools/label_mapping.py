#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一标签映射模块 - 系统级与模块级标签定义

该模块定义了系统级故障类别的中英文映射关系，确保：
1. 所有诊断输出、标签读取、评估脚本使用统一的映射
2. 输出顺序固定，不依赖 sorted() 推断
3. 前端展示、评估脚本、softmax 层都使用相同顺序

使用方式：
    from tools.label_mapping import (
        SYS_CLASS_TO_CN, CN_TO_SYS_CLASS, 
        SYS_LABEL_ORDER_CN, SYS_LABEL_ORDER_EN,
        normalize_module_name
    )
"""

# ============== 系统级类别映射（固定，不允许排序推断）==============

# 系统级故障类别：英文 -> 中文
SYS_CLASS_TO_CN = {
    "normal": "正常",
    "amp_error": "幅度失准",
    "freq_error": "频率失准",
    "ref_error": "参考电平失准",
}

# 系统级故障类别：中文 -> 英文
CN_TO_SYS_CLASS = {v: k for k, v in SYS_CLASS_TO_CN.items()}

# 统一系统输出顺序（前端、评估、softmax都用这一套）
SYS_LABEL_ORDER_CN = ["正常", "幅度失准", "频率失准", "参考电平失准"]
SYS_LABEL_ORDER_EN = ["normal", "amp_error", "freq_error", "ref_error"]

# 系统级类别数量
NUM_SYSTEM_CLASSES = len(SYS_LABEL_ORDER_CN)


# ============== 禁用模块列表（单频段模式）==============

# 单频段模式下禁用的模块（不参与 TopK 排序和命中统计）
DISABLED_MODULES = ["前置放大器"]


# ============== 模块名规范化 ==============

def normalize_module_name(name: str) -> str:
    """规范化模块名字符串，防止"看起来一样但不相等"的问题。
    
    处理规则：
    1. 去掉首尾空格
    2. 全角半角括号统一（中文括号 → 英文括号）
    3. 连续空格压缩为单个空格
    4. 保留括号内描述（如"本振源（谐波发生器）"）
    
    Parameters
    ----------
    name : str
        原始模块名字符串
        
    Returns
    -------
    str
        规范化后的模块名
    """
    if not name or not isinstance(name, str):
        return ""
    
    # 1. 去掉首尾空格
    result = name.strip()
    
    # 2. 全角半角括号统一（中文括号 → 英文括号）
    result = result.replace("（", "(").replace("）", ")")
    
    # 3. 连续空格压缩为单个空格
    import re
    result = re.sub(r'\s+', ' ', result)
    
    return result


def is_module_disabled(module_name: str) -> bool:
    """检查模块是否被禁用。
    
    Parameters
    ----------
    module_name : str
        模块名称
        
    Returns
    -------
    bool
        True 表示模块被禁用，不参与 TopK 和命中统计
    """
    normalized = normalize_module_name(module_name)
    return any(
        normalize_module_name(disabled) == normalized 
        for disabled in DISABLED_MODULES
    )


def get_topk_modules(module_probs: dict, k: int = 3, skip_disabled: bool = True) -> list:
    """获取概率最高的前K个模块，可选跳过禁用模块。
    
    Parameters
    ----------
    module_probs : dict
        模块概率分布字典，key=模块名，value=概率
    k : int
        返回的模块数量
    skip_disabled : bool
        是否跳过禁用模块，默认 True
        
    Returns
    -------
    list
        [(模块名, 概率), ...] 元组列表，按概率降序排列
    """
    items = list(module_probs.items())
    
    if skip_disabled:
        items = [(name, prob) for name, prob in items if not is_module_disabled(name)]
    
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    return sorted_items[:k]


# ============== 标签验证工具 ==============

def validate_system_class(sys_class: str, allow_none: bool = True) -> bool:
    """验证系统级类别是否有效。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
    allow_none : bool
        是否允许 None/空值（normal 类型可能无 system_fault_class）
        
    Returns
    -------
    bool
        True 表示有效
    """
    if sys_class is None or sys_class == "":
        return allow_none
    
    return sys_class in SYS_CLASS_TO_CN or sys_class in CN_TO_SYS_CLASS


def get_system_class_cn(sys_class: str) -> str:
    """获取系统级类别的中文名称。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
        
    Returns
    -------
    str
        中文类别名称，如果无法识别则返回原值
    """
    if sys_class is None:
        return "正常"  # None 默认为正常
    
    if sys_class in SYS_CLASS_TO_CN:
        return SYS_CLASS_TO_CN[sys_class]
    
    if sys_class in CN_TO_SYS_CLASS:
        return sys_class  # 已经是中文
    
    return sys_class  # 无法识别，返回原值


def get_system_class_en(sys_class: str) -> str:
    """获取系统级类别的英文名称。
    
    Parameters
    ----------
    sys_class : str
        系统级类别（英文或中文）
        
    Returns
    -------
    str
        英文类别名称，如果无法识别则返回原值
    """
    if sys_class is None:
        return "normal"  # None 默认为正常
    
    if sys_class in CN_TO_SYS_CLASS:
        return CN_TO_SYS_CLASS[sys_class]
    
    if sys_class in SYS_CLASS_TO_CN:
        return sys_class  # 已经是英文
    
    return sys_class  # 无法识别，返回原值
