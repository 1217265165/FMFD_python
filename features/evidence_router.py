#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据路由器 (Evidence Router)
============================
将检测到的"突变证据"知识映射到模块层，实现候选集收缩、特征注入和报告可解释。

本模块实现知识驱动的 Evidence→Module 路由器：
1. Level-1: 按系统级异常类型(amp/freq/ref)收缩模块组
2. Level-2: 按证据的频段/形态进一步缩小到具体模块子集
3. 输出解释性字段用于报告

使用方法:
    from features.evidence_router import route_modules_by_evidence
    
    result = route_modules_by_evidence(evidence, system_result, config)
    candidate_modules = result['candidate_modules']
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Set
import numpy as np

# ============================================================================
# 模块定义与索引
# ============================================================================

MODULE_LABELS: List[str] = [
    "衰减器",           # 0
    "前置放大器",        # 1 - 前放OFF时永远排除
    "低频段前置低通滤波器",  # 2
    "低频段第一混频器",    # 3
    "高频段YTF滤波器",   # 4
    "高频段混频器",      # 5
    "时钟振荡器",        # 6
    "时钟合成与同步网络",  # 7
    "本振源（谐波发生器）", # 8
    "本振混频组件",      # 9
    "校准源",          # 10
    "存储器",          # 11
    "校准信号开关",      # 12
    "中频放大器",        # 13
    "ADC",             # 14
    "数字RBW",         # 15
    "数字放大器",        # 16
    "数字检波器",        # 17
    "VBW滤波器",       # 18
    "电源模块",         # 19
    "未定义/其他",       # 20
]

# 模块名称到索引的映射
MODULE_NAME_TO_IDX = {name: idx for idx, name in enumerate(MODULE_LABELS)}

# ============================================================================
# 模块分组
# ============================================================================

MODULE_GROUPS: Dict[str, List[str]] = {
    # 幅度链路模块组
    'amp_group': [
        '衰减器', '前置放大器', '中频放大器', '数字放大器', 'ADC',
        '数字RBW', '数字检波器', 'VBW滤波器'
    ],
    # 频率链路模块组
    'freq_group': [
        '时钟振荡器', '时钟合成与同步网络', '本振源（谐波发生器）', '本振混频组件',
        '高频段YTF滤波器', '高频段混频器', '低频段前置低通滤波器', '低频段第一混频器'
    ],
    # 参考电平链路模块组
    'ref_group': [
        '校准源', '存储器', '校准信号开关', '衰减器'
    ],
    # 通用模块
    'other_group': [
        '电源模块', '未定义/其他'
    ],
}

# 系统级异常类型到模块组的映射
FAULT_TYPE_TO_GROUP: Dict[str, str] = {
    'amp': 'amp_group',
    'freq': 'freq_group',
    'ref': 'ref_group',
    '幅度失准': 'amp_group',
    '频率失准': 'freq_group',
    '参考电平失准': 'ref_group',
}

# ============================================================================
# 频段→模块的知识映射（单频段简化版）
# ============================================================================

# 频段边界（Hz）- 用于判断异常发生在哪个频段
# 单频段数据时，这些边界用于判断相对位置
FREQUENCY_BAND_BOUNDARIES = {
    'low_band': (1e4, 3e9),       # 低频段
    'mid_band': (3e9, 8.2e9),     # 中频段
    'high_band': (8.2e9, 67e9),   # 高频段
}

# 频段→主要候选模块的映射（知识库）
BAND_TO_MODULES: Dict[str, List[str]] = {
    # 低频段主要涉及的模块
    'low_band': [
        '低频段前置低通滤波器',
        '低频段第一混频器',
        '中频放大器',
        'ADC',
        '衰减器',
    ],
    # 中频段主要涉及的模块
    'mid_band': [
        '高频段YTF滤波器',
        '高频段混频器',
        '本振混频组件',
        '中频放大器',
        'ADC',
        '衰减器',
    ],
    # 高频段主要涉及的模块
    'high_band': [
        '高频段YTF滤波器',
        '高频段混频器',
        '本振源（谐波发生器）',
        '本振混频组件',
        '衰减器',
    ],
}

# ============================================================================
# 突变类型→模块的知识映射
# ============================================================================

# 突变形态特征→候选模块的映射（知识库）
JUMP_TYPE_TO_MODULES: Dict[str, List[str]] = {
    # 阶跃型突变 - 可能是衰减器切换或放大器故障
    'step': [
        '衰减器',
        '中频放大器',
        '数字放大器',
        '校准源',
    ],
    # 尖峰型突变 - 可能是本振泄漏或信号干扰
    'spike': [
        '本振源（谐波发生器）',
        '本振混频组件',
        '时钟振荡器',
        'ADC',
    ],
    # 台阶型突变 - 可能是滤波器或混频器故障
    'plateau': [
        '高频段YTF滤波器',
        '低频段前置低通滤波器',
        '高频段混频器',
        '低频段第一混频器',
    ],
    # 斜坡型突变 - 可能是增益漂移或校准问题
    'ramp': [
        '中频放大器',
        '数字放大器',
        '校准源',
        '存储器',
    ],
    # 振铃/纹波型 - 可能是滤波器或时钟问题
    'ripple': [
        'VBW滤波器',
        '数字RBW',
        '时钟合成与同步网络',
        '数字检波器',
    ],
}

# 包络违例类型→候选模块的映射
ENVELOPE_VIOLATION_TO_MODULES: Dict[str, List[str]] = {
    # 上包络违例（幅度偏高）
    'upper': [
        '数字放大器',
        '中频放大器',
        '衰减器',
        '校准源',
    ],
    # 下包络违例（幅度偏低）
    'lower': [
        '衰减器',
        '中频放大器',
        '高频段YTF滤波器',
        '高频段混频器',
    ],
    # 双向违例
    'both': [
        '衰减器',
        '中频放大器',
        '数字放大器',
        '校准源',
        '存储器',
    ],
}


# ============================================================================
# 路由器配置
# ============================================================================

DEFAULT_ROUTER_CONFIG = {
    # 是否排除前置放大器（前放OFF时）
    'exclude_preamp': True,
    
    # 系统级概率阈值（低于此值认为不确定）
    'system_prob_threshold': 0.35,
    
    # 证据分数阈值（高于此值认为证据有效）
    'evidence_score_threshold': 0.5,
    
    # 最大候选模块数量
    'max_candidates': 10,
    
    # 候选模块先验权重（可选）
    'enable_prior_weights': True,
    
    # 默认权重
    'default_module_weight': 0.1,
}


# ============================================================================
# 路由器主函数
# ============================================================================

def route_modules_by_evidence(
    evidence: Dict[str, Any],
    system_result: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    根据检测证据和系统级推理结果，路由到候选模块集合。
    
    实现两级路由：
    - Level-1：根据系统级异常类型(amp/freq/ref)收缩模块组
    - Level-2：根据证据的频段/形态进一步缩小到具体模块子集
    
    Parameters
    ----------
    evidence : dict
        检测输出的证据字段，可能包含：
        - jump_flag: bool，是否检测到突变
        - jump_type: str，突变类型（step/spike/plateau/ramp/ripple）
        - jump_band_hz: tuple，突变发生的频率范围
        - jump_freq_hz: float，突变峰值频率
        - jump_max_db: float，突变最大幅度
        - jump_max_score: float，突变得分
        - max_env_violation_db: float，最大包络违例幅度
        - env_violation_energy: float，包络违例能量
        - env_violation_band_hz: tuple，包络违例频率范围
        - env_violation_type: str，upper/lower/both
    system_result : dict
        系统级推理结果，包含：
        - top_class: str，最可能的异常类型
        - probabilities: dict，各类异常的概率
    config : dict, optional
        路由器配置参数
        
    Returns
    -------
    dict
        路由结果，包含：
        - candidate_modules: List[int]，候选模块索引列表
        - candidate_module_names: List[str]，候选模块名称列表
        - candidate_groups: List[str]，激活的模块组
        - routing_reason: Dict[str, Any]，路由解释
        - weights: Dict[int, float]，候选模块先验权重
    """
    config = {**DEFAULT_ROUTER_CONFIG, **(config or {})}
    
    # 初始化结果
    result = {
        'candidate_modules': [],
        'candidate_module_names': [],
        'candidate_groups': [],
        'routing_reason': {},
        'weights': {},
    }
    
    # ========== Level-1: 系统级异常类型 → 模块组 ==========
    level1_result = _level1_route_by_system_class(system_result, config)
    result['candidate_groups'] = level1_result['candidate_groups']
    candidate_modules: Set[str] = level1_result['candidate_module_set']
    result['routing_reason']['level1'] = level1_result['reason']
    
    # ========== Level-2: 证据 → 模块子集 ==========
    level2_result = _level2_route_by_evidence(evidence, candidate_modules, config)
    
    # 合并 Level-2 结果
    if level2_result['refined_modules']:
        # Level-2 成功细化，使用细化后的结果
        candidate_modules = level2_result['refined_modules']
        result['routing_reason']['level2'] = level2_result['reason']
    else:
        # Level-2 无法细化，保持 Level-1 结果
        result['routing_reason']['level2'] = {'status': 'no_refinement', 'reason': 'Evidence not strong enough'}
    
    # ========== 后处理 ==========
    
    # 排除前置放大器（如果配置要求）
    if config.get('exclude_preamp', True):
        candidate_modules.discard('前置放大器')
        result['routing_reason']['preamp_excluded'] = True
    
    # 始终包含通用模块
    candidate_modules.update(MODULE_GROUPS['other_group'])
    
    # 限制最大候选数量
    max_candidates = config.get('max_candidates', 10)
    if len(candidate_modules) > max_candidates:
        # 按先验权重排序，保留权重最高的
        candidate_modules = _truncate_by_priority(candidate_modules, level2_result.get('weights', {}), max_candidates)
    
    # 转换为索引和名称列表
    result['candidate_module_names'] = list(candidate_modules)
    result['candidate_modules'] = [MODULE_NAME_TO_IDX[name] for name in candidate_modules if name in MODULE_NAME_TO_IDX]
    
    # 计算先验权重
    if config.get('enable_prior_weights', True):
        result['weights'] = _compute_prior_weights(
            result['candidate_modules'],
            system_result,
            evidence,
            level2_result.get('weights', {})
        )
    
    return result


def _level1_route_by_system_class(
    system_result: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Level-1 路由：根据系统级异常类型收缩模块组。
    
    Parameters
    ----------
    system_result : dict
        系统级推理结果
    config : dict
        路由器配置
        
    Returns
    -------
    dict
        Level-1 路由结果
    """
    result = {
        'candidate_groups': [],
        'candidate_module_set': set(),
        'reason': {},
    }
    
    # 提取系统级概率
    probs = system_result.get('probabilities', {})
    top_class = system_result.get('top_class', '')
    
    # 标准化类名
    class_mapping = {
        'amp': 'amp', '幅度失准': 'amp', 'amplitude': 'amp',
        'freq': 'freq', '频率失准': 'freq', 'frequency': 'freq',
        'ref': 'ref', '参考电平失准': 'ref', 'reference': 'ref',
    }
    
    normalized_class = class_mapping.get(top_class, '')
    
    # 获取各类概率
    amp_prob = probs.get('幅度失准', probs.get('amp', 0.0))
    freq_prob = probs.get('频率失准', probs.get('freq', 0.0))
    ref_prob = probs.get('参考电平失准', probs.get('ref', 0.0))
    
    max_prob = max(amp_prob, freq_prob, ref_prob)
    threshold = config.get('system_prob_threshold', 0.35)
    
    result['reason']['probabilities'] = {
        'amp': amp_prob,
        'freq': freq_prob,
        'ref': ref_prob,
        'max_prob': max_prob,
        'threshold': threshold,
    }
    
    # 根据概率确定激活的模块组
    if max_prob >= threshold:
        # 确定性高，只激活对应的模块组
        if normalized_class == 'amp' or (amp_prob == max_prob and not normalized_class):
            result['candidate_groups'] = ['amp_group']
            result['reason']['selected_class'] = 'amp'
        elif normalized_class == 'freq' or (freq_prob == max_prob and not normalized_class):
            result['candidate_groups'] = ['freq_group']
            result['reason']['selected_class'] = 'freq'
        elif normalized_class == 'ref' or (ref_prob == max_prob and not normalized_class):
            result['candidate_groups'] = ['ref_group']
            result['reason']['selected_class'] = 'ref'
    else:
        # 确定性低，激活所有模块组
        result['candidate_groups'] = ['amp_group', 'freq_group', 'ref_group']
        result['reason']['selected_class'] = 'uncertain'
    
    # 收集候选模块
    for group in result['candidate_groups']:
        if group in MODULE_GROUPS:
            result['candidate_module_set'].update(MODULE_GROUPS[group])
    
    return result


def _level2_route_by_evidence(
    evidence: Dict[str, Any],
    candidate_modules: Set[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Level-2 路由：根据证据进一步细化候选模块。
    
    Parameters
    ----------
    evidence : dict
        检测证据
    candidate_modules : set
        Level-1 确定的候选模块集合
    config : dict
        路由器配置
        
    Returns
    -------
    dict
        Level-2 路由结果
    """
    result = {
        'refined_modules': set(),
        'reason': {},
        'weights': {},
    }
    
    evidence_score_threshold = config.get('evidence_score_threshold', 0.5)
    refinement_sources = []
    
    # ===== 1. 根据突变类型路由 =====
    if evidence.get('jump_flag', False):
        jump_type = evidence.get('jump_type', '')
        jump_score = evidence.get('jump_max_score', 0.0)
        
        if jump_type in JUMP_TYPE_TO_MODULES and jump_score >= evidence_score_threshold:
            modules_from_jump = set(JUMP_TYPE_TO_MODULES[jump_type])
            # 取交集（仅保留在 Level-1 候选中的模块）
            refined_by_jump = modules_from_jump & candidate_modules
            if refined_by_jump:
                result['refined_modules'].update(refined_by_jump)
                refinement_sources.append('jump_type')
                result['reason']['jump_type'] = {
                    'type': jump_type,
                    'score': jump_score,
                    'modules': list(refined_by_jump),
                }
                # 为这些模块增加权重
                for mod in refined_by_jump:
                    result['weights'][mod] = result['weights'].get(mod, 0) + 0.3
    
    # ===== 2. 根据突变频段路由 =====
    jump_freq = evidence.get('jump_freq_hz', 0)
    if jump_freq > 0:
        band = _get_frequency_band(jump_freq)
        if band and band in BAND_TO_MODULES:
            modules_from_band = set(BAND_TO_MODULES[band])
            refined_by_band = modules_from_band & candidate_modules
            if refined_by_band:
                result['refined_modules'].update(refined_by_band)
                refinement_sources.append('jump_band')
                result['reason']['jump_band'] = {
                    'freq_hz': jump_freq,
                    'band': band,
                    'modules': list(refined_by_band),
                }
                for mod in refined_by_band:
                    result['weights'][mod] = result['weights'].get(mod, 0) + 0.2
    
    # ===== 3. 根据包络违例类型路由 =====
    env_violation_db = evidence.get('max_env_violation_db', 0)
    if abs(env_violation_db) > 0.5:  # 至少 0.5dB 的违例
        violation_type = evidence.get('env_violation_type', '')
        if not violation_type:
            # 根据符号推断
            violation_type = 'upper' if env_violation_db > 0 else 'lower'
        
        if violation_type in ENVELOPE_VIOLATION_TO_MODULES:
            modules_from_env = set(ENVELOPE_VIOLATION_TO_MODULES[violation_type])
            refined_by_env = modules_from_env & candidate_modules
            if refined_by_env:
                result['refined_modules'].update(refined_by_env)
                refinement_sources.append('envelope_violation')
                result['reason']['envelope_violation'] = {
                    'type': violation_type,
                    'max_db': env_violation_db,
                    'modules': list(refined_by_env),
                }
                for mod in refined_by_env:
                    result['weights'][mod] = result['weights'].get(mod, 0) + 0.2
    
    # ===== 4. 根据包络违例频段路由 =====
    env_band = evidence.get('env_violation_band_hz')
    if env_band and len(env_band) >= 2:
        center_freq = (env_band[0] + env_band[1]) / 2
        band = _get_frequency_band(center_freq)
        if band and band in BAND_TO_MODULES:
            modules_from_env_band = set(BAND_TO_MODULES[band])
            refined_by_env_band = modules_from_env_band & candidate_modules
            if refined_by_env_band:
                result['refined_modules'].update(refined_by_env_band)
                refinement_sources.append('env_band')
                result['reason']['env_violation_band'] = {
                    'band_hz': env_band,
                    'band_name': band,
                    'modules': list(refined_by_env_band),
                }
                for mod in refined_by_env_band:
                    result['weights'][mod] = result['weights'].get(mod, 0) + 0.15
    
    # 如果没有任何细化，返回空结果
    if not result['refined_modules']:
        result['reason']['status'] = 'no_evidence'
    else:
        result['reason']['status'] = 'refined'
        result['reason']['sources'] = refinement_sources
    
    return result


def _get_frequency_band(freq_hz: float) -> Optional[str]:
    """
    根据频率确定所属频段。
    
    Parameters
    ----------
    freq_hz : float
        频率值（Hz）
        
    Returns
    -------
    str or None
        频段名称
    """
    for band_name, (low, high) in FREQUENCY_BAND_BOUNDARIES.items():
        if low <= freq_hz <= high:
            return band_name
    return None


def _truncate_by_priority(
    candidate_modules: Set[str],
    weights: Dict[str, float],
    max_count: int
) -> Set[str]:
    """
    按优先级截断候选模块列表。
    
    Parameters
    ----------
    candidate_modules : set
        候选模块集合
    weights : dict
        模块权重
    max_count : int
        最大数量
        
    Returns
    -------
    set
        截断后的候选模块集合
    """
    # 为每个模块分配优先级（权重 + 默认值）
    priorities = {}
    for mod in candidate_modules:
        priorities[mod] = weights.get(mod, 0.1)
    
    # 始终保留的模块
    always_keep = {'电源模块', '未定义/其他'}
    
    # 排序并截断
    sorted_modules = sorted(
        [(m, p) for m, p in priorities.items() if m not in always_keep],
        key=lambda x: x[1],
        reverse=True
    )
    
    result = set(m for m, _ in sorted_modules[:max_count - len(always_keep)])
    result.update(always_keep & candidate_modules)
    
    return result


def _compute_prior_weights(
    candidate_indices: List[int],
    system_result: Dict[str, Any],
    evidence: Dict[str, Any],
    level2_weights: Dict[str, float]
) -> Dict[int, float]:
    """
    计算候选模块的先验权重。
    
    Parameters
    ----------
    candidate_indices : list
        候选模块索引
    system_result : dict
        系统级推理结果
    evidence : dict
        检测证据
    level2_weights : dict
        Level-2 路由产生的权重
        
    Returns
    -------
    dict
        模块索引 → 权重 的映射
    """
    weights = {}
    default_weight = 0.1
    
    for idx in candidate_indices:
        name = MODULE_LABELS[idx]
        
        # 基础权重
        weight = level2_weights.get(name, default_weight)
        
        # 根据系统级概率调整
        probs = system_result.get('probabilities', {})
        
        # 如果模块属于最可能的异常类型对应的组，增加权重
        top_class = system_result.get('top_class', '')
        if top_class in FAULT_TYPE_TO_GROUP:
            group_name = FAULT_TYPE_TO_GROUP[top_class]
            if name in MODULE_GROUPS.get(group_name, []):
                weight += 0.1
        
        weights[idx] = min(1.0, weight)  # 限制最大值为 1.0
    
    return weights


# ============================================================================
# 辅助函数：用于 BRB 模块级推理
# ============================================================================

def get_active_module_mask(
    routing_result: Dict[str, Any],
    total_modules: int = 21
) -> List[bool]:
    """
    根据路由结果生成模块激活掩码。
    
    Parameters
    ----------
    routing_result : dict
        route_modules_by_evidence 的输出
    total_modules : int
        总模块数量
        
    Returns
    -------
    list
        布尔列表，True 表示模块激活
    """
    mask = [False] * total_modules
    for idx in routing_result.get('candidate_modules', []):
        if 0 <= idx < total_modules:
            mask[idx] = True
    return mask


def apply_routing_to_module_probs(
    module_probs: Dict[str, float],
    routing_result: Dict[str, Any],
    deactivated_prob: float = 0.0
) -> Dict[str, float]:
    """
    将路由结果应用到模块概率分布。
    
    非候选模块的概率设为 deactivated_prob。
    
    Parameters
    ----------
    module_probs : dict
        原始模块概率分布
    routing_result : dict
        路由结果
    deactivated_prob : float
        非激活模块的概率值
        
    Returns
    -------
    dict
        调整后的模块概率分布
    """
    candidate_names = set(routing_result.get('candidate_module_names', []))
    adjusted_probs = {}
    
    for name, prob in module_probs.items():
        if name in candidate_names:
            adjusted_probs[name] = prob
        else:
            adjusted_probs[name] = deactivated_prob
    
    return adjusted_probs


def format_routing_explanation(routing_result: Dict[str, Any]) -> str:
    """
    格式化路由解释为可读文本。
    
    Parameters
    ----------
    routing_result : dict
        路由结果
        
    Returns
    -------
    str
        格式化的解释文本
    """
    reason = routing_result.get('routing_reason', {})
    lines = []
    
    # Level-1 解释
    level1 = reason.get('level1', {})
    if level1:
        selected = level1.get('selected_class', 'unknown')
        probs = level1.get('probabilities', {})
        lines.append(f"【Level-1 系统级路由】")
        lines.append(f"  - 检测到的主要异常类型: {selected}")
        lines.append(f"  - 概率分布: amp={probs.get('amp', 0):.2f}, freq={probs.get('freq', 0):.2f}, ref={probs.get('ref', 0):.2f}")
    
    # Level-2 解释
    level2 = reason.get('level2', {})
    if level2 and level2.get('status') == 'refined':
        lines.append(f"【Level-2 证据路由】")
        sources = level2.get('sources', [])
        lines.append(f"  - 使用的证据来源: {', '.join(sources)}")
        
        if 'jump_type' in level2:
            jt = level2['jump_type']
            lines.append(f"  - 突变类型: {jt.get('type')} (得分={jt.get('score', 0):.2f})")
            lines.append(f"    候选模块: {', '.join(jt.get('modules', []))}")
        
        if 'envelope_violation' in level2:
            ev = level2['envelope_violation']
            lines.append(f"  - 包络违例: {ev.get('type')} (幅度={ev.get('max_db', 0):.2f}dB)")
            lines.append(f"    候选模块: {', '.join(ev.get('modules', []))}")
    
    # 最终候选模块
    candidates = routing_result.get('candidate_module_names', [])
    lines.append(f"【最终候选模块】 共{len(candidates)}个")
    lines.append(f"  {', '.join(candidates)}")
    
    # 前放排除说明
    if reason.get('preamp_excluded'):
        lines.append(f"【说明】 前置放大器已排除（前放OFF模式）")
    
    return '\n'.join(lines)
