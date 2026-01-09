"""
真实频响数据读取模块

读取规则（强制）：
- freqHz = CSV 第 1 列（index=0）
- amp_dbm = CSV 倒数第二列（index=-2）
- 忽略其它列（包括中文/字符串列）
- 兼容科学计数法（1e+07）
- 清洗：drop NaN、按频率排序、频率去重（重复频率可取平均）
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


def read_real_response_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取真实频响CSV文件。
    
    规则：
    - freqHz = CSV 第 1 列（index=0）
    - amp_dbm = CSV 倒数第二列（index=-2）
    - 忽略其它列（包括中文/字符串列）
    - 兼容科学计数法（1e+07）
    - 清洗：drop NaN、按频率排序、频率去重（重复频率可取平均）
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        (freq, amp_dbm) - 频率数组(Hz)和幅度数组(dBm)，等长且已排序
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    # 尝试多种编码读取
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, header=None)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if df is None:
        raise ValueError(f"无法解析文件: {csv_path}")
    
    # 检测是否有表头（第一行是否为数据）
    # 如果第一行第一列无法转换为数字，则认为是表头
    try:
        float(df.iloc[0, 0])
        has_header = False
    except (ValueError, TypeError):
        has_header = True
        df = df.iloc[1:].reset_index(drop=True)
    
    # 确保至少有2列
    if df.shape[1] < 2:
        raise ValueError(f"文件列数不足2列: {csv_path}")
    
    # 提取频率（第0列）和幅度（倒数第二列）
    try:
        freq = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        amp = pd.to_numeric(df.iloc[:, -2], errors='coerce').values
    except Exception as e:
        raise ValueError(f"无法解析数值列: {e}")
    
    # Create DataFrame for data cleaning
    data = pd.DataFrame({'freq': freq, 'amp': amp})
    
    # Remove NaN values
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError(f"文件无有效数据: {csv_path}")
    
    # Sort by frequency
    data = data.sort_values('freq')
    
    # Handle duplicate frequencies by taking mean of amplitude values
    # Mean is used because duplicate readings at same frequency likely represent
    # measurement variations, and averaging provides a reasonable estimate
    data = data.groupby('freq', as_index=False).mean()
    
    freq = data['freq'].values.astype(np.float64)
    amp = data['amp'].values.astype(np.float64)
    
    return freq, amp


def read_simple_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取简单格式CSV（两列：freq_Hz, amplitude_dB）。
    
    这是原有的数据格式，向后兼容。
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        (freq, amp_dbm) - 频率数组(Hz)和幅度数组(dB)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.shape[1] >= 2:
        freq = df.iloc[:, 0].values
        amp = df.iloc[:, 1].values
        return freq.astype(np.float64), amp.astype(np.float64)
    else:
        raise ValueError(f"文件列数不足: {csv_path}")


def auto_read_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    自动检测CSV格式并读取。
    
    优先尝试简单格式（两列），如果失败则尝试真实频响格式。
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        (freq, amp_dbm) - 频率数组(Hz)和幅度数组(dBm)
    """
    csv_path = Path(csv_path)
    
    # 先检测文件格式
    try:
        df_peek = pd.read_csv(csv_path, nrows=2, encoding='utf-8')
        n_cols = df_peek.shape[1]
    except:
        try:
            df_peek = pd.read_csv(csv_path, nrows=2, encoding='gbk')
            n_cols = df_peek.shape[1]
        except:
            n_cols = 0
    
    # 如果是简单两列格式
    if n_cols == 2:
        try:
            return read_simple_csv(csv_path)
        except:
            pass
    
    # 尝试真实频响格式（多列）
    return read_real_response_csv(csv_path)


def load_all_real_responses(folder_path: str | Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    加载文件夹内所有CSV频响数据。
    
    自动适配真实频响格式和简单格式。
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        (freqs_list, amps_list, file_names) - 频率列表、幅度列表、文件名列表
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    freqs_list = []
    amps_list = []
    file_names = []
    
    csv_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == '.csv'])
    
    for csv_file in csv_files:
        try:
            freq, amp = auto_read_csv(csv_file)
            freqs_list.append(freq)
            amps_list.append(amp)
            file_names.append(csv_file.name)
        except Exception as e:
            warnings.warn(f"跳过文件 {csv_file.name}: {e}")
            continue
    
    if not freqs_list:
        raise FileNotFoundError(f"未在 {folder_path} 找到有效CSV数据")
    
    return freqs_list, amps_list, file_names
