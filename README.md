# 基于知识驱动规则优化与分层推理的频谱分析仪故障诊断方法

本项目实现了一种基于知识驱动的分层推理架构用于频谱分析仪故障诊断。方法结合BRB（信念规则库）和特征-模块映射策略，通过优化规则库和推理流程实现高准确率诊断，并实现规则压缩和特征分流。

---

## 目录

1. [项目简介](#1-项目简介)
2. [数据来源说明](#2-数据来源说明)
3. [整体流程](#3-整体流程)
4. [特征体系](#4-特征体系)
5. [BRB推理与分层注入机制](#5-brb推理与分层注入机制)
6. [真实诊断主链路](#6-真实诊断主链路)
7. [CLI使用方法](#7-cli使用方法)
8. [对比实验入口](#8-对比实验入口)
9. [Baseline/Ours调参流程](#9-baselineours调参流程)
10. [输出文件说明](#10-输出文件说明)
11. [开发规范](#11-开发规范)

---

## 1. 项目简介

### 一句话简介
频谱分析仪故障诊断系统：基于知识引导特征提取（弱依赖样本）+ 分层BRB推理（系统级三类异常 → 模块级21模块）。

### 方法核心思想
- **知识驱动特征提取**：从频响曲线提取系统级/模块级特征，弱依赖大量样本
- **分层BRB推理**：系统级三分支异常（幅度/频率/参考电平）→ 激活模块层 → 21模块概率
- **规则压缩**：通过物理链路知识，仅激活相关模块子集，规则数≈45（vs传统130+）

### 项目结构
```
project/
├── baseline/                # RRS构建 + 动态包络提取
│   ├── baseline.py         # 基线计算核心
│   ├── config.py           # 频段/输出路径配置
│   └── viz.py              # 可视化工具
├── features/                # 特征提取
│   ├── extract.py          # 系统特征提取
│   ├── feature_extraction.py # 完整特征工程
│   └── feature_router.py   # 特征分流规则
├── BRB/                     # BRB推理模块
│   ├── system_brb.py       # 系统级推理（三分支）
│   ├── system_brb_amp.py   # 幅度异常子BRB
│   ├── system_brb_freq.py  # 频率异常子BRB
│   ├── system_brb_ref.py   # 参考电平异常子BRB
│   ├── aggregator.py       # 结果聚合器
│   ├── module_brb.py       # 模块级推理（21模块）
│   └── utils.py            # BRB工具函数
├── methods/                 # 对比方法封装
│   ├── ours/               # 本文方法
│   ├── brb_mu/, dbrb/, aifd/, a_ibrb/, brb_p/
├── pipelines/               # 实验脚本
│   ├── run_baseline.py     # 基线构建
│   ├── detect.py           # 故障检测
│   ├── optimize_brb.py     # BRB参数优化
│   ├── visualize_results.py # 结果可视化
│   ├── compare_methods.py  # 对比实验
│   └── simulate/           # 仿真数据生成
├── tools/
│   └── brb_report.py       # 诊断报告生成
├── config/                  # 配置文件
│   └── feature_config.py   # 特征参数配置
├── Output/                  # 运行输出
│   └── runs/<run_name>/    # 每次运行的独立目录
├── normal_response_data/    # 正常频响数据
├── to_detect/               # 待检测数据
├── brb_diagnosis_cli.py    # CLI入口
├── brb_rules.yaml          # BRB规则配置
└── thresholds.json         # 检测阈值
```

---

## 2. 数据来源说明

### 数据采集方式
采用**信号发生器逐频点步进注入**方式：
1. 信号发生器按预设频率列表逐点输出
2. 频谱分析仪接收并测量各频点的峰峰值
3. 汇总形成完整的频响曲线

### 数据格式
CSV文件，包含两列：
```csv
frequency,amplitude_dB
1000000,0.05
2000000,0.03
...
```

### 数据类型
- **正常数据**：实测数据，放置于 `normal_response_data/`
- **故障数据**：仿真生成或实测故障数据，放置于 `to_detect/`

### 频段划分
```python
BAND_RANGES = {
    "低频段": (0, 3e9),
    "中频段": (3e9, 8e9),
    "高频段": (8e9, 26.5e9),
}
```

---

## 3. 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        整体诊断流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. baseline/RRS/动态包络构建                                     │
│     └─ 正常数据 → RRS曲线 → 上下包络 → 切换点特征                  │
│                      ↓                                           │
│  2. 系统级特征提取 (X1-X22)                                       │
│     └─ 待检数据 vs 基线 → 22维特征向量                            │
│                      ↓                                           │
│  3. 系统级三分支异常推理                                          │
│     └─ 特征分流 → amp/freq/ref子BRB → softmax聚合                 │
│                      ↓                                           │
│  4. 激活模块层推理                                                │
│     └─ 根据系统级结果 → 仅激活相关模块组 → 21模块概率              │
│                      ↓                                           │
│  5. 输出诊断报告                                                  │
│     └─ 系统级分类 + 模块级Top-K + 证据字段 + 可视化                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 特征体系

### 4.1 系统级特征 (X1-X10)
| 特征 | 物理含义 | 用途 |
|------|----------|------|
| X1 | 整体幅度偏移 | 幅度/参考电平异常识别 |
| X2 | 带内平坦度指标 | 幅度链路评估 |
| X3 | 高频段衰减斜率 | 参考电平异常识别 |
| X4 | 频率标度非线性度 | 频率异常识别 |
| X5 | 幅度缩放一致性 | 幅度/参考电平异常识别 |
| X6 | 纹波 | 模块级症状特征 |
| X7 | 增益非线性 | 模块级症状特征 |
| X8 | 本振泄漏 | 频率异常识别 |
| X9 | 调谐线性度残差 | 频率异常识别 |
| X10 | 不同频段幅度一致性 | 幅度/参考电平异常识别 |

### 4.2 包络/残差特征 (X11-X15)
| 特征 | 物理含义 |
|------|----------|
| X11 | 包络超出率 (envelope overrun rate) |
| X12 | 最大包络违规幅度 |
| X13 | 包络违规能量 |
| X14 | 低频段残差均值 |
| X15 | 高频段残差标准差 |

### 4.3 频率对齐特征 (X16-X18)
| 特征 | 物理含义 |
|------|----------|
| X16 | 互相关滞后/频移 (corr_shift_bins) |
| X17 | 频轴缩放因子 (warp_scale) |
| X18 | 频轴平移因子 (warp_bias) |

### 4.4 幅度细粒度特征 (X19-X22)
| 特征 | 物理含义 |
|------|----------|
| X19 | 低频段斜率 (slope_low) |
| X20 | 去趋势残差峰度 (kurtosis_detrended) |
| X21 | 残差峰值数量 (peak_count_residual) |
| X22 | 残差主频能量占比 (ripple_dom_freq_energy) |

### 4.5 特征分流规则
```python
SYSTEM_BRANCH_FEATURES = {
    'amp': ['X1', 'X2', 'X5', 'X6', 'X7', 'X10', 'X11', 'X12', 'X13', 'X19', 'X20', 'X21', 'X22'],
    'freq': ['X4', 'X8', 'X9', 'X14', 'X15', 'X16', 'X17', 'X18'],
    'ref': ['X1', 'X3', 'X5', 'X10', 'X11', 'X12', 'X13'],
}
```

---

## 5. BRB推理与分层注入机制

### 5.1 系统级三分支推理
```
                    ┌─────────────────┐
                    │   所有特征      │
                    │   (X1-X22)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ 幅度子BRB   │  │ 频率子BRB   │  │ 参考子BRB   │
    │ (amp_features)│ │(freq_features)│ │(ref_features)│
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            ↓
                   ┌────────────────┐
                   │    聚合器      │
                   │ (softmax α=2.0)│
                   └────────┬───────┘
                            ↓
              ┌─────────────────────────┐
              │ 系统级分类概率          │
              │ {正常, 幅度, 频率, 参考}│
              └─────────────────────────┘
```

### 5.2 模块级分层注入
根据系统级推理结果，仅激活相关模块组：

| 系统级异常 | 激活模块组 |
|------------|-----------|
| 幅度失准 | 衰减器、中频放大器、数字放大器、ADC、数字RBW、数字检波器、VBW滤波器 |
| 频率失准 | 时钟振荡器、时钟合成网络、本振源、本振混频组件、YTF滤波器、混频器 |
| 参考电平失准 | 校准源、存储器、校准信号开关、衰减器 |

### 5.3 与DBRB的对应关系
本方法的"分层注入"与DBRB的"逐级注入"在工程结构上对应：
- DBRB：层间传递中间激活值
- 本方法：系统级结果作为先验，激活模块级子集

### 5.4 21个模块列表
1. 衰减器
2. 低频段前置低通滤波器
3. 低频段第一混频器 
4. 高频段YTF滤波器 
5. 高频段混频器 
6. 时钟振荡器 
7. 时钟合成与同步网络 
8. 本振源（谐波发生器） 
9. 本振混频组件 
10. 校准源 
11. 存储器 
12. 校准信号开关 
13. 中频放大器 
14. ADC 
15. 数字RBW 
16. 数字放大器 
17. 数字检波器 
18. VBW滤波器 
19. 电源模块

---

## 6. 真实诊断主链路

### 6.1 主链路流程
```
detect → optimize_brb(可选) → visualize_results → brb_report
```

### 6.2 输出目录规范
所有诊断输出落在统一的run目录：
```
Output/runs/<timestamp_or_run_name>/
├── artifacts/          # 基线、特征、归一化统计量、优化参数快照
│   ├── features_*.csv
│   ├── baseline_snapshot.npz
│   └── optimized_params.json
├── tables/             # 检测结果表、模块概率表、汇总表
│   ├── detection_results.csv
│   ├── module_probabilities.csv
│   └── summary.csv
├── plots/              # 可视化图表
│   ├── envelope_violation.png
│   ├── system_proba.png
│   └── module_top_k.png
├── reports/            # 最终报告
│   └── brb_report.md
└── logs/               # 运行日志
    └── run.log
```

### 6.3 各步骤输入输出

#### detect.py
- **输入**: `to_detect/*.csv` + 基线 + 阈值 + 规则
- **输出**: 
  - `tables/detection_results.csv`
  - `artifacts/features_*.csv`
  - 证据字段（包络违例、突变位置、切换点偏移等）

#### optimize_brb.py
- **输入**: detect产物 + 标签/约束（如有）
- **输出**:
  - `artifacts/optimized_params.json`
  - `tables/optimization_summary.csv`

#### visualize_results.py
- **输入**: detect + optimize输出
- **输出**: `plots/*.png`

#### brb_report.py
- **输入**: 以上所有产物
- **输出**: `reports/brb_report.md`

### 6.4 证据字段说明
detect输出包含以下解释型字段：
- `envelope_violation`: 是否触发包络违例
- `violation_max_db`: 最大违例幅度(dB)
- `violation_energy`: 违例能量
- `switching_offset_bins`: 切换点偏移量
- `non_switching_mutation`: 非切换点突变检测
- `mutation_freq_hz`: 突变位置(Hz)
- `top_k_rules`: 触发的规则Top-K

---

## 7. CLI使用方法

### 7.1 一键模式
```bash
python brb_diagnosis_cli.py \
    --input to_detect/sample.csv \
    --output result.json \
    --baseline Output/baseline_artifacts.npz \
    --mode sub_brb \
    --run_name my_diagnosis
```

### 7.2 分步模式
```bash
# 步骤1: 仅检测
python pipelines/detect.py --input to_detect/ --out_dir Output/runs/run_001/

# 步骤2: 优化（可选）
python pipelines/optimize_brb.py --run_dir Output/runs/run_001/

# 步骤3: 可视化
python pipelines/visualize_results.py --run_dir Output/runs/run_001/

# 步骤4: 生成报告
python tools/brb_report.py --run_dir Output/runs/run_001/
```

### 7.3 CLI参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入CSV文件路径 | 必需 |
| `--output, -o` | 输出JSON文件路径 | 必需 |
| `--baseline, -b` | 基线数据目录 | 内置路径 |
| `--mode, -m` | BRB推理模式(er/sub_brb) | sub_brb |
| `--run_name` | 运行名称 | 时间戳 |
| `--do_optimize` | 是否执行优化 | False |
| `--verbose, -v` | 显示详细输出 | False |

### 7.4 常见报错处理
| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'numpy'` | 缺少依赖 | `pip install numpy pandas scipy scikit-learn` |
| `FileNotFoundError: baseline_artifacts.npz` | 未生成基线 | 先运行 `python pipelines/run_baseline.py` |
| `FileNotFoundError: to_detect/*.csv` | 无待检数据 | 将CSV文件放入 `to_detect/` 目录 |

---

## 8. 对比实验入口

> **注意**: 对比实验仅用于论文验证，不是现场诊断主链路。

### 8.1 运行对比实验
```bash
# 1. 生成仿真数据
python pipelines/simulate/run_simulation_brb.py

# 2. 运行对比
python pipelines/compare_methods.py

# 3. 查看结果
cat Output/comparison_table.csv
```

### 8.2 对比方法
**（1）本文方法：基于知识驱动的特征–模块映射与分层 BRB 推理框架

本文方法以频谱分析仪的物理结构和信号传递机理为先验约束，构建“系统级–模块级”两层 BRB 推理架构。在建模过程中，首先依据设备功能划分与信号链路关系，对频率响应数据中具有明确物理含义的特征进行筛选，并建立特征与功能模块之间的知识驱动映射关系，从而避免无关特征参与规则组合。

在推理阶段，模型先在系统层面对异常类型进行判别，用以刻画幅度失准、频率失准与参考电平失准等全局性异常；随后，系统级推理结果作为先验信息引导模块级 BRB 的激活，仅对物理上可能受影响的模块子集执行推理。该分层推理机制有效削减了规则组合空间，显著缓解了传统 BRB 中随特征数增加而产生的规则爆炸问题。

得益于上述结构设计，本文方法在规则数量和参数规模显著降低的同时，仍保持了较高的诊断精度与良好的可解释性，尤其适用于真实故障样本稀缺、模块数量较多的精密仪器诊断场景。

（2）HCF（Hierarchical Cognition Framework, Zhang 2022）

HCF 方法基于认知建模思想，将复杂系统的故障诊断过程抽象为多层次的认知推理过程。该方法通过构建分层认知结构，将系统行为分析、状态理解与故障决策划分为不同层级，并在每一层内引入基于规则的推理机制，以增强诊断过程的逻辑一致性与可解释性。

相比单层规则模型，HCF 在结构上引入了层次划分，有助于缓解部分复杂度问题。然而，该方法在规则构建阶段仍依赖较完整的规则组合，当特征维度和模块数量较高时，规则规模与推理复杂度仍然较大，且其层级划分更多基于认知逻辑而非设备物理结构。

（3）BRB-P（Belief Rule Base with Probability Table, Ming 2023）

BRB-P 是对传统 BRB 推理框架的重要改进，其核心思想是在规则初始化阶段引入概率表结构，并在参数优化过程中加入解释性约束。具体而言，该方法利用概率表对规则后件信念度进行初始化，使模型在训练初期即具备合理的先验分布，从而提升收敛稳定性。

此外，BRB-P 在优化过程中通过显式约束规则参数的变化范围，防止模型在追求高精度的同时破坏原有规则语义，有效缓解了传统 BRB 在参数学习阶段可能出现的解释性退化问题。然而，由于其仍采用相对完整的规则组合方式，当输入特征维度较高时，规则规模和参数数量依然较大。

（4）BRB-MU（Trustworthy BRB with Multisource Uncertainty, Feng 2024）

BRB-MU 针对复杂系统中多源信息不一致和不确定性显著的问题，引入了多源不确定信息融合机制。该方法在 BRB 推理过程中为不同信息源分配不确定度权重，使模型能够根据数据可信度自适应调整推理结果，从而提高诊断稳定性和可信性。

该方法在噪声较大或数据质量波动明显的场景中表现出较强鲁棒性，但多源信息融合和不确定度建模也带来了额外的计算负担，在规则规模较大时可能影响推理效率。

（5）DBRB（Deep Belief Rule Base, Zhao 2024）

DBRB 通过引入深层结构对传统 BRB 进行扩展，其核心思想是将高维规则组合问题分解为多层 BRB 子模型，逐层完成推理与信息传递。每一层仅处理特定特征子集或中间状态，从而在结构层面有效缓解规则数量随特征维度指数增长的问题。

DBRB 在降低单层规则规模方面具有明显优势，但其多层推理结构增加了模型整体复杂度，对参数配置和推理效率提出了更高要求。

（6）A-IBRB（Automated Interval BRB, Wan 2025）

A-IBRB 通过区间化参考值与自动化规则生成策略，减少对专家知识的依赖。该方法首先对连续特征进行区间划分，并基于数据分布自动构建区间规则结构，从而实现 BRB 规则库的自动生成。

该方法在专家经验不足或系统结构未知的场景中具有一定优势，但由于区间划分主要依赖数据分布，其规则语义与设备物理机理的对应关系相对较弱，在工程可追溯性方面存在一定局限。**

### 8.3 评估指标
- **系统级**: 准确率、宏平均F1、混淆矩阵
- **模块级**: Top-1/Top-3准确率
- **推理效率**: 单样本推理时间(ms)
- **模型复杂度**: 规则数量 + 参数数量

为从结构复杂度、诊断性能与工程适用性等多个维度对不同方法进行综合评估，本文选取以下评价指标：

规则库规模：统计各方法在相同任务下所需的规则数量，用以衡量其对规则爆炸问题的控制能力；

参数总数与特征维度：反映模型的结构复杂度与参数优化难度；

诊断准确率：包括系统级与模块级诊断准确率，用以评价模型的判别能力；

推理时间：衡量单样本推理所需的计算时间，反映模型的实时性；

小样本适应性：在训练样本数量受限条件下模型性能的稳定性，用以评估其在实际工程场景中的适用性。

## 9. Baseline/Ours调参流程

以下命令均在仓库根目录执行，确保可复现基线与系统级诊断的评估流程：

1) 生成 baseline（RRS + quantile envelope）：  
```bash
python pipelines/run_babeline.py
```
输出：`Output/baseline_artifacts.npz`, `Output/baseline_meta.json`, `Output/baseline_quality.json`, `Output/normal_feature_stats.csv`。

2) 运行诊断/评估（示例）：  
```bash
python pipelines/brb_diagnosis_cli.py --include_baseline
```

3) 调参（ours 方法）：  
```bash
python tools/tune_ours.py --features Output/sim_spectrum/features_brb.csv --labels Output/sim_spectrum/labels.json
```
输出：`Output/ours_best_config.json`（包含门控权重、温度等参数与验证指标）。

4) 对比评估（ours vs brb_mu）：  
```bash
python tools/eval_compare.py --dataset Output/sim_spectrum --config Output/ours_best_config.json
```
输出：两种方法的准确率、每类 precision/recall、混淆矩阵。

---

## 10. 输出文件说明

### 9.1 基线构建输出 (`run_baseline.py`)
| 文件 | 说明 |
|------|------|
| `baseline_artifacts.npz` | 基线数据(frequency, rrs, upper, lower) |
| `baseline_meta.json` | 元信息(band_ranges, k_list) |
| `switching_features.csv` | 切换点特征 |
| `normal_feature_stats.csv` | 正常特征统计量 |
| `baseline_switching.png` | 基线可视化 |

### 9.2 检测输出 (`detect.py`)
| 文件 | 说明 |
|------|------|
| `detection_results.csv` | 检测结果(特征、概率、告警标志) |
| `feature_usage_debug.json` | 特征使用调试信息 |

### 9.3 对比实验输出 (`compare_methods.py`)
| 文件 | 说明 |
|------|------|
| `comparison_table.csv` | 方法vs指标对比表 |
| `performance_table.csv` | 详细性能表 |
| `confusion_system.png` | 系统级混淆矩阵 |
| `confusion_module.png` | 模块级混淆矩阵 |

### 9.4 报告输出 (`brb_report.py`)
| 文件 | 说明 |
|------|------|
| `brb_report.md` | 诊断报告(Markdown格式) |
| `system_proba.png` | 系统级概率条形图 |
| `module_top_k.png` | 模块级Top-K概率 |

---

## 11. 开发规范

### 10.1 环境要求
- Python 3.8+
- 依赖包: numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml

### 10.2 安装依赖
```bash
pip install -r requirements.txt
```

### 10.3 新增特征
1. 在 `features/feature_extraction.py` 中实现提取函数
2. 在 `features/feature_router.py` 中更新分流规则
3. 在 `config/feature_config.py` 中添加归一化参数
4. 更新 `BRB/system_brb.py` 中的权重

### 10.4 新增模块
1. 在 `brb_rules.yaml` 中添加模块定义
2. 更新 `BRB/module_brb.py` 中的MODULE_LABELS
3. 更新 `features/feature_extraction.py` 中的MODULES_ORDER

### 10.5 新增报告字段
1. 在 `pipelines/detect.py` 中输出新字段
2. 在 `tools/brb_report.py` 中解析并展示

### 10.6 代码规范
- 使用类型注解
- 每个函数有docstring
- 配置参数集中在 `config/` 目录

---

## 快速开始

### 1. 生成基线
```bash
python pipelines/run_baseline.py
```

### 2. 运行检测
```bash
# 将待检CSV放入to_detect/目录
python pipelines/detect.py
```

### 3. 一键诊断（CLI）
```bash
python brb_diagnosis_cli.py -i to_detect/sample.csv -o result.json
```

### 4. 查看结果
```bash
cat Output/detection_results.csv
```

---

## 许可证

MIT License

---

## 联系方式

如有问题，请提交Issue。
