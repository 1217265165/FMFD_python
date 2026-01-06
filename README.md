# 频谱分析仪故障诊断（知识驱动规则压缩 + 分层 BRB）

本仓库实现论文《基于知识驱动规则优化与分层推理的频谱分析仪故障诊断方法》的工程化流程，并为五类最新 BRB 变体预留了统一接口，便于一键对比实验。

## 代码结构速览
- `baseline/`：RRS 构建与包络提取。
- `features/`：系统级与模块级特征工程（X1~X5 等）。
- `features/feature_extraction.py` 也保留了旧版动态阈值/频段切换症状，使用
  `compute_dynamic_threshold_features` 可在需要时叠加包络越界率、切换台阶等特征，
  不会影响 BRB 体系的主特征。
- `BRB/`：系统级与模块级 BRB 推理核心。
- `methods/`：六种方法的统一封装（ours、brb_mu、dbrb、aifd、a_ibrb、brb_p）。
- `pipelines/`：流水线脚本（基线生成、仿真、检测、方法对比）。
- `comparison/`：可选的旧版对比实现与文档。

## 方法列表（对应 Related Work）
- **ours**：知识驱动规则压缩 + 分层 BRB，规则数≈45、参数≈38、特征维度=4。
- **brb_mu**：BRB-MU / Trustworthy Fault Diagnosis，突出多源不确定建模。
- **dbrb**：Deep BRB，多层软融合缓解组合爆炸。
- **aifd**：Adaptive Interpretable BRB，小样本场景自适应温度，保持可解释概率。
- **a_ibrb**：自动化 interval-BRB，支持区间输入弱化专家依赖。
- **brb_p**：BRB-P，加入概率表约束提升一致性。

## 对比方法与评价指标（工程实现说明）

为保证对比实验的公平性与可重复性，本仓库在统一的数据集、特征输入及评价指标下，对各对比方法进行了工程化实现。实现遵循各方法的核心思想与建模逻辑，而非逐公式复现原文算法。

1. **本文方法（ours）：知识驱动的特征–模块映射与分层 BRB 推理架构**
   - **特征-模块映射**：系统级特征刻画幅频特性与频率标度异常，模块级特征仅取对应功能模块的物理相关频段与信号特性。
   - **分层推理**：系统级 BRB 先输出异常类型概率，再作为先验激活模块级 BRB，仅对受影响的模块子集推理，结构性地压缩规则数。

2. **HCF（Zhang 2022）：分层认知框架**
   - **层次化建模**：系统层与模块层均使用完整特征集构建规则，保持“层层认知”与上下文传递。
   - **实现要点**：不做显式特征筛选，规则组合相对完整，推理效率在高维/强耦合场景下更依赖阈值收敛。

3. **BRB-P（Ming 2023）：概率表初始化 + 解释性约束**
   - **概率表初始化**：依据样本统计生成概率表初始化规则后件，减少对人工设定的依赖。
   - **解释性约束**：对规则权重与信念度变化设置约束，避免偏离先验语义，提升收敛稳定性与透明度。

4. **BRB-MU（Feng 2024）：多源不确定信息融合**
   - **不确定度建模**：为不同来源/可靠性的特征分配不确定度系数，在规则激活与后件融合时做加权抑制。
   - **优势**：在噪声大、数据质量不稳定的环境中保持诊断稳健性，但计算复杂度更高。

5. **DBRB（Zhao 2024）：深层分层 BRB 缓解规则爆炸**
   - **特征重要性分组**：先对候选特征做重要性评估（此处用启发式代替 XGBoost 排序），再拆分为多个子集。
   - **多层推理**：为每个子集构建浅层/深层 BRB 子模型，层间通过中间变量逐级融合，显著降低单层规则规模。

6. **A-IBRB（Wan 2025）：基于区间参考值的自动化 BRB**
   - **区间化参考值**：对连续特征按数据分布自动区间化，自动生成规则参考等级，弱化专家依赖。
   - **自动化建模**：通过区间粗分段构建规则库，灵活但性能受区间划分质量影响。

### 实验评价指标
- **规则库规模**：统计各方法所需规则数量，衡量规则爆炸控制能力与推理速度。
- **参数总数与特征维度**：反映模型结构复杂度与优化难度，越小越高效。
- **诊断准确率**：系统级与模块级故障识别能力的核心指标。
- **推理时间**：单样本推理耗时，衡量实时性。
- **小样本适应性**：训练样本受限时的稳定性与准确率。

## 一键复现实验
```bash
# 1) 生成或准备 baseline（如有需要）
python -m pipelines.run_baseline

# 2) 仿真故障数据（含系统级/模块级标签 + 原始曲线 CSV）
python -m pipelines.simulate.run_simulation_brb

# 3) 运行六种方法的统一对比
python -m pipelines.compare_methods
```
输出位于 `Output/comparison_results/`：
- `comparison_table.csv`：规模与性能总览。
- `performance_table.csv`：各异常类型准确率。
- `comparison_plot.png`：准确率 vs 规则数量散点图。
- `comparison_summary.txt`：文字化总结（含模块级 Top-K 指标）。
  （如未安装 matplotlib 会提示跳过生成，可按需 `pip install matplotlib` 后重跑脚本。）

仿真产物位于 `Output/sim_spectrum/`：
- `raw_curves/*.csv`：每个样本一份频率/幅度文件，可直接喂给其他对比方法或 `brb_diagnosis_cli.py`。
- `raw_manifest.csv`：raw_curves 内文件清单（含标签与模块）。
- `features_brb.csv`：系统/模块 BRB 直接可用的特征与概率。
- `simulated_features.csv`：X1~X5 + 旧动态阈值症状 + 标签，便于自定义特征选择。
- `simulated_curves.csv`/`simulated_curves.npz`：完整的频率-幅度矩阵，可按方法重新提取特征。
- `labels.json`：系统/模块标签，`pipelines.compare_methods` 会自动读取。

## 系统级 BRB 接口设计
- 入口：`BRB/system_brb.py` 中的 `system_level_infer`。
- 特征：X1~X5 分别表示整体幅度偏移、带内平坦度、高频段衰减斜率、频率标度非线性、幅度缩放一致性。
- 机制：三角隶属度 → 规则压缩激活 → softmax(α=2.0) → 正常识别（整体异常度阈值 + 最大概率阈值）。
- 输出：概率分布、最大概率、是否判定为正常、不确定度、overall_score，便于与模块层推理联动。

## 与原有代码如何衔接
- **保留旧特征**：`features/feature_extraction.py` 中的 `compute_dynamic_threshold_features` 仍可生成动态包络越界、非频段切换跳变等遗留特征，不会覆盖 X1~X5；在组装特征时将其字典并入即可。
- **旧流水线兼容**：`pipelines/detect.py`、`pipelines/simulate/` 里的逻辑未被移除，直接复用即可；新增的对比方法接口放在 `methods/` 下，不影响原始 baseline。
- **BRB 核心未改名**：系统层入口 `system_level_infer`、模块层入口 `module_level_infer` 均保持原名，可直接替换之前的 import。
- **数据与输出路径不变**：输入数据仍在 `data/`、`to_detect/`，运行结果写入 `Output/`，不会破坏你已有的数据组织；仿真生成的 `raw_curves/*.csv` 可直接复制到 `to_detect/` 后用 `brb_diagnosis_cli.py` 复现文件输入流程。

## 在 PyCharm/终端运行什么命令
在 PyCharm 的 Terminal（或系统终端）执行以下命令即可（确保当前目录是仓库根目录）。

1. **生成/更新 baseline**（如未生成）：
   ```bash
   python -m pipelines.run_baseline
   ```
2. **仿真并跑完整 BRB 流水线**（含系统 + 模块诊断）：
   ```bash
   python -m pipelines.simulate.run_simulation_brb
   ```
3. **单次检测真实/仿真数据**（复用你旧的 `detect.py` 调用方式）：
   ```bash
   python -m pipelines.detect
   ```
4. **对比六种方法**（可选）：
   ```bash
   python -m pipelines.compare_methods
   ```
   - 默认会从 `Output/sim_spectrum/features_brb.csv + labels.json` 读取特征与标签；如果缺失，就退回内部的合成特征。每个方法依然保留自身的特征重标定与决策阈值，因此诊断结果不会再“一刀切”。
   - 如果概率分布整体偏平（阈值以下），脚本会用基于偏置/频偏/平坦度/幅度一致性的兜底启发式给出预测，避免所有方法同时输出低准确率的“正常”结果，便于快速对比差异。
   - 对比脚本使用各方法各自的推理配置：BRB-MU 会随不确定度放大频率/参考特征，DBRB 采用逐层收敛并对高频/参考指标加权，A-IBRB 依据区间宽度调整平坦度/参考权重，BRB-P 使用概率表约束并偏向幅度+参考类。

## 当前整体架构（保留旧代码的同时扩展）
- baseline → features → BRB（system + module）/methods → pipelines：遵循原有分层。
- `methods/` 子目录仅新增封装，不会影响你已有的类与函数；需要时可在 `pipelines/compare_methods.py` 注册或移除某个方法。
- `BRB/` 仍只依赖特征字典，旧调用代码可直接传入新旧特征的合并结果；未使用的特征会被安全忽略。
- 输出格式保持“系统级概率 + 模块级 21 维概率 + 不确定度”的字典/CSV 结构，与你之前的后处理脚本兼容。
