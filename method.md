# FMFD 方法说明（单频段）

本文档概述 FMFD 单频段流程、RRS/包络、仿真与特征提取，并按方法逐一说明核心思想、推理流程、补偿/校准机制与主要参数来源。

## 1. RRS 基准与包络提取

### 1.1 正常曲线对齐与 RRS
- `baseline/baseline.py` 会加载 `normal_response_data` 中的 CSV 曲线，并插值到统一频率轴。每条曲线会被线性插值到一致频点后叠成矩阵，作为基线样本集。该对齐过程保证了后续 RRS 和包络在相同频率网格上构建。\n【F:baseline/baseline.py†L1-L82】
- RRS 由**点位中位数**构成，并支持“极轻微平滑但默认关闭”，以避免“平滑导致形状漂移”。这一点由 `RRS_SMOOTH_ENABLED = False` 默认关闭体现。\n【F:baseline/baseline.py†L18-L24】
- 每条曲线会先估计“全局偏移量（median residual）”，再把曲线通过该偏移量对齐，用于更稳定地估计包络残差分布，避免全局偏移造成包络过宽。\n【F:baseline/baseline.py†L84-L102】

### 1.2 包络（Envelope）与覆盖率约束
- 包络是基于“残差分位数”与平滑约束构建：残差网格分位数（覆盖率网格）+ 平滑 + 截断与扩展策略，确保整体覆盖率与滑窗覆盖率满足目标。\n【F:baseline/baseline.py†L26-L44】【F:baseline/baseline.py†L118-L168】
- 覆盖率统计包含“整体覆盖率、单曲线覆盖率、点位覆盖率”等指标，用于验证包络质量。\n【F:baseline/baseline.py†L118-L168】

### 1.3 产物输出
- 基线管线 `pipelines/run_babeline.py` 负责加载数据、计算 RRS/包络、输出元数据与可视化图；并统一输出 `Output/baseline_artifacts.npz`、`baseline_meta.json`、`baseline_quality.json` 等文件。\n【F:pipelines/run_babeline.py†L1-L118】

## 2. 故障仿真（Simulation）

- 仿真入口为 `pipelines/simulate/run_simulation_brb.py`，默认无参数运行时会读取基线产物并生成 400 个平衡样本（四类各 100），同时生成 `raw_curves`、`features_brb.csv`、`labels.json` 等用于对比评估的文件。\n【F:pipelines/simulate/run_simulation_brb.py†L1-L118】【F:pipelines/simulate/run_simulation_brb.py†L360-L509】
- 该脚本使用基线曲线与包络进行故障注入，并输出系统级标签（幅度/频率/参考电平/正常）以及对应特征。\n【F:pipelines/simulate/run_simulation_brb.py†L360-L509】

## 3. 特征提取（System + Module）

### 3.1 系统级特征（X1–X34）
- 系统级特征由 `features/feature_extraction.py` 的 `extract_system_features` 生成，覆盖基础偏差、纹波、斜率、包络违规、频率扭曲与高/低频能量等信息；X1–X5 为基础特征，X11–X15 为包络/残差相关，X16–X18 为频率对齐/形变，X19–X22 为幅度链路细粒度特征，X23–X34 为增强频率/参考电平特征与谱结构统计。\n【F:features/feature_extraction.py†L292-L758】

### 3.2 模块级特征与模块排序
- 模块级特征（模块推理输入）由 `extract_module_features` 使用系统特征与传统特征组合生成，保证系统层与模块层特征一致。\n【F:features/feature_extraction.py†L869-L934】
- 模块顺序在 `MODULES_ORDER` 中统一定义（已移除“未定义/其他”），并用于 BRB 映射矩阵与模块列输出。\n【F:features/feature_extraction.py†L49-L62】

## 4. 方法总览（System-Level）

评估入口为 `pipelines/compare_methods.py`，统一使用相同的特征列表、标签映射、训练/验证/测试划分，并写入审计信息与复现摘要。\n【F:pipelines/compare_methods.py†L53-L1150】

### 4.1 Ours（层次 BRB + 规则压缩 + 正常锚点）

**核心思想**\n- 系统层：以 BRB 规则为核心，先做“正常锚点”判别，再进行幅度/频率/参考电平的软门控融合。系统层支持子 BRB 模式（sub_brb）进行多分支推理并输出概率。\n【F:BRB/normal_anchor.py†L1-L239】【F:BRB/aggregator.py†L737-L892】\n- 模块层：使用系统层结果作为先验，只激活与异常类型相关的模块组，从而减少规则数量并保持物理可解释性。\n【F:BRB/module_brb.py†L21-L241】

**特征与推理流程**\n1. 输入特征为系统级 X1–X22（主路径），以及模块层特征流（用作模块推理）。\n【F:methods/ours_adapter.py†L27-L114】【F:features/feature_extraction.py†L292-L934】\n2. 系统层通过 `system_level_infer` 输出系统级概率（正常/幅度/频率/参考电平）。\n【F:methods/ours_adapter.py†L159-L188】\n3. 模块层调用 `module_level_infer_with_activation`，仅激活相关模块组，输出模块概率并归一化。\n【F:methods/ours_adapter.py†L190-L219】【F:BRB/module_brb.py†L241-L360】

**补偿与校准**\n- 通过 `pipelines/calibrate_ours.py` 的网格搜索，得到 `best_params.json`/`calibration.json`，并由 `OursAdapter` 加载以覆盖 `SystemBRBConfig` 的 `alpha`、阈值与权重等参数。\n【F:pipelines/calibrate_ours.py†L682-L780】【F:methods/ours_adapter.py†L12-L92】

**主要参数/规则来源**\n- 系统层规则与软门控逻辑在 `BRB/aggregator.py` 与 `BRB/normal_anchor.py`，关键阈值如 `T_low/T_high`、`pmax_threshold`、`margin_threshold` 等均在该层实现并可通过校准覆盖。\n【F:BRB/aggregator.py†L777-L892】【F:BRB/normal_anchor.py†L39-L287】\n- 模块层规则与模块组定义在 `BRB/module_brb.py` 中，依据异常类型进行规则压缩。\n【F:BRB/module_brb.py†L21-L360】

### 4.2 BRB-MU（多源不确定性融合）

**核心思想**\n- 把特征按“幅度/频率/噪声/切换”等语义分组，针对每个来源训练一个简化统计模型；估计 SNR/SVD 不确定度并转为来源权重，实现多源融合。\n【F:methods/brb_mu_adapter.py†L9-L178】

**实现流程**\n1. 依据特征名的语义（amp/freq/noise/switch）分配到来源组。\n【F:methods/brb_mu_adapter.py†L98-L155】\n2. 每个来源拟合“类条件均值/方差”的高斯模型，得到来源内预测。\n【F:methods/brb_mu_adapter.py†L169-L222】\n3. 估计来源不确定度，权重归一化后融合来源概率。\n【F:methods/brb_mu_adapter.py†L56-L116】【F:methods/brb_mu_adapter.py†L124-L148】

### 4.3 DBRB（深层 BRB 级联 / 分级注入）

**核心思想**\n- 先用 XGBoost/GradientBoosting 对特征重要性排序，再按重要性划分三层特征池；每层输出的概率向量作为“隐变量”注入下一层，形成分级注入式 BRB。\\n【F:methods/dbrb_adapter.py†L10-L141】

**实现流程（按源码逻辑）**\n1. **特征池与重要性排序**：使用 XGBoost（或回退到 GradientBoosting）训练初始分类器，读取 `feature_importances_` 排序；无依赖时回退到方差排序。\\n【F:methods/dbrb_adapter.py†L29-L63】\n2. **分层切分**：排序后划分 Layer1（最重要特征）、Layer2（次重要特征）、Layer3（剩余特征）。\\n【F:methods/dbrb_adapter.py†L66-L77】\n3. **分级注入**：\n   - Layer1：只使用 `layer1_features` 训练并输出 `z1`。\\n【F:methods/dbrb_adapter.py†L80-L83】\n   - Layer2：拼接 `layer2_features + z1` 训练并输出 `z2`。\\n【F:methods/dbrb_adapter.py†L85-L92】\n   - Layer3：拼接 `layer3_features + z2`，输出最终系统概率。\\n【F:methods/dbrb_adapter.py†L94-L141】\n4. **层内推理**：每层用“类条件均值/方差”的高斯似然推理，输出概率。\\n【F:methods/dbrb_adapter.py†L150-L191】\n\n> 注：该实现遵循“XGBoost重要性排序 + 分级注入”的思想，但层内推理为简化高斯模型，并非传统 BRB 规则库。\n\n### 4.4 HCF（分层认知框架）

**核心思想**\n- Level-a：Fisher 评分挑选主/次特征；\n- Level-b：按特征来源分组做 GMM 聚类并编码；\n- Level-c：拼接编码后用逻辑回归输出系统结果。\\n【F:methods/hcf_adapter.py†L11-L176】

### 4.5 AIFD（自适应可解释 BRB）

**核心思想与流程**\n- 先选择少量特征并标准化；初始化属性权重与规则权重；使用有限差分估计损失梯度并迭代更新规则权重；每次更新强制非负并归一化，保持可解释性。\\n【F:methods/aifd_adapter.py†L11-L126】【F:methods/aifd_adapter.py†L128-L204】

### 4.6 BRB-P（规则分区 BRB）

**核心思想与流程**\n- 通过特征分区构造规则（区间/分段），以规则激活度进行推理输出系统概率；规则权重在训练中学习并用于推理。\\n【F:methods/brb_p_adapter.py†L32-L96】

### 4.7 A-IBRB（区间规则 BRB）

**核心思想与流程**\n- 对每个特征构建区间规则并进行区间匹配；规则的置信度由区间内样本统计估计，再进行简化推断。\\n【F:methods/a_ibrb_adapter.py†L33-L117】

### 4.8 Fast-BRB（快速规则合并）

**核心思想与流程**\n- 量化特征生成规则，合并相似规则并削减冗余规则，用少量规则近似全量规则推理以提升效率。\\n【F:methods/fast_brb_adapter.py†L27-L179】

## 5. 规则推导与参数说明（通用）

- BRB 系列方法的规则结构与模块组定义来自 `BRB/module_brb.py`（模块层）与 `BRB/system_brb*.py`（系统层），在 `SimpleBRB/ERBRB` 组合器中执行加权融合与归一化。\n【F:BRB/module_brb.py†L21-L360】【F:BRB/utils.py†L6-L61】
- 系统层“正常锚点 + 软门控”由 `BRB/normal_anchor.py` 与 `BRB/aggregator.py` 共同实现，阈值（`T_low/T_high`、`pmax_threshold`、`margin_threshold`）可通过校准覆盖。\n【F:BRB/normal_anchor.py†L39-L287】【F:BRB/aggregator.py†L777-L892】

## 6. 对比流程（审计与一致性）

- 对比评估使用统一标签顺序 `['正常','幅度失准','频率失准','参考电平失准']`，并在加载特征时进行泄漏列检查与审计记录，确保方法间公平比较。\n【F:pipelines/compare_methods.py†L53-L1150】
