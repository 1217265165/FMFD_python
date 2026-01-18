# FMFD 方法说明（单频段）

本文档按“真实代码实现”的口径整理 FMFD 单频段流程与方法细节：RRS/包络、仿真、特征提取、方法的特征池划分、推理与校准机制，并尽量对应到实际实现文件。

> 说明：以下内容以当前仓库实现为准，强调“用了哪些特征、如何分层/注入、如何校准”。

## 1. RRS 基准与包络提取

### 1.1 正常曲线对齐与 RRS
- 正常曲线读取自 `normal_response_data/`，在 `baseline/baseline.py` 中通过线性插值对齐到统一频率轴（`N_POINTS`），得到 `traces` 矩阵。 【F:baseline/baseline.py†L1-L82】
- RRS 采用**点位中位数**（默认不平滑），避免过度平滑导致形状漂移；`RRS_SMOOTH_ENABLED=False` 为默认行为。 【F:baseline/baseline.py†L18-L24】
- 为避免“全局偏移”导致包络过宽，先用 `compute_offsets` 计算每条曲线相对 RRS 的中位数偏移量，再在对齐后的残差上估计包络分布。 【F:baseline/baseline.py†L84-L102】

### 1.2 包络（Envelope）
- 包络宽度来自残差分位数搜索（覆盖率网格），并通过平滑/裁剪/扩张确保覆盖率满足目标；滑窗覆盖率也被检查。 【F:baseline/baseline.py†L26-L44】【F:baseline/baseline.py†L118-L168】
- 输出覆盖率指标（整体、单曲线、点位分位数），用来评估包络质量。 【F:baseline/baseline.py†L118-L168】

## 2. 故障仿真（Simulation）

- 仿真入口：`pipelines/simulate/run_simulation_brb.py`，默认无参数生成 400 样本（平衡四类），输出 `raw_curves/*.csv`、`features_brb.csv`、`labels.json` 等。 【F:pipelines/simulate/run_simulation_brb.py†L1-L118】【F:pipelines/simulate/run_simulation_brb.py†L360-L509】
- 仿真流程：读取 RRS/包络后注入故障（幅度/频率/参考电平/正常），再提取系统与模块特征。 【F:pipelines/simulate/run_simulation_brb.py†L360-L509】

## 3. 特征提取（System + Module）

### 3.1 系统特征（X1–X34）
- `features/feature_extraction.py` 中 `extract_system_features` 生成系统级特征：
  - X1–X5：基础偏差/纹波/斜率/频偏等基础特征。
  - X11–X15：包络/残差相关（越界率、最大越界、越界能量、低/高频残差）。
  - X16–X18：频率对齐与形变（相关峰移位、warp scale/bias）。
  - X19–X22：幅度链路细粒度（低频斜率、峰度、峰值数、主频能量占比）。
  - X23–X34：增强频率/参考电平特征与谱结构统计（如相位斜率差、PSDs）。 【F:features/feature_extraction.py†L292-L758】

### 3.2 模块特征与模块顺序
- `extract_module_features` 复用系统特征与传统特征，保持系统层与模块层一致性。 【F:features/feature_extraction.py†L869-L934】
- 模块顺序统一定义在 `MODULES_ORDER`，并与 BRB 规则列表对齐（已移除“未定义/其他”）。 【F:features/feature_extraction.py†L47-L60】

## 4. 方法总览（System-Level）

评估入口为 `pipelines/compare_methods.py`，统一特征/标签顺序与 split，并输出审计与可复现摘要。 【F:pipelines/compare_methods.py†L53-L1150】

### 4.1 Ours（层次 BRB + 规则压缩 + 正常锚点）

**核心思想**
- 系统层先做“正常锚点”判别，随后对幅度/频率/参考电平分支进行软门控融合，避免硬阈值误判。 【F:BRB/normal_anchor.py†L1-L239】【F:BRB/aggregator.py†L737-L892】
- 模块层只激活与异常类型相关的模块组，实现规则压缩与可解释推理。 【F:BRB/module_brb.py†L21-L360】

**特征池与推理流程**
1. 系统层主要依赖 X1–X22（`OursAdapter.kd_features`），并支持特征别名映射。 【F:methods/ours_adapter.py†L27-L114】
2. `system_level_infer` 输出系统级概率（正常/幅度/频率/参考电平）。 【F:methods/ours_adapter.py†L159-L188】
3. 模块层调用 `module_level_infer_with_activation`，仅激活相关模块组，输出模块概率（20 模块）。 【F:methods/ours_adapter.py†L190-L219】【F:BRB/module_brb.py†L21-L360】

**补偿与校准**
- 校准由 `pipelines/calibrate_ours.py` 网格搜索得到 `best_params.json`/`calibration.json`，`OursAdapter` 加载覆盖 `SystemBRBConfig` 的参数（`alpha`、阈值、权重等）。 【F:pipelines/calibrate_ours.py†L682-L780】【F:methods/ours_adapter.py†L12-L92】

**参数/规则来源**
- 系统层规则与软门控逻辑在 `BRB/aggregator.py` 与 `BRB/normal_anchor.py`；阈值（`T_low/T_high`、`pmax_threshold`、`margin_threshold`）可通过校准覆盖。 【F:BRB/aggregator.py†L777-L892】【F:BRB/normal_anchor.py†L39-L287】
- 模块层规则与模块组定义在 `BRB/module_brb.py`，基于异常类型进行规则压缩。 【F:BRB/module_brb.py†L21-L360】

### 4.2 BRB-MU（多源不确定性融合）

**特征池划分**
- 使用特征名语义把输入分为 amplitude/frequency/noise/switching 四个来源（或均分为 3–4 组）。 【F:methods/brb_mu_adapter.py†L98-L155】

**推理与融合**
1. 对每个来源训练“类条件均值/方差”高斯模型输出概率。 【F:methods/brb_mu_adapter.py†L56-L116】【F:methods/brb_mu_adapter.py†L169-L222】
2. 计算来源不确定度（SNR + SVD），按 `w_s ∝ 1-u_s` 融合来源概率。 【F:methods/brb_mu_adapter.py†L124-L148】

### 4.3 DBRB（深层 BRB / 重要性排序 + 分级注入）

**核心思想**
- 先用 XGBoost/GradientBoosting 得到特征重要性排序，再按排序分成 3 层特征池；每层输出的概率向量作为“隐变量”注入下一层（分级注入）。 【F:methods/dbrb_adapter.py†L29-L141】

**实现流程（按源码）**
1. **特征池**：训练 XGBoost（或回退 GradientBoosting）得到 `feature_importances_`；依赖缺失时回退到方差排序。 【F:methods/dbrb_adapter.py†L29-L63】
2. **重要性排序与分层**：按重要性排序切分 Layer1/Layer2/Layer3。 【F:methods/dbrb_adapter.py†L66-L77】
3. **分级注入**：
   - Layer1：使用 `layer1_features` 推理得到 `z1`。 【F:methods/dbrb_adapter.py†L80-L83】
   - Layer2：拼接 `layer2_features + z1` 推理得到 `z2`。 【F:methods/dbrb_adapter.py†L85-L92】
   - Layer3：拼接 `layer3_features + z2` 输出最终概率。 【F:methods/dbrb_adapter.py†L94-L141】
4. **层内推导**：每层使用高斯似然（类均值/方差 + 先验）推断。 【F:methods/dbrb_adapter.py†L150-L191】

### 4.4 HCF（分层认知框架）

**特征池划分与流程**
- Level‑a：Fisher 分数选择主/次特征。 【F:methods/hcf_adapter.py†L40-L76】
- Level‑b：按语义来源分组，使用 GMM 聚类编码为 one‑hot。 【F:methods/hcf_adapter.py†L75-L117】
- Level‑c：拼接编码特征，逻辑回归输出系统结果。 【F:methods/hcf_adapter.py†L121-L176】

### 4.5 AIFD（自适应可解释 BRB）

**特征池与推导**
- 先选择 6 个最有区分性的特征（方差 × 标签相关），再标准化。 【F:methods/aifd_adapter.py†L31-L76】
- 规则权重通过有限差分估计梯度进行迭代更新，并强制非负与归一化。 【F:methods/aifd_adapter.py†L88-L117】

### 4.6 BRB‑P（规则分区 BRB）

**特征池与推理**
- 使用特征分区构造规则并进行 BRB 推理；规则权重在训练中学习。 【F:methods/brb_p_adapter.py†L32-L96】

### 4.7 A‑IBRB（区间规则 BRB）

**特征池与推理**
- 构建区间规则并做区间匹配；规则置信度由区间内样本统计估计。 【F:methods/a_ibrb_adapter.py†L33-L117】

### 4.8 Fast‑BRB（快速规则合并）

**特征池与推理**
- 量化特征生成规则，合并相似规则并削减冗余规则，用少量规则近似全量推理。 【F:methods/fast_brb_adapter.py†L27-L179】

## 5. 规则推导与参数说明（通用）

- 系统/模块层 BRB 规则结构与模块分组定义位于 `BRB/module_brb.py`、`BRB/system_brb*.py`，推理由 `SimpleBRB/ERBRB` 完成归一化融合。 【F:BRB/module_brb.py†L21-L360】【F:BRB/utils.py†L6-L61】
- 系统层“正常锚点 + 软门控”逻辑在 `BRB/normal_anchor.py` 与 `BRB/aggregator.py`，关键阈值可通过校准覆盖。 【F:BRB/normal_anchor.py†L39-L287】【F:BRB/aggregator.py†L777-L892】
