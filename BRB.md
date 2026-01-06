Remove-Item -Recurse -Force FMFD\Output\sim_spectrum
python -m FMFD.pipelines.run_babeline
python -m FMFD.pipelines.simulate.run_sinulation_brb
python -m FMFD.pipelines.detect
python -m FMFD.tools.brb_report  # 如果放在 FMFD/tools 下
python -m FMFD.pipelines.eval_system_level
python -m FMFD.pipelines.compare_methods  # 运行方法对比

# BRB 两层方案（21 模块）使用说明

## 模块列表（与 `brb_rules.yaml` / `BRB/module_brb.py` / `features/feature_extraction.py` 一致）
1. 衰减器  
2. 前置放大器  
3. 低频段前置低通滤波器  
4. 低频段第一混频器  
5. 高频段 YTF 滤波器  
6. 高频段混频器  
7. 时钟振荡器  
8. 时钟合成与同步网络  
9. 本振源（谐波发生器）  
10. 本振混频组件  
11. 校准源  
12. 存储器  
13. 校准信号开关  
14. 中频放大器  
15. ADC  
16. 数字 RBW  
17. 数字放大器  
18. 数字检波器  
19. VBW 滤波器  
20. 电源模块  
21. 未定义/其他  

系统级异常：幅度失准、频率失准、参考电平失准。

## 目录与关键文件（按当前结构）
FMFD/  
├─ baseline/  
│  ├─ baseline.py          # 读取/对齐正常 CSV，计算 RRS/包络、切换点特性  
│  ├─ config.py            # 频段、输出路径、文件名等配置（输出到 Output/）  
│  ├─ viz.py               # RRS/包络/切换点可视化  
├─ BRB/  
│  ├─ system_brb.py        # 第一层 BRB（幅度/频率/参考电平）  
│  ├─ module_brb.py        # 第二层 BRB（21 模块）  
│  ├─ utils.py             # BRB 工具  
├─ comparison/             # 对比方法实现（新增）
│  ├─ hcf.py              # HCF (Zhang 2022) 方法
│  ├─ brb_p.py            # BRB-P (Ming 2023) 方法
│  ├─ er_c.py             # ER-c (Zhang 2024) 方法
│  └─ README.md           # 对比方法说明文档
├─ features/  
│  ├─ extract.py           # 系统特征提取（切换点/非切换台阶异常）  
│  ├─ feature_extraction.py# 采集数据特征工程 + module_meta(21 维)，与 brb_rules.yaml 对齐  
├─ pipelines/  
│  ├─ run_babeline.py      # 基线构建：对齐正常数据，算 RRS/包络，输出至 Output/ 下 npz/json/png/csv  
│  ├─ compare_methods.py   # 方法对比评估脚本（新增）
│  ├─ simulate/
│  │  └─ run_sinulation_brb.py # 仿真故障→特征→两层 BRB，输出 sim_spectrum/ 下结果  
│  ├─ detect.py            # 检测 to_detect/ 下 CSV，输出 Output/detection_results.csv  
├─ Output/                 # 所有运行输出（baseline_artifacts.npz 等；sim_spectrum/ 等子目录）  
├─ normal_response_data/   # 正常频响 CSV（frequency, amplitude_dB 两列）  
├─ to_detect/              # 待检测 CSV（frequency, amplitude 两列）  
├─ brb_rules.yaml          # 21 模块规则/先验（与 feature_extraction / module_brb 对齐）  
├─ brb_chains_generated.yaml# 可选：自动链路规则，当前主链未使用  
├─ thresholds.json         # 检测阈值（detect.py 使用）  

## 使用步骤

### 0. 环境
依赖：numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml。可选：cma（规则优化）、pywt/statsmodels（若扩展时序/小波特征）。

### 1. 基线构建（正常数据）
1) 将 ≥30 条正常频响 CSV 放到 `FMFD/normal_response_data/`（两列：frequency, amplitude_dB）。  
2) 运行（在仓库根）：  
```bash
python pipelines/run_babeline.py
```
输出（在 `Output/`）：  
- baseline_artifacts.npz（frequency, rrs, upper, lower, traces）  
- baseline_meta.json（band_ranges, k_list）  
- switching_features.csv / switching_features.json  
- normal_feature_stats.csv  
- baseline_switching.png

### 2. 故障仿真 + BRB（可选）
在仓库根运行：  
```bash
python pipelines/simulate/run_sinulation_brb.py
# 或指定基线/输出
python pipelines/simulate/run_sinulation_brb.py \
  --baseline_npz ./Output/baseline_artifacts.npz \
  --out_dir ./Output/sim_spectrum \
  --n_normal 50 --n_fault 225 --seed 20251204
```
输出：`Output/sim_spectrum/` 下的 normal_*.csv / fault_*.csv / labels.json / statistics.json / features_brb.csv 等。

### 3. 检测待检数据
1) 待检 CSV 放 `FMFD/to_detect/`（两列：frequency, amplitude）。  
2) 确认 `thresholds.json` 在仓库根。  
3) 在仓库根运行：  
```bash
python pipelines/detect.py
```
输出：`Output/detection_results.csv`（特征、系统/模块概率、warn/alarm/ok 标志）。

### 4. 采集数据的症状/聚类/模块元信息（可选）
针对采集的 measurement CSV 运行：  
```bash
python FMFD/features/feature_extraction.py \
  --input acquired_measurements.csv \
  --prefix run_enh \
  --out_dir ./Output
```
输出：`run_enh_features_enhanced.csv` / `run_enh_module_meta.csv` / `run_enh_feature_summary.csv`（可选 feature_importances）。

## 一致性说明
- 模块顺序在 `brb_rules.yaml`、`BRB/module_brb.py`、`features/feature_extraction.py` 中保持一致（21 模块）。  
- system_brb.py 输出三类系统级异常；module_brb.py 输出 21 模块概率。  
- detect.py 与 run_sinulation_brb.py 均使用两层 BRB（system_brb + module_brb）。  
- `brb_chains_generated.yaml` 当前未接入主链，若需链路推理请另行接入对应引擎。

## 方法对比（新增）
本项目包含与其他分层诊断方法的对比实验模块，位于 `comparison/` 目录。

### 对比方法
1. **HCF (Zhang 2022)**: 分层认知框架，130条规则，200+参数
2. **BRB-P (Ming 2023)**: 概率约束BRB，81条规则，571个参数
3. **ER-c (Zhang 2024)**: 增强可信度推理，60条规则，150个参数
4. **本文方法**: 知识驱动分层BRB，45条规则，38个参数

### 运行对比实验
```bash
# 1. 生成仿真数据
python -m FMFD.pipelines.simulate.run_sinulation_brb

# 2. 运行对比评估
python -m FMFD.pipelines.compare_methods

# 3. 查看结果
# Output/sim_spectrum/comparison_table.csv - 对比表
# Output/sim_spectrum/performance_table.csv - 性能表
# Output/sim_spectrum/comparison_plot.png - 对比图
# Output/sim_spectrum/confusion_matrices.png - 混淆矩阵
```

### 关键发现
- **规则压缩**: 45条 vs 130条 (↓59%)
- **参数简化**: 38个 vs 571个 (↓93%)
- **特征降维**: 4维 vs 15维 (↓73%)
- **准确率**: 94.18% (优于BRB-P和ER-c)
- **推理加速**: 3.08倍
- **小样本需求**: 19条 vs 62-100条 (↓70%)

详细说明见 `comparison/README.md`。