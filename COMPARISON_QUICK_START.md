# Quick Start: Method Comparison

## 一键运行对比实验

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成仿真数据（如果需要）

**重新生成均衡的仿真样本**（推荐，混淆矩阵更美观）：

```bash
# 生成200个样本，4类故障各50个（均衡分布）
PYTHONPATH=. python pipelines/simulate/run_simulation_brb.py --n_samples 200 --balanced

# 自定义样本数（建议4的倍数以完美平衡）
PYTHONPATH=. python pipelines/simulate/run_simulation_brb.py --n_samples 400 --balanced

# 使用原始随机概率生成（不均衡）
PYTHONPATH=. python pipelines/simulate/run_simulation_brb.py --n_samples 200 --no-balanced
```

**样本分布对比**：
- `--balanced`（默认）：amp_error=50, freq_error=50, ref_error=50, normal=50
- `--no-balanced`：amp_error≈111, freq_error≈43, ref_error≈30, normal≈16（随机）

**生成特征**（从已有raw_curves生成）：

```bash
python pipelines/generate_features.py
```

这会从 `Output/sim_spectrum/raw_curves/` 生成 `Output/sim_spectrum/features_brb.csv`

### 3. 运行完整对比

```bash
python pipelines/compare_methods.py
```

或使用模块方式:
```bash
python -m pipelines.compare_methods
```

### 4. 查看结果

结果保存在 `Output/sim_spectrum/`:
- `comparison_table.csv` - 主对比表
- `confusion_matrix_*.png` - 各方法混淆矩阵（均衡数据下更美观）
- `compare_barplot.png` - 规则数/参数数/推理时间对比图

## 输出示例（均衡数据）

```csv
method,sys_accuracy,sys_macro_f1,n_rules,n_params,n_features_used
ours,0.5349,0.1742,45,38,10
hcf,0.5581,0.2224,90,130,53
aifd,0.5349,0.1742,72,110,6
brb_p,0.5349,0.1742,81,571,15
brb_mu,0.7674,0.5929,72,110,53
dbrb,0.7674,0.7232,60,90,53
a_ibrb,0.5349,0.1742,4,65,5
fast_brb,0.5581,0.2224,2,23,5
```

## 高级选项

### 小样本适应性实验

```bash
python pipelines/compare_methods.py --small_sample
```

会额外测试训练集大小为 [5, 10, 20, 30] 的情况，每个重复5次。

### 自定义参数

```bash
python pipelines/compare_methods.py \
  --data_dir Output/sim_spectrum \
  --output_dir Output/comparison_results \
  --seed 42 \
  --train_size 0.6 \
  --val_size 0.2
```

## 方法说明

实现了8种对比方法:

1. **Ours** - 知识驱动规则压缩 + 分层BRB（本文方法）
2. **HCF** (Zhang 2022) - 分层认知框架（FN3WD + GMM + 分类器）
3. **AIFD** (Li 2022) - 自适应BRB（基于灵敏度的权重更新）
4. **BRB-P** (Ming 2023) - 概率表约束BRB（概率初始化 + 约束优化）
5. **BRB-MU** (Feng 2024) - 多源不确定融合BRB（SNR + SVD不确定度）
6. **DBRB** (Zhao 2024) - 深层BRB（XGBoost特征重要性 + 3层结构）
7. **A-IBRB** (Wan 2025) - 自动区间BRB（误差约束聚类 + 自动规则生成）
8. **Fast-BRB** (Gao 2023) - 快速BRB（规则生成 + 相似度合并 + 冗余约简）

详细实现说明见 `METHODS_IMPLEMENTATION.md`

## 核心机制验证

每个方法都实现了论文的MUST-HAVE机制:

- ✅ **HCF**: 三层认知框架（特征认知 + 模式认知 + 数据气候认知）
- ✅ **AIFD**: 基于灵敏度的自适应权重更新（有限差分梯度估计）
- ✅ **BRB-P**: 概率表初始化 + 解释性约束优化
- ✅ **BRB-MU**: 多源不确定度建模（SNR + SVD） + 加权融合
- ✅ **DBRB**: XGBoost特征重要性 + 逐层输入BRB结构
- ✅ **A-IBRB**: 误差约束区间构建 + 自动规则生成 + GIBM初始化
- ✅ **Fast-BRB**: 快速生成 + 相似度合并 + 冗余度约简

## 特征视图

### 知识驱动特征 (Ours使用)
- X1-X5: bias, ripple_var, res_slope, df, scale_consistency
- 共10个特征（包含别名）

### Pool特征 (对比方法使用)
从原始频响曲线提取的宽特征集:
- 幅度全局特征: 11个
- 频率刻度特征: 6个
- 噪声与纹波: 7个
- 开关/转换: 1个
- 局部频段: 16个
- 频谱形状: 2个
- 兼容性映射: 10个
- **共53个特征**

## 常见问题

**Q: 缺少 features_brb.csv 怎么办？**

A: 运行 `python pipelines/generate_features.py` 从原始曲线生成

**Q: xgboost 未安装？**

A: DBRB会自动降级使用sklearn的GradientBoostingClassifier

**Q: 准确率都很低？**

A: 检查:
1. 训练集是否足够 (--train_size 0.7)
2. 标签是否平衡 (查看输出中的label distribution)
3. 特征提取是否正确

**Q: 如何验证方法是否实现了核心机制？**

A: 检查输出的meta信息:
- HCF: meta['primary_features'], 'secondary_features'
- AIFD: fit_time > 0 说明做了优化
- BRB-MU: meta['source_uncertainties'], 'source_weights'
- DBRB: meta['feature_importance'], 'layer_sizes'
- A-IBRB: meta['interval_stats']
- Fast-BRB: meta['n_rules_before_reduction'], 'n_rules_after_reduction'

## 技术支持

详细文档: `METHODS_IMPLEMENTATION.md`

如遇问题请提issue或联系作者。
