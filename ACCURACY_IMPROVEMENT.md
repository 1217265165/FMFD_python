# 系统级BRB诊断准确率改进说明

## 改进内容

本次改进针对 `BRB/system_brb.py` 中的 `system_level_infer_er` 函数和 `pipelines/compare_methods.py` 中的分类逻辑，主要解决诊断准确率低的问题。

## 主要问题

原实现存在以下问题：

1. **缺少正常状态建模**：只输出3种故障类型概率，无法有效区分正常样本
2. **阈值设置不当**：特征归一化阈值过高，导致正常样本也产生较高故障概率
3. **Softmax温度过高**：alpha=5.0导致过度偏向单一类别，缺乏不确定性表达
4. **分类逻辑简单**：仅用0.5阈值判断正常/故障，区分度不足

## 改进措施

### 1. 特征归一化阈值优化

调整各特征的归一化范围，更好地适应数据分布：

```python
# 改进前：阈值过高
bias_score = normalize_feature(abs(bias), 0.1, 1.0)
gain_score = normalize_feature(abs(gain - 1.0), 0.02, 0.2)
freq_raw = normalize_feature(abs(df), 1e6, 5e7)

# 改进后：阈值降低，更灵敏
bias_score = normalize_feature(abs(bias), 0.05, 0.5)
gain_score = normalize_feature(abs(gain - 1.0), 0.01, 0.15)
freq_raw = normalize_feature(abs(df), 5e5, 3e7)
```

### 2. 增加正常状态检测机制

使用综合得分判断是否为正常状态：

```python
# 加权平均综合得分
overall_score = 0.4 * amp_raw + 0.3 * freq_raw + 0.3 * ref_raw

# 低于阈值判为正常
if overall_score < 0.15:
    # 返回接近零的均匀分布
    return {"幅度失准": overall_score/3, ...}
```

### 3. Softmax温度系数降低

```python
# 改进前：alpha = 5.0（过度偏向）
# 改进后：alpha = 2.0（保留不确定性）
alpha = 2.0
```

### 4. 改进分类逻辑

在 `compare_methods.py` 中：

```python
# 改进前：阈值0.5
pred_label = "正常" if max_prob < 0.5 else max(sys_probs, key=sys_probs.get)

# 改进后：阈值0.3，对正常和故障样本都使用一致的逻辑
max_prob = max(sys_probs.values())
if max_prob < 0.3:
    pred_label = "正常"
else:
    pred_label = max(sys_probs, key=sys_probs.get)
```

## 预期效果

改进后的系统应该能够：

1. **更好地识别正常样本**：正常样本的最大故障概率应 < 0.3
2. **提高故障分类准确率**：故障样本能产生 > 0.3 的特定故障类型概率
3. **保持不确定性表达**：不会过度自信，允许中等置信度
4. **提升整体准确率**：预计从26%提升到80%以上

## 测试验证

```python
# 正常样本测试
normal_features = {
    'bias': 0.01, 'gain': 1.0, 'comp': 0.001,
    'df': 1e5, 'step_score': 0.05, 'viol_rate': 0.005
}
# 预期：所有故障概率接近0

# 幅度故障测试
fault_amp = {
    'bias': 0.3, 'gain': 1.15, 'comp': 0.05,
    'df': 1e5, 'step_score': 0.1, 'viol_rate': 0.01
}
# 预期：幅度失准概率 > 0.7

# 频率故障测试
fault_freq = {
    'bias': 0.02, 'gain': 1.01, 'comp': 0.005,
    'df': 1e7, 'step_score': 0.05, 'viol_rate': 0.005
}
# 预期：频率失准概率 > 0.6
```

## 进一步优化建议

如果准确率仍不理想，可以：

1. 根据实际数据统计调整归一化阈值
2. 考虑使用完整的ERBRB方法（在 `utils.py` 中已实现）
3. 引入更多特征或优化特征组合方式
4. 使用数据驱动的方法学习最优阈值参数
