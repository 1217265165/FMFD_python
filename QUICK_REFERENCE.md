# 对比方法快速参考指南

## 环境设置（重要）

在使用对比方法前，需要创建 FMFD 符号链接：

```bash
# 进入仓库父目录
cd /home/runner/work/FMFD-PY  # 或你的工作目录父目录

# 创建符号链接（如果还没有）
ln -s FMFD-PY FMFD

# 验证链接
ls -la | grep FMFD
# 应该看到: lrwxrwxrwx ... FMFD -> FMFD-PY
```

**为什么需要这个？**  
代码使用 `from FMFD.xxx import` 格式，符号链接使 Python 能够将 `FMFD-PY` 目录识别为 `FMFD` 包。

## 新增文件列表

### 核心实现文件
```
comparison/
├── __init__.py              (12 lines)  - 模块导入
├── hcf.py                   (247 lines) - HCF方法实现  
├── brb_p.py                 (232 lines) - BRB-P方法实现
├── er_c.py                  (237 lines) - ER-c方法实现
├── demo.py                  (133 lines) - 演示脚本
└── README.md                (153 lines) - 详细文档

pipelines/
└── compare_methods.py       (477 lines) - 对比评估脚本

其他文档:
├── IMPLEMENTATION_SUMMARY.md (223 lines) - 实现总结
└── .gitignore               (35 lines)  - Git忽略规则
```

**总计**: ~1338行新代码

## 快速开始

### 1. 运行对比评估（一键生成所有结果）
```bash
# 确保在 FMFD 符号链接的父目录
cd /home/runner/work/FMFD-PY
python -m FMFD.pipelines.compare_methods
```

**输出**:
- `FMFD-PY/Output/comparison_results/comparison_table.csv` - 方法对比表
- `FMFD-PY/Output/comparison_results/performance_table.csv` - 性能详细表  
- `FMFD-PY/Output/comparison_results/comparison_plot.png` - 对比图表
- `FMFD-PY/Output/comparison_results/confusion_matrices.png` - 混淆矩阵
- `FMFD-PY/Output/comparison_results/comparison_summary.txt` - 文字报告

**注意**: 仿真CSV数据在 `Output/sim_spectrum/`，对比结果在 `Output/comparison_results/`

### 2. 查看演示（理解如何使用）
```bash
# 从 FMFD 符号链接父目录运行
cd /home/runner/work/FMFD-PY
python -m FMFD.comparison.demo
```

**展示**:
- 单样本推理过程
- 各方法参数对比
- 模块级诊断结果

## 对比结果示例

### 方法规模对比
| 方法              | 规则数 | 参数数 | 特征维度 |
|-------------------|--------|--------|----------|
| HCF (Zhang 2022)  | 174    | 1392   | 15       |
| BRB-P (Ming 2023) | 81     | 571    | 15       |
| ER-c (Zhang 2024) | 60     | 150    | 10       |
| **本文方法**      | **45** | **38** | **4**    |

### 本文方法优势
- 规则数减少: **74.1%** (vs HCF)
- 参数数减少: **97.3%** (vs HCF)
- 特征维度减少: **73.3%** (vs HCF)
- 推理速度: **最快** (0.01ms)

## 扩展新方法

### 添加自定义对比方法

1. **创建方法类** (`comparison/my_method.py`)
```python
class MyMethod:
    def get_num_rules(self) -> int:
        return 100  # 你的规则数
    
    def get_num_parameters(self) -> int:
        return 500  # 你的参数数
    
    def get_feature_dimension(self) -> int:
        return 10  # 你的特征维度
    
    def infer_system(self, features: Dict[str, float]) -> Dict[str, float]:
        # 系统级推理逻辑
        return {"幅度失准": 0.5, "频率失准": 0.3, "参考电平失准": 0.2}
    
    def infer_module(self, features: Dict[str, float],
                    sys_result: Dict[str, float]) -> Dict[str, float]:
        # 模块级推理逻辑
        return {module: 0.05 for module in all_modules}
```

2. **注册到对比框架** (`pipelines/compare_methods.py`)
```python
from comparison.my_method import MyMethod

# 在 MethodComparison.__init__ 中添加:
self.methods = {
    "HCF (Zhang 2022)": HCFMethod(),
    "BRB-P (Ming 2023)": BRBPMethod(),
    "ER-c (Zhang 2024)": ERCMethod(),
    "本文方法": None,
    "我的方法": MyMethod(),  # 新增
}
```

3. **重新运行对比**
```bash
python -m pipelines.compare_methods
```

## 常见问题

### Q: 为什么准确率与论文不同？
A: 对比方法基于论文描述实现，参数未针对本数据集优化。主要用于框架对比和指标计算。

### Q: 如何只运行特定方法？
A: 修改 `compare_methods.py` 中的 `self.methods` 字典，注释掉不需要的方法。

### Q: 输出结果在哪里？
A: 所有结果保存在 `Output/sim_spectrum/` 目录下，文件名以 `comparison_` 开头。

### Q: 如何自定义评估指标？
A: 修改 `compare_methods.py` 中的 `evaluate_method()` 方法，添加新的指标计算。

## 文件依赖关系

```
comparison/
├── hcf.py ────────┐
├── brb_p.py ──────┤
├── er_c.py ───────┼──> pipelines/compare_methods.py ──> 输出文件
└── __init__.py ───┘

BRB/
├── system_brb.py ─┬──> pipelines/compare_methods.py
└── module_brb.py ─┘
```

## 更多信息

- 详细文档: `comparison/README.md`
- 实现总结: `IMPLEMENTATION_SUMMARY.md`
- 主文档: `BRB.md` (包含对比模块说明)
