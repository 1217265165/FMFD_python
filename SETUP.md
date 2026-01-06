# 安装和设置说明

## 环境要求

- Python 3.8+
- 依赖包：numpy, pandas, scikit-learn, matplotlib, scipy, pyyaml, openpyxl

## 重要：FMFD 包设置

本项目代码使用 `from FMFD.xxx import` 格式导入模块。为了使 Python 能够正确识别包路径，需要创建符号链接：

### 步骤

1. **进入仓库父目录**
```bash
cd /home/runner/work/FMFD-PY  # 或你克隆仓库的位置
```

2. **创建 FMFD 符号链接**
```bash
ln -s FMFD-PY FMFD
```

3. **验证设置**
```bash
ls -la | grep FMFD
# 应该看到:
# lrwxrwxrwx ... FMFD -> FMFD-PY
# drwxr-xr-x ... FMFD-PY
```

4. **测试导入**
```bash
python -c "from FMFD.comparison import HCFMethod; print('✓ 导入成功')"
```

### 运行脚本

所有脚本都应该从符号链接父目录运行：

```bash
# 正确 ✓
cd /home/runner/work/FMFD-PY
python -m FMFD.pipelines.compare_methods
python -m FMFD.comparison.demo

# 错误 ✗ 
cd /home/runner/work/FMFD-PY/FMFD-PY
python -m pipelines.compare_methods  # 会找不到 FMFD 包
```

## 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib scipy pyyaml openpyxl
```

## 快速验证

运行演示脚本验证环境设置：

```bash
cd /home/runner/work/FMFD-PY
python -m FMFD.comparison.demo
```

如果看到对比结果输出，说明环境设置正确。

## 常见问题

### Q: 提示 "ModuleNotFoundError: No module named 'FMFD'"

**A**: 检查：
1. 是否创建了 FMFD 符号链接
2. 是否在正确的目录运行（应该在符号链接父目录）
3. 符号链接是否指向正确的目录

```bash
# 检查符号链接
cd /home/runner/work/FMFD-PY
ls -la FMFD

# 重新创建（如果需要）
rm FMFD  # 删除错误的链接
ln -s FMFD-PY FMFD
```

### Q: 为什么需要符号链接？

**A**: 项目代码使用 `from FMFD.xxx` 导入格式，这是标准的 Python 包结构。由于仓库名称是 `FMFD-PY`（包含连字符），不能直接作为 Python 包名（Python 包名不允许连字符）。符号链接提供了一个没有连字符的名称 `FMFD`，使 Python 能够正确导入。

### Q: 可以不使用符号链接吗？

**A**: 可以，但需要修改所有导入语句。不推荐这样做，因为会与仓库中其他文件的导入风格不一致。
