# TLAFS (Time Series Learning and Feature Selection)

TLAFS是一个用于时间序列预测的高级特征选择和学习的Python包。它结合了多种先进的技术，包括多视角序列编码(MVSE)、探针预测器和基于LLM的特征工程。

## 项目结构

```
tlafs/
├── core/           # 核心算法实现
│   ├── __init__.py
│   └── algorithm.py
├── models/         # 模型定义
│   ├── __init__.py
│   └── neural_models.py
├── features/       # 特征工程
│   ├── __init__.py
│   └── mvse.py
├── utils/          # 工具函数
│   ├── __init__.py
│   └── common.py
├── experiments/    # 实验脚本
│   ├── __init__.py
│   ├── baseline.py
│   ├── control.py
│   └── sonata.py
└── visualization/  # 可视化工具
    ├── __init__.py
    └── plotting.py
```

## 主要功能

1. 多视角序列编码 (MVSE)
   - 多视角LSTM编码
   - 注意力机制
   - 掩码学习

2. 探针预测器
   - 基于Transformer的预测
   - 位置编码
   - 代理注意力

3. 特征工程
   - 自动特征选择
   - 基于LLM的特征生成
   - 特征重要性分析

4. 模型评估
   - 多模型比较
   - 性能指标计算
   - 可视化工具

## 安装

```bash
pip install -e .
```

## 使用示例

```python
from tlafs.core.algorithm import TLAFS_Algorithm
import pandas as pd

# 加载数据
df = pd.read_csv('your_data.csv')

# 初始化TLAFS算法
tlafs = TLAFS_Algorithm(
    base_df=df,
    target_col='target',
    n_iterations=5
)

# 运行算法
results = tlafs.run()
```

## 依赖

- Python >= 3.7
- PyTorch >= 1.7.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- numpy >= 1.19.2
- matplotlib >= 3.3.0
- lightgbm >= 3.1.0
- xgboost >= 1.3.0
- catboost >= 0.24.0
- pytorch-tabnet >= 3.0.0
- google-generativeai >= 0.1.0

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT License