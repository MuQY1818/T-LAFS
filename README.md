# T-LAFS: 时序语言增强特征搜索框架

T-LAFS (Time-series Language-model Augmented Feature Search) 是一个先进的、自动化的时间序列特征工程框架。它利用大语言模型 (LLM) 的推理能力，为特定的预测模型（尤其是Transformer）智能地、自适应地生成最匹配其"归纳偏置"的特征组合。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心理念与特性

T-LAFS的核心目标是解决复杂预测模型中的 **"模型-特征不匹配"** 问题。我们观察到，简单地将原始数据喂给强大的模型（如Transformer）往往效果不佳。T-LAFS通过构建一个闭环的、类似强化学习的自治系统来应对这一挑战。

- **LLM驱动的特征策略**: 使用Google的Gemini模型作为"策略大脑"，根据历史成败经验和当前特征集的状态，动态生成特征工程计划。
- **两阶段探索策略**: 模拟专家工作流，在特征搜索的**前期（前40%迭代）**优先构建稳固的基础特征（如滞后、滚动），在**后期（后60%迭代）**则侧重于探索复杂的学习型特征。
- **高效MVSE探针**: 集成了创新的**多视角序列嵌入 (Multi-View Sequential Embedding, MVSE)** 技术。它能从多个视角（全局平均、全局最大、遮罩平均）高效地从原始序列中提取少量但强大的特征，极大加速了优质特征的发现过程。
- **可学习嵌入特征**: 通过掩码自编码器，从不同尺度的时间窗口中自主学习高密度的信息表示，为模型提供超越传统手工特征的深层模式。
- **全面的评估与分析**: 框架会自动评估多种模型（LightGBM, SimpleNN, Transformer等）的性能，并提供迭代贡献分析和排列重要性分析报告。

## 如何运行实验

本项目被构建为一个标准的Python包，您可以通过模块化的方式轻松运行任何预设的实验。

### 1. 安装

首先，请确保您已安装所有依赖。推荐在虚拟环境中进行安装。

```bash
# 安装项目依赖
pip install -r requirements.txt # (如果提供了requirements.txt)

# 以可编辑模式安装T-LAFS包
pip install -e .
```
> **注意**: `-e` 参数代表"可编辑"模式，这意味着您对源代码的任何修改都会立刻生效，无需重新安装。

### 2. 设置API密钥
T-LAFS需要使用Google Gemini API。请在您的环境中设置环境变量 `GOOGLE_API_KEY`。

```bash
# 在Linux或macOS
export GOOGLE_API_KEY="您的API密钥"

# 在Windows (CMD)
set GOOGLE_API_KEY="您的API密钥"

# 在Windows (PowerShell)
$env:GOOGLE_API_KEY="您的API密钥"
```

### 3. 运行核心实验
我们推荐从核心的 `sonata` 实验开始。该实验完整地展示了T-LAFS的所有功能。

在项目根目录下，运行以下命令：

```bash
python -m tlafs.experiments.sonata
```

实验开始后，您将在控制台看到详细的日志输出，包括每一轮的特征生成计划、性能探针的评估分数（MAE），以及最终的模型验证结果。

### 4. 查看结果
每一次运行的结果，包括生成的特征、模型性能、历史记录摘要(`sonata_tlafs_summary.json`)以及预测结果的可视化图表，都会保存在一个新的、带有时间戳的子目录中，位于 `results/` 文件夹下。

## 项目结构

```
.
├── checkpoints/        # 存储训练好的模型权重
├── results/            # 存放所有实验的结果、日志和图表
├── tlafs/              # T-LAFS核心源码包
│   ├── core/           # 核心算法 (TLAFS_Algorithm)
│   ├── experiments/    # 实验定义脚本 (如 sonata.py)
│   ├── features/       # 特征生成器 (包括 mvse_features.py)
│   ├── models/         # 模型定义 (包括 mvse.py, neural_models.py)
│   ├── utils/          # 工具函数 (评估、训练、数据处理等)
│   └── visualization/  # 可视化工具
├── README.md           # 项目说明文档
└── setup.py            # 包定义文件
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