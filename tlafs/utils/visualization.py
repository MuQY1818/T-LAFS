"""
可视化工具模块

提供常用的实验结果可视化函数。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    title: str = "预测结果对比",
                    save_path: Optional[str] = None) -> None:
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="真实值", color="dodgerblue")
    plt.plot(y_pred, label="预测值", color="orangered", linestyle="--")
    plt.title(title)
    plt.xlabel("样本序号")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                         save_path: Optional[str] = None) -> None:
    """
    绘制模型性能对比图
    
    Args:
        metrics_dict: 模型指标字典
        save_path: 保存路径
    """
    models = list(metrics_dict.keys())
    metrics = ['rmse', 'mae', 'r2']
    metric_names = ['RMSE', 'MAE', 'R²']
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [metrics_dict[m][metric] for m in models]
        plt.bar(x + i*width, values, width, label=name)
    
    plt.xticks(x + width, models, rotation=30)
    plt.title("不同模型性能对比")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_learning_curves(train_losses: List[float], 
                        val_losses: List[float],
                        save_path: Optional[str] = None) -> None:
    """
    绘制学习曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="训练损失", color="dodgerblue")
    plt.plot(val_losses, label="验证损失", color="orangered")
    plt.title("学习曲线")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 