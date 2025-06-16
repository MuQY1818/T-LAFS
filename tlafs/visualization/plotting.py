"""
Visualization tools for TLAFS
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import os

def plot_time_series(dates: List[str], values: List[float], 
                    title: str, save_path: Optional[str] = None):
    """
    Plot time series data
    
    Args:
        dates: List of dates
        values: List of values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions(dates: List[str], y_true: List[float], y_pred: List[float],
                    title: str, save_path: Optional[str] = None):
    """
    Plot predictions against true values
    
    Args:
        dates: List of dates
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='True', color='blue')
    plt.plot(dates, y_pred, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_names: List[str], importance_scores: List[float],
                          title: str, save_path: Optional[str] = None):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: List of importance scores
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_comparison(model_names: List[str], metrics: Dict[str, List[float]],
                         title: str, save_path: Optional[str] = None):
    """
    Plot model comparison
    
    Args:
        model_names: List of model names
        metrics: Dictionary of metrics and their values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width, label=metric_name)
    
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(x + width * (len(metrics) - 1) / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        title: str, save_path: Optional[str] = None):
    """
    Plot learning curves
    
    Args:
        train_scores: List of training scores
        val_scores: List of validation scores
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    plt.plot(epochs, train_scores, label='Training')
    plt.plot(epochs, val_scores, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, probe_name, best_model_metrics, results_dir, filename_suffix=""):
    """
    可视化最终的真实值与预测值对比图。
    """
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    
    # 确保 dates 是 pandas DatetimeIndex 或类似类型，以便绘图
    # 如果不是，尝试转换
    try:
        plot_dates = pd.to_datetime(dates)
    except (TypeError, ValueError):
        # 如果转换失败，可能已经是可绘制的格式，直接使用
        plot_dates = dates
        
    plt.plot(plot_dates, y_true, label='实际值', color='dodgerblue', alpha=0.9)
    plt.plot(plot_dates, y_pred, label=f'预测值 ({best_model_name})', color='orangered', linestyle='--')
    
    title = (f"最终验证 (探针: {probe_name}) - 表现最佳模型: {best_model_name}\n"
             f"R²: {best_model_metrics['r2']:.4f} | MAE: {best_model_metrics['mae']:.4f} | RMSE: {best_model_metrics['rmse']:.4f}")
    
    plt.title(title, fontsize=14)
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.legend()
    plt.tight_layout()
    
    # 使用 suffix 构建文件名
    plot_filename = f"final_predictions_probe_{probe_name}{filename_suffix}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    
    plt.savefig(plot_path)
    print(f"✅ 最终预测图已保存至: {plot_path}")
    plt.close() 