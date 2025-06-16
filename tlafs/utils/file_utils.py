"""
文件工具模块

提供文件操作和结果保存的工具函数。
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np

def save_results(results, filepath):
    """保存实验结果到JSON文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 将结果转换为可序列化的格式
    serializable_results = {}
    for model_name, model_results in results.items():
        if model_name == 'test_data':
            serializable_results[model_name] = model_results.tolist() if isinstance(model_results, np.ndarray) else model_results
            continue
            
        serializable_results[model_name] = {
            'metrics': model_results['metrics'],
            'predictions': model_results['predictions'].tolist() if isinstance(model_results['predictions'], np.ndarray) else model_results['predictions'],
            'history': {
                'train_losses': model_results['history']['train_losses'],
                'val_losses': model_results['history']['val_losses']
            }
        }
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

def load_results(filepath: str) -> Dict[str, Any]:
    """
    加载实验结果
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的结果数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f) 