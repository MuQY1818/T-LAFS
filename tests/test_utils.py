"""
测试TLAFS工具函数模块
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
from tlafs.utils.common import (
    create_sequences,
    evaluate_model,
    save_results,
    load_results
)

class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        self.data = np.random.randn(100)
        self.sequence_length = 14
        self.test_dir = 'test_results'
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试目录
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)
    
    def test_create_sequences(self):
        """测试序列创建函数"""
        X, y = create_sequences(self.data, sequence_length=self.sequence_length)
        
        # 检查输入序列形状
        self.assertEqual(X.shape[0], len(self.data) - self.sequence_length)
        self.assertEqual(X.shape[1], self.sequence_length)
        
        # 检查目标序列形状
        self.assertEqual(y.shape[0], len(self.data) - self.sequence_length)
        
        # 检查序列内容
        for i in range(len(X)):
            self.assertTrue(np.array_equal(X[i], self.data[i:i+self.sequence_length]))
            self.assertEqual(y[i], self.data[i+self.sequence_length])
    
    def test_evaluate_model(self):
        """测试模型评估函数"""
        # 创建测试数据
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        # 计算评估指标
        metrics = evaluate_model(y_true, y_pred)
        
        # 检查指标类型
        self.assertIsInstance(metrics, dict)
        
        # 检查指标键
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # 检查指标值范围
        self.assertTrue(metrics['rmse'] >= 0)
        self.assertTrue(metrics['mae'] >= 0)
        self.assertTrue(metrics['r2'] <= 1)
    
    def test_save_and_load_results(self):
        """测试结果保存和加载函数"""
        # 创建测试结果
        results = {
            'model_name': 'TestModel',
            'metrics': {
                'rmse': 0.316,
                'mae': 0.25,
                'r2': 0.95
            },
            'history': [0.5, 0.3, 0.2, 0.1]
        }
        
        # 保存结果
        save_path = os.path.join(self.test_dir, 'test_results.json')
        save_results(results, save_path)
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(save_path))
        
        # 加载结果
        loaded_results = load_results(save_path)
        
        # 检查加载的结果
        self.assertEqual(loaded_results['model_name'], results['model_name'])
        self.assertEqual(loaded_results['metrics'], results['metrics'])
        self.assertEqual(loaded_results['history'], results['history'])
    
    def test_save_results_with_numpy(self):
        """测试保存包含NumPy数组的结果"""
        # 创建包含NumPy数组的结果
        results = {
            'model_name': 'TestModel',
            'predictions': np.array([1, 2, 3, 4, 5]).tolist(),
            'metrics': {
                'rmse': float(np.float32(0.316)),
                'mae': float(np.float64(0.25))
            }
        }
        
        # 保存结果
        save_path = os.path.join(self.test_dir, 'test_results_numpy.json')
        save_results(results, save_path)
        
        # 加载结果
        loaded_results = load_results(save_path)
        
        # 检查NumPy数组是否正确转换
        self.assertIsInstance(loaded_results['predictions'], list)
        self.assertIsInstance(loaded_results['metrics']['rmse'], float)
        self.assertIsInstance(loaded_results['metrics']['mae'], float)

if __name__ == '__main__':
    unittest.main() 