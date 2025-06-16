"""
测试TLAFS特征工程模块
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tlafs.features.mvse import create_mvse_features
from tlafs.experiments.control import create_static_features
from tlafs.experiments.sonata import create_sonata_features

class TestFeatureEngineering(unittest.TestCase):
    """测试特征工程函数"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(len(dates))
        })
        self.target_col = 'value'
    
    def test_static_features(self):
        """测试静态特征创建"""
        # 创建特征
        df_features = create_static_features(self.df, self.target_col)
        
        # 检查时间特征
        self.assertIn('year', df_features.columns)
        self.assertIn('month', df_features.columns)
        self.assertIn('day', df_features.columns)
        self.assertIn('dayofweek', df_features.columns)
        
        # 检查滞后特征
        self.assertIn('value_lag_1', df_features.columns)
        self.assertIn('value_lag_7', df_features.columns)
        self.assertIn('value_lag_14', df_features.columns)
        
        # 检查滚动统计特征
        self.assertIn('value_rolling_mean_7', df_features.columns)
        self.assertIn('value_rolling_std_7', df_features.columns)
        self.assertIn('value_rolling_min_7', df_features.columns)
        self.assertIn('value_rolling_max_7', df_features.columns)
        
        # 检查数据完整性
        self.assertFalse(df_features.isnull().any().any())
    
    def test_sonata_features(self):
        """测试Sonata特征创建"""
        # 创建特征
        df_features = create_sonata_features(self.df, self.target_col)
        
        # 检查基本时间特征
        self.assertIn('year', df_features.columns)
        self.assertIn('month', df_features.columns)
        self.assertIn('day', df_features.columns)
        self.assertIn('dayofweek', df_features.columns)
        self.assertIn('quarter', df_features.columns)
        
        # 检查滞后特征
        self.assertIn('lag_1', df_features.columns)
        self.assertIn('lag_7', df_features.columns)
        self.assertIn('lag_14', df_features.columns)
        self.assertIn('lag_30', df_features.columns)
        
        # 检查滚动统计特征
        self.assertIn('rolling_mean_7', df_features.columns)
        self.assertIn('rolling_std_7', df_features.columns)
        self.assertIn('rolling_skew_7', df_features.columns)
        self.assertIn('rolling_kurt_7', df_features.columns)
        
        # 检查季节性特征
        self.assertIn('month_sin', df_features.columns)
        self.assertIn('month_cos', df_features.columns)
        self.assertIn('day_sin', df_features.columns)
        self.assertIn('day_cos', df_features.columns)
        
        # 检查趋势特征
        self.assertIn('trend', df_features.columns)
        self.assertIn('trend_squared', df_features.columns)
        
        # 检查数据完整性
        self.assertFalse(df_features.isnull().any().any())
    
    def test_mvse_features(self):
        """测试MVSE特征创建"""
        # 创建两个视图
        view1 = self.df.copy()
        view2 = self.df.copy()
        view2['value'] = view2['value'] * 2  # 修改第二个视图
        
        # 创建MVSE特征
        features = create_mvse_features(view1, view2, self.target_col)
        
        # 检查特征维度
        self.assertEqual(features.shape[0], len(self.df))
        self.assertEqual(features.shape[1], 64)  # 假设嵌入维度为64
        
        # 检查特征值范围
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(np.abs(features) < 1e10))  # 检查是否有极端值

if __name__ == '__main__':
    unittest.main() 