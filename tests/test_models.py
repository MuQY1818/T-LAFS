"""
测试TLAFS模型模块
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import math
from tlafs.models.neural_models import (
    SimpleNN,
    EnhancedNN,
    TransformerModel,
    PositionalEncoding
)
from tlafs.models.probe import ProbeForecaster

class TestNeuralModels(unittest.TestCase):
    """测试神经网络模型"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.batch_size = 32
        self.seq_length = 14
        self.input_size = 10
        self.hidden_size = 64
        self.flat_input_size = self.seq_length * self.input_size
    
    def test_simple_nn(self):
        """测试SimpleNN模型"""
        model = SimpleNN(input_size=self.flat_input_size)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        x_flat = x.view(self.batch_size, -1)
        output = model(x_flat)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_enhanced_nn(self):
        """测试EnhancedNN模型"""
        model = EnhancedNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2
        )
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_transformer_model(self):
        """测试Transformer模型"""
        model = TransformerModel(
            input_size=self.input_size,
            d_model=self.hidden_size,
            nhead=4,
            num_encoder_layers=2
        )
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_probe_forecaster(self):
        """测试ProbeForecaster模型"""
        model = ProbeForecaster(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            num_layers=2,
            nhead=4
        )
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))

class TestModelTraining(unittest.TestCase):
    """测试模型训练过程"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.batch_size = 32
        self.seq_length = 14
        self.input_size = 10
        self.hidden_size = 64
        self.flat_input_size = self.seq_length * self.input_size
        
        # 创建训练数据
        self.X_train = torch.randn(self.batch_size, self.seq_length, self.input_size)
        self.y_train = torch.randn(self.batch_size, 1)
        
        # 创建测试数据
        self.X_test = torch.randn(self.batch_size, self.seq_length, self.input_size)
        self.y_test = torch.randn(self.batch_size, 1)
    
    def test_simple_nn_training(self):
        """测试SimpleNN训练过程"""
        model = SimpleNN(input_size=self.flat_input_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        X_train = torch.randn(self.batch_size, self.seq_length, self.input_size)
        y_train = torch.randn(self.batch_size, 1)
        X_train_flat = X_train.view(self.batch_size, -1)
        output = model(X_train_flat)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
    
    def test_enhanced_nn_training(self):
        """测试EnhancedNN训练过程"""
        model = EnhancedNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 前向传播
        output = model(self.X_train)
        loss = criterion(output, self.y_train)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 检查损失值
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)

if __name__ == '__main__':
    unittest.main() 