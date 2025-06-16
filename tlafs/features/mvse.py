"""
Multi-View Sequential Embedding (MVSE) Feature Engineering
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class MVSEEmbedding(nn.Module):
    """
    Multi-View Sequential Embedding model
    
    This model learns embeddings from time series data using multiple views
    and masking techniques.
    """
    
    def __init__(self, d_input: int, d_hidden: int, d_out: int, 
                 mask_rate: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.mask_rate = mask_rate
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layers for different views
        self.lstm1 = nn.LSTM(d_input, d_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(d_input, d_hidden, batch_first=True)
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(d_hidden, num_heads=4)
        self.attention2 = nn.MultiheadAttention(d_hidden, num_heads=4)
        
        # Output layers
        self.fc1 = nn.Linear(d_hidden, d_out)
        self.fc2 = nn.Linear(d_hidden, d_out)
        
    def global_average_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pooling operation"""
        return torch.mean(x, dim=1)
        
    def global_max_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Global max pooling operation"""
        return torch.max(x, dim=1)[0]
        
    def masked_global_average_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Masked global average pooling operation"""
        mask = torch.rand_like(x) > self.mask_rate
        masked_x = x * mask
        return torch.sum(masked_x, dim=1) / (torch.sum(mask, dim=1) + 1e-6)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MVSE model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_input)
            
        Returns:
            Tuple of (view1_embedding, view2_embedding)
        """
        # First view
        lstm1_out, _ = self.lstm1(x)
        attn1_out, _ = self.attention1(lstm1_out, lstm1_out, lstm1_out)
        view1 = self.fc1(self.dropout(attn1_out))
        
        # Second view
        lstm2_out, _ = self.lstm2(x)
        attn2_out, _ = self.attention2(lstm2_out, lstm2_out, lstm2_out)
        view2 = self.fc2(self.dropout(attn2_out))
        
        return view1, view2
        
    def get_pooling_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get pooled features from the model
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated pooling features
        """
        view1, view2 = self.forward(x)
        
        # Apply different pooling operations
        avg_pool1 = self.global_average_pooling(view1)
        max_pool1 = self.global_max_pooling(view1)
        masked_avg_pool1 = self.masked_global_average_pooling(view1)
        
        avg_pool2 = self.global_average_pooling(view2)
        max_pool2 = self.global_max_pooling(view2)
        masked_avg_pool2 = self.masked_global_average_pooling(view2)
        
        # Concatenate all features
        return torch.cat([
            avg_pool1, max_pool1, masked_avg_pool1,
            avg_pool2, max_pool2, masked_avg_pool2
        ], dim=1)

class MVSEEncoder(nn.Module):
    """多视图序列编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

def create_mvse_features(
    view1: pd.DataFrame,
    view2: pd.DataFrame,
    target_col: str,
    embedding_dim: int = 64
) -> np.ndarray:
    """
    创建多视图序列嵌入特征
    
    参数:
        view1: 第一个视图的DataFrame
        view2: 第二个视图的DataFrame
        target_col: 目标列名
        embedding_dim: 嵌入维度
        
    返回:
        np.ndarray: 嵌入特征
    """
    # 确保两个视图具有相同的索引
    assert view1.index.equals(view2.index), "两个视图必须具有相同的索引"
    
    # 提取目标列数据
    data1 = view1[target_col].values
    data2 = view2[target_col].values
    
    # 转换为张量
    x1 = torch.FloatTensor(data1).unsqueeze(-1)
    x2 = torch.FloatTensor(data2).unsqueeze(-1)
    
    # 创建编码器
    encoder = MVSEEncoder(input_dim=1, hidden_dim=32, output_dim=embedding_dim)
    
    # 生成嵌入
    with torch.no_grad():
        emb1 = encoder(x1)
        emb2 = encoder(x2)
    
    # 合并嵌入
    combined_emb = (emb1 + emb2) / 2
    
    return combined_emb.numpy()

def generate_mvse_features(df: pd.DataFrame, target_col: str, 
                          hist_len: int = 90, num_lags: int = 14) -> pd.DataFrame:
    """
    Generate MVSE features for a time series dataset
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        hist_len: Length of historical window
        num_lags: Number of lag features to generate
        
    Returns:
        DataFrame with MVSE features
    """
    # Implementation will be moved from mvse_embedding.py
    pass 