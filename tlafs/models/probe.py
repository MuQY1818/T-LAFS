"""
Probe Forecaster Models for TLAFS
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class AgentAttentionProbe(nn.Module):
    """代理注意力探针"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.fc(attn_output)

class ProbeForecaster(nn.Module):
    """探针预测器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 nhead: int = 4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.probe = AgentAttentionProbe(hidden_dim, hidden_dim, nhead)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.probe(x)
        return self.fc(x[:, -1])

def create_probe_model(config: dict) -> ProbeForecaster:
    """
    创建探针预测器模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        配置好的探针预测器模型
    """
    return ProbeForecaster(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        nhead=config['nhead']
    ) 