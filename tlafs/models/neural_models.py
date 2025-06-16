"""
Neural Network Models for Time Series Forecasting
"""

import math
import torch
import torch.nn as nn
from typing import Optional

class SimpleNN(nn.Module):
    """一个简单的全连接神经网络模型。"""
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

class EnhancedNN(nn.Module):
    """一个带有LSTM和注意力机制的增强型神经网络模型。"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnhancedNN, self).__init__()
        # LSTM层期望输入是 (batch, seq, feature)，所以我们需要在forward中调整维度
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # 注意力机制
        self.attention = nn.Linear(hidden_size, 1)
        # 回归器
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x的原始形状是 (batch, features)，需要增加一个序列维度
        # unsqueeze(1) -> (batch, 1, features)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        # lstm_out 形状: (batch, 1, hidden_size)
        
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attn_weights 形状: (batch, 1, 1)
        
        # 应用注意力权重
        # bmm要求 (b, n, m) * (b, m, p) -> (b, n, p)
        # lstm_out.transpose(1, 2) -> (batch, hidden_size, 1)
        context = torch.bmm(lstm_out.transpose(1, 2), attn_weights).squeeze(2)
        # context 形状: (batch, hidden_size)
        
        return self.regressor(context)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:x.size(1)].transpose(0, 1)

class TransformerModel(nn.Module):
    """一个基于Transformer编码器的模型。"""
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x.unsqueeze(1) 将形状从 (batch, features) 变为 (batch, 1, features)
        x = self.input_layer(x.unsqueeze(1))
        x = self.transformer_encoder(x)
        # x.squeeze(1) 将形状从 (batch, 1, d_model) 变回 (batch, d_model)
        x = self.output_layer(x.squeeze(1))
        return x

# --- 用于预训练的自编码器模型 ---

class MaskedEncoder(nn.Module):
    """掩码编码器，用于从时间序列中学习表征。"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, final_embedding_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, final_embedding_dim)
        )
    
    def forward(self, x):
        _, h_n = self.gru(x)
        # 将双向GRU的最后一个时间步的前向和后向隐藏状态拼接起来
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        embedding = self.projection(last_hidden)
        return embedding

class MaskedDecoder(nn.Module):
    """掩码解码器，用于从表征中重建时间序列。"""
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.expansion_fc = nn.Linear(embedding_dim, hidden_dim * 2)
        self.gru = nn.GRU(
            input_size=hidden_dim * 2, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x_expanded = self.expansion_fc(x)
        # 将单个向量重复seq_len次，以作为GRU的输入序列
        x_repeated = x_expanded.unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.gru(x_repeated)
        reconstruction = self.fc(outputs)
        return reconstruction

class MaskedTimeSeriesAutoencoder(nn.Module):
    """掩码时序自编码器，结合了编码器和解码器。"""
    def __init__(self, input_dim: int, encoder_hidden_dim: int, encoder_layers: int, 
                 decoder_hidden_dim: int, decoder_layers: int, 
                 final_embedding_dim: int, seq_len: int):
        super().__init__()
        self.encoder = MaskedEncoder(
            input_dim=input_dim, hidden_dim=encoder_hidden_dim, num_layers=encoder_layers,
            final_embedding_dim=final_embedding_dim
        )
        self.decoder = MaskedDecoder(
            embedding_dim=final_embedding_dim, hidden_dim=decoder_hidden_dim, 
            output_dim=input_dim, seq_len=seq_len, num_layers=decoder_layers
        )

    def forward(self, x_masked):
        latent_embedding = self.encoder(x_masked)
        reconstruction = self.decoder(latent_embedding)
        return reconstruction 