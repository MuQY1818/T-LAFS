import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MVSEEmbedding(nn.Module):
    """
    Multi-View Sequential Embedding (MVSE) 模块
    
    将时间序列 (B, T, D) 编码成全局低维特征向量 (B, d_out)
    使用三种不同的池化策略：GAP、GMP、MaskedGAP
    
    Args:
        d_input (int): 输入特征维度 D
        d_hidden (int): 隐藏层维度
        d_out (int): 输出特征维度
        mask_rate (float): 随机遮罩比例，范围 [0, 1)
        dropout (float): Dropout 比例，默认 0.1
    """
    
    def __init__(self, d_input, d_hidden, d_out, mask_rate=0.3, dropout=0.1):
        super(MVSEEmbedding, self).__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.mask_rate = mask_rate
        
        # 拼接后的特征维度：3种池化 × 输入维度
        self.concat_dim = 3 * d_input
        
        # LayerNorm 用于归一化拼接后的特征
        self.layer_norm = nn.LayerNorm(self.concat_dim)
        
        # 前馈网络：两层线性层 + ReLU + Dropout
        self.feedforward = nn.Sequential(
            nn.Linear(self.concat_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Sigmoid()  # 最终使用 Sigmoid 激活
        )
        
    def global_average_pooling(self, x):
        """
        全局平均池化 (GAP)
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        # 在时间维度 T 上求平均
        return torch.mean(x, dim=1)  # (B, T, D) -> (B, D)
    
    def global_max_pooling(self, x):
        """
        全局最大池化 (GMP)
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        # 在时间维度 T 上求最大值
        return torch.max(x, dim=1)[0]  # (B, T, D) -> (B, D)，[0]取值，[1]取索引
    
    def masked_global_average_pooling(self, x):
        """
        随机遮罩平均池化 (MaskedGAP)
        
        类似 Dropout，随机将部分时间步置零，然后对剩余值求平均
        
        Args:
            x (torch.Tensor): 输入张量 (B, T, D)
            
        Returns:
            torch.Tensor: 池化结果 (B, D)
        """
        B, T, D = x.shape
        
        if self.training and self.mask_rate > 0:
            # 训练模式下应用随机遮罩
            # 生成遮罩：1表示保留，0表示遮罩
            mask = torch.rand(B, T, 1, device=x.device) > self.mask_rate  # (B, T, 1)
            
            # 应用遮罩
            masked_x = x * mask.float()  # (B, T, D)
            
            # 计算每个样本实际保留的时间步数量
            valid_counts = mask.sum(dim=1, keepdim=True).float()  # (B, 1, 1)
            valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
            
            # 计算遮罩后的平均值
            masked_sum = torch.sum(masked_x, dim=1)  # (B, D)
            masked_avg = masked_sum / valid_counts.squeeze(-1)  # (B, D)
            
            return masked_avg
        else:
            # 推理模式下或mask_rate=0时，直接使用全局平均池化
            return self.global_average_pooling(x)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入时间序列 (B, T, D)
            
        Returns:
            torch.Tensor: 编码后的特征向量 (B, d_out)
        """
        # 检查输入维度
        if len(x.shape) != 3:
            raise ValueError(f"输入应为3维张量 (B, T, D)，但得到形状: {x.shape}")
        
        B, T, D = x.shape
        if D != self.d_input:
            raise ValueError(f"输入特征维度应为 {self.d_input}，但得到 {D}")
        
        # 1. 应用三种池化策略
        gap_features = self.global_average_pooling(x)      # (B, D)
        gmp_features = self.global_max_pooling(x)          # (B, D)
        masked_gap_features = self.masked_global_average_pooling(x)  # (B, D)
        
        # 2. 拼接三种池化结果
        concat_features = torch.cat([
            gap_features,           # 全局平均
            gmp_features,           # 全局最大
            masked_gap_features     # 遮罩平均
        ], dim=1)  # (B, 3*D)
        
        # 3. LayerNorm 归一化
        normalized_features = self.layer_norm(concat_features)  # (B, 3*D)
        
        # 4. 前馈网络降维
        output = self.feedforward(normalized_features)  # (B, d_out)
        
        return output
    
    def get_pooling_features(self, x):
        """
        获取三种池化的中间特征，用于分析和可视化
        
        Args:
            x (torch.Tensor): 输入时间序列 (B, T, D)
            
        Returns:
            dict: 包含三种池化结果的字典
        """
        gap_features = self.global_average_pooling(x)
        gmp_features = self.global_max_pooling(x)
        masked_gap_features = self.masked_global_average_pooling(x)
        
        return {
            'gap': gap_features,
            'gmp': gmp_features,
            'masked_gap': masked_gap_features,
            'concat': torch.cat([gap_features, gmp_features, masked_gap_features], dim=1)
        }


class MVSEProbeForecaster(nn.Module):
    """
    基于 MVSEEmbedding 的时间序列预测模型
    
    结合了多视角序列编码和传统的滞后特征
    """
    
    def __init__(self, input_dim, mvse_d_hidden=128, mvse_d_out=32, 
                 num_lags=14, mask_rate=0.3, final_hidden=64):
        super(MVSEProbeForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.num_lags = num_lags
        self.mvse_d_out = mvse_d_out
        
        # MVSE 编码器：将历史序列编码为全局特征
        self.mvse_encoder = MVSEEmbedding(
            d_input=input_dim,
            d_hidden=mvse_d_hidden,
            d_out=mvse_d_out,
            mask_rate=mask_rate
        )
        
        # 最终预测层：结合 MVSE 特征和滞后特征
        total_features = mvse_d_out + num_lags  # MVSE特征 + 滞后特征
        self.predictor = nn.Sequential(
            nn.Linear(total_features, final_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(final_hidden, final_hidden // 2),
            nn.ReLU(),
            nn.Linear(final_hidden // 2, 1)
        )
        
    def forward(self, hist_seq, lag_features):
        """
        前向传播
        
        Args:
            hist_seq (torch.Tensor): 历史序列 (B, T, D)
            lag_features (torch.Tensor): 滞后特征 (B, num_lags)
            
        Returns:
            torch.Tensor: 预测结果 (B, 1)
        """
        # 1. 使用 MVSE 编码历史序列
        mvse_features = self.mvse_encoder(hist_seq)  # (B, mvse_d_out)
        
        # 2. 拼接 MVSE 特征和滞后特征
        combined_features = torch.cat([mvse_features, lag_features], dim=1)  # (B, total_features)
        
        # 3. 最终预测
        prediction = self.predictor(combined_features)  # (B, 1)
        
        return prediction 