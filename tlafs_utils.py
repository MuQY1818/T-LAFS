import pandas as pd
import numpy as np
import torch
from mvse_embedding import MVSEEmbedding
from mvse_probe_integration import create_mvse_probe_features, train_mvse_probe_model

def generate_mvse_features_for_tlafs(df, target_col='temp', hist_len=90, num_lags=14):
    """
    为 T-LAFS 框架生成 MVSE 探针特征
    
    Args:
        df (pd.DataFrame): 输入数据
        target_col (str): 目标列名
        hist_len (int): 历史序列长度
        num_lags (int): 滞后特征数量
        
    Returns:
        pd.DataFrame: 添加了 MVSE 特征的数据框
    """
    print("🔮 生成 MVSE 探针特征用于 T-LAFS...")
    
    # 创建特征
    hist_sequences, lag_features, targets, valid_indices, target_scaler = create_mvse_probe_features(
        df, target_col=target_col, hist_len=hist_len, num_lags=num_lags
    )
    
    # 训练模型
    model, best_loss = train_mvse_probe_model(
        hist_sequences, lag_features, targets,
        epochs=50, mask_rate=0.3
    )
    
    # 生成特征
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 获取 MVSE 编码特征
        hist_tensor = torch.FloatTensor(hist_sequences).to(device)
        mvse_features = model.mvse_encoder(hist_tensor).cpu().numpy()
        
        # 获取池化特征用于分析
        pooling_features = model.mvse_encoder.get_pooling_features(hist_tensor)
        gap_features = pooling_features['gap'].cpu().numpy()
        gmp_features = pooling_features['gmp'].cpu().numpy()
        masked_gap_features = pooling_features['masked_gap'].cpu().numpy()
    
    # 创建特征 DataFrame
    feature_names = []
    all_features = []
    
    # 1. MVSE 主要特征
    mvse_cols = [f"mvse_feat_{i}" for i in range(mvse_features.shape[1])]
    feature_names.extend(mvse_cols)
    all_features.append(mvse_features)
    
    # 2. 池化特征的统计摘要
    gap_stats = np.column_stack([
        gap_features.mean(axis=1),
        gap_features.std(axis=1),
        gap_features.max(axis=1),
        gap_features.min(axis=1)
    ])
    gap_stat_cols = ['mvse_gap_mean', 'mvse_gap_std', 'mvse_gap_max', 'mvse_gap_min']
    feature_names.extend(gap_stat_cols)
    all_features.append(gap_stats)
    
    gmp_stats = np.column_stack([
        gmp_features.mean(axis=1),
        gmp_features.std(axis=1),
        gmp_features.max(axis=1),
        gmp_features.min(axis=1)
    ])
    gmp_stat_cols = ['mvse_gmp_mean', 'mvse_gmp_std', 'mvse_gmp_max', 'mvse_gmp_min']
    feature_names.extend(gmp_stat_cols)
    all_features.append(gmp_stats)
    
    # 合并所有特征
    final_features = np.concatenate(all_features, axis=1)
    
    # 创建 DataFrame
    features_df = pd.DataFrame(
        final_features,
        index=valid_indices,
        columns=feature_names
    )
    
    print(f"✅ MVSE 探针特征生成完成:")
    print(f"   - 特征数量: {len(feature_names)}")
    print(f"   - 样本数量: {len(features_df)}")
    print(f"   - 训练损失: {best_loss:.6f}")
    
    # 使用 shift(1) 避免数据泄露
    return df.join(features_df.shift(1)) 