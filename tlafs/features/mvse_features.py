import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from ..models.mvse import MVSEProbeForecaster

def create_mvse_probe_features_data(df, target_col='temp', hist_len=90, num_lags=14):
    """
    为 MVSEProbeForecaster 创建特征
    """
    df_copy = df.copy()
    if 'dayofweek' not in df_copy:
        df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    if 'month' not in df_copy:
        df_copy['month'] = df_copy['date'].dt.month
    
    target_scaler = MinMaxScaler()
    df_copy['temp_scaled'] = target_scaler.fit_transform(df_copy[[target_col]])
    
    feature_cols = ['temp_scaled', 'dayofweek', 'month']
    
    hist_sequences = []
    lag_features = []
    targets = []
    valid_indices = []
    
    for i in range(hist_len + num_lags, len(df_copy)):
        hist_start = i - hist_len - num_lags
        hist_end = i - num_lags
        hist_seq = df_copy.iloc[hist_start:hist_end][feature_cols].values
        hist_sequences.append(hist_seq)
        
        lag_start = i - num_lags
        lag_end = i
        lag_vals = df_copy.iloc[lag_start:lag_end]['temp_scaled'].values
        lag_features.append(lag_vals)
        
        target_val = df_copy.iloc[i]['temp_scaled']
        targets.append(target_val)
        
        valid_indices.append(df_copy.index[i])
    
    hist_sequences = np.array(hist_sequences, dtype=np.float32)
    lag_features = np.array(lag_features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return hist_sequences, lag_features, targets, valid_indices, target_scaler


def train_mvse_probe_model(hist_sequences, lag_features, targets, 
                          input_dim=3, epochs=30, batch_size=64, 
                          learning_rate=0.001, mask_rate=0.3):
    """
    训练 MVSEProbeForecaster 模型
    """
    train_size = int(len(hist_sequences) * 0.8)
    
    X_hist_train = hist_sequences[:train_size]
    X_lag_train = lag_features[:train_size]
    y_train = targets[:train_size]
    
    X_hist_val = hist_sequences[train_size:]
    X_lag_val = lag_features[train_size:]
    y_val = targets[train_size:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_hist_train = torch.FloatTensor(X_hist_train).to(device)
    X_lag_train = torch.FloatTensor(X_lag_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_hist_val = torch.FloatTensor(X_hist_val).to(device)
    X_lag_val = torch.FloatTensor(X_lag_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    model = MVSEProbeForecaster(
        input_dim=input_dim,
        mask_rate=mask_rate,
        num_lags=lag_features.shape[1]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = model.state_dict()
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_hist_train), batch_size):
            end_idx = min(i + batch_size, len(X_hist_train))
            batch_hist, batch_lag, batch_y = X_hist_train[i:end_idx], X_lag_train[i:end_idx], y_train[i:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_hist, batch_lag)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_hist_val, X_lag_val)
            val_loss = criterion(val_predictions, y_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def create_mvse_features(df, target_col='temp', hist_len=90, num_lags=14, **kwargs):
    """
    为 T-LAFS 框架生成 MVSE 探针特征
    """
    print("  - 🔮 Generating MVSE probe features...")
    
    try:
        hist_sequences, lag_features, targets, valid_indices, target_scaler = create_mvse_probe_features_data(
            df, target_col=target_col, hist_len=hist_len, num_lags=num_lags
        )
        
        if len(hist_sequences) == 0:
            print("  - ⚠️ Not enough data to generate MVSE features.")
            return df, "mvse_features"
        
        model, best_loss = train_mvse_probe_model(hist_sequences, lag_features, targets)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        with torch.no_grad():
            hist_tensor = torch.FloatTensor(hist_sequences).to(device)
            mvse_features = model.mvse_encoder(hist_tensor).cpu().numpy()
            pooling_features = model.mvse_encoder.get_pooling_features(hist_tensor)
            gap_features = pooling_features['gap'].cpu().numpy()
            gmp_features = pooling_features['gmp'].cpu().numpy()
        
        feature_names = []
        all_features = []
        
        mvse_cols = [f"mvse_feat_{i}" for i in range(min(16, mvse_features.shape[1]))]
        feature_names.extend(mvse_cols)
        all_features.append(mvse_features[:, :len(mvse_cols)])
        
        gap_stats = np.column_stack([
            gap_features.mean(axis=1), gap_features.std(axis=1),
            gap_features.max(axis=1), gap_features.min(axis=1)
        ])
        gap_stat_cols = ['mvse_gap_mean', 'mvse_gap_std', 'mvse_gap_max', 'mvse_gap_min']
        feature_names.extend(gap_stat_cols)
        all_features.append(gap_stats)
        
        gmp_stats = np.column_stack([
            gmp_features.mean(axis=1), gmp_features.std(axis=1),
            gmp_features.max(axis=1), gmp_features.min(axis=1)
        ])
        gmp_stat_cols = ['mvse_gmp_mean', 'mvse_gmp_std', 'mvse_gmp_max', 'mvse_gmp_min']
        feature_names.extend(gmp_stat_cols)
        all_features.append(gmp_stats)
        
        final_features = np.concatenate(all_features, axis=1)
        
        features_df = pd.DataFrame(final_features, index=valid_indices, columns=feature_names)
        
        print(f"  - ✅ MVSE features generated: {len(feature_names)} features, train loss: {best_loss:.6f}")
        
        return df.join(features_df.shift(1)), "mvse_features"
        
    except Exception as e:
        print(f"  - ❌ MVSE feature generation failed: {e}")
        return df, "mvse_features_failed" 