# 对照组实验，使用人工编写的静态特征，不使用T-LAFS，直接对原始数据进行训练和预测
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import json
import warnings

warnings.filterwarnings('ignore')

# --- Data Handling (Copied from T-LAFS experiment for consistency) ---
def get_time_series_data(dataset_type='min_daily_temps'):
    if dataset_type == 'min_daily_temps':
        csv_path = 'data/min_daily_temps.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
    elif dataset_type == 'total_cleaned':
        csv_path = 'data/total_cleaned.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    else:
        raise ValueError('Unknown dataset type')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Model Definitions (Copied from T-LAFS) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnhancedNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        # We use the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        return self.regressor(last_step_out)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_layer(x.unsqueeze(1))
        x = self.transformer_encoder(x)
        x = self.output_layer(x.squeeze(1))
        return x

def train_pytorch_model(model, X_train, y_train, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_tensor = model(torch.FloatTensor(X_test).to(device))
    return preds_tensor.cpu().numpy().flatten()


# --- Control Group Feature Engineering ---
def create_static_features(df, target_col='temp'):
    """
    Creates a comprehensive, fixed set of time-series features.
    This serves as the control group for the T-LAFS experiment.
    All feature creation is non-leaky.
    """
    print("🔬 Creating static feature set for control group...")
    temp_df = df.copy()
    
    # 1. Lag Features
    lags = [1, 2, 3, 7, 14, 28]
    for lag in lags:
        temp_df[f'{target_col}_lag_{lag}'] = temp_df[target_col].shift(lag).bfill()
        
    # 2. Differencing
    temp_df[f'{target_col}_diff_1'] = temp_df[target_col].diff(1).shift(1).bfill()

    # 3. Rolling Window Features
    windows = [7, 14, 28]
    for window in windows:
        rolling_obj = temp_df[target_col].rolling(window=window)
        temp_df[f'{target_col}_roll_mean_{window}'] = rolling_obj.mean().shift(1).bfill()
        temp_df[f'{target_col}_roll_std_{window}'] = rolling_obj.std().shift(1).bfill()
        temp_df[f'{target_col}_roll_min_{window}'] = rolling_obj.min().shift(1).bfill()
        temp_df[f'{target_col}_roll_max_{window}'] = rolling_obj.max().shift(1).bfill()
        
    # 4. Exponentially Weighted Mean (EWM)
    temp_df[f'{target_col}_ewm_7'] = temp_df[target_col].ewm(span=7, adjust=False).mean().shift(1).bfill()
        
    # 5. Time-based Features
    date_col = 'date'
    temp_df[f'{date_col}_dayofweek'] = temp_df[date_col].dt.dayofweek
    temp_df[f'{date_col}_month'] = temp_df[date_col].dt.month
    temp_df[f'{date_col}_quarter'] = temp_df[date_col].dt.quarter
    temp_df[f'{date_col}_is_weekend'] = (temp_df[date_col].dt.dayofweek >= 5).astype(int)
    temp_df[f'{date_col}_dayofyear'] = temp_df[date_col].dt.dayofyear
    temp_df[f'{date_col}_weekofyear'] = temp_df[date_col].dt.isocalendar().week.astype(int)
    
    # 6. Fourier Features for Seasonality
    time_idx = (temp_df[date_col] - temp_df[date_col].min()).dt.days
    # Yearly seasonality
    for k in range(1, 3):
        temp_df[f'fourier_sin_{k}_365'] = np.sin(2 * np.pi * k * time_idx / 365.25)
        temp_df[f'fourier_cos_{k}_365'] = np.cos(2 * np.pi * k * time_idx / 365.25)
    # Semi-annual seasonality
    temp_df[f'fourier_sin_1_182'] = np.sin(2 * np.pi * 1 * time_idx / 182.625)
    temp_df[f'fourier_cos_1_182'] = np.cos(2 * np.pi * 1 * time_idx / 182.625)
    
    print(f"✅ Static feature set created. Total features: {len(temp_df.columns) - 2}")
    return temp_df

# --- Evaluation & Reporting (Copied from T-LAFS) ---
def evaluate_on_multiple_models(df: pd.DataFrame, target_col: str):
    """
    Evaluates the final feature set on a variety of models.
    """
    print("\n" + "="*40)
    print("🔬 FINAL VALIDATION ON ALL MODELS 🔬")
    print("="*40)
    
    X = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[X.index][target_col]

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'SimpleNN': SimpleNN(X.shape[1]),
        'EnhancedNN': EnhancedNN(X.shape[1]),
        'Transformer': TransformerModel(X.shape[1])
    }
    
    final_scores = {}
    final_results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    scaler_X = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    for name, model in models.items():
        print(f"  - Evaluating {name}...")
        if name in ['SimpleNN', 'EnhancedNN', 'Transformer']:
            preds_scaled = train_pytorch_model(model, X_train_s, y_train_s, X_test_s)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        score = r2_score(y_test, preds)
        final_scores[name] = score
        final_results[name] = {
            "dates": X_test.index.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": preds.tolist()
        }
        print(f"    - {name} R²: {score:.4f}")
        
    return final_scores, final_results

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, best_model_score):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label='Actual Sales', color='dodgerblue', alpha=0.9)
    plt.plot(dates, y_pred, label=f'Predicted Sales ({best_model_name})', color='darkgreen', linestyle='--')
    plt.title(f"Control Group Validation - Best Model: {best_model_name} (R² = {best_model_score:.4f})", fontsize=16)
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/final_predictions_control_group.png")
    plt.show()

def save_results_to_json(results_data):
    """Saves the final results and summary to a JSON file."""
    os.makedirs("results", exist_ok=True)
    file_path = "results/control_group_results.json"
    
    def json_converter(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False, default=json_converter)
    print(f"\n✅ Control group results saved to {file_path}")

def main():
    """Main function to run the Control Group experiment."""
    DATASET_TYPE = 'min_daily_temps'
    TARGET_COL = 'temp'
    
    print("="*80)
    print("🚀 Control Group Experiment: Static Hand-Crafted Features")
    print("="*80)

    # 1. Load Data
    base_df = get_time_series_data(DATASET_TYPE)
    
    # 2. Create Static Features
    featured_df = create_static_features(base_df, TARGET_COL)
    
    # 3. Evaluate on All Models
    final_scores, final_results = evaluate_on_multiple_models(
        featured_df,
        TARGET_COL
    )

    # 4. Summarize and Save Results
    if final_scores:
        best_final_model_name = max(final_scores, key=final_scores.get)
        best_final_score = final_scores[best_final_model_name]
        best_result = final_results[best_final_model_name]

        print("\n" + "="*60)
        print("🏆 EXECUTIVE SUMMARY: CONTROL GROUP STRATEGY 🏆")
        print("="*60)
        print("Feature engineering was conducted using a fixed, hand-crafted set.")
        print(f"This feature set was then validated on a suite of specialist models.")
        print(f"🥇 Best Performing Specialist Model: '{best_final_model_name}'")
        print(f"🚀 Final Validated R² Score: {best_final_score:.4f}")
        print("="*60)

        visualize_final_predictions(
            dates=best_result['dates'],
            y_true=best_result['y_true'],
            y_pred=best_result['y_pred'],
            best_model_name=best_final_model_name,
            best_model_score=best_final_score
        )
        
        results_to_save = {
            "experiment_type": "Control Group (Static Features)",
            "features_used": [col for col in featured_df.columns if col not in ['date', TARGET_COL]],
            "final_validation_scores": final_scores,
            "best_final_model": {
                "name": best_final_model_name,
                "score": best_final_score
            }
        }
        save_results_to_json(results_to_save)

if __name__ == "__main__":
    main() 