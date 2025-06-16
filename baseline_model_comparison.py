# baseline_model_comparison.py 用于对比不同模型在不做特征工程的情况下，直接对原始数据进行训练和预测的效果
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# --- 1. Data Loading ---
def get_raw_sales_data():
    """Loads only the sales column from the CSV."""
    csv_path = 'data/min_daily_temps.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}. Please run t_lafs_demo.py first if it's missing.")
    
    df = pd.read_csv(csv_path)
    df.rename(columns={'Date': 'date', 'Temp': 'sales'}, inplace=True) # Rename to be generic
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df[['date', 'sales']]

# --- 2. Sequence Creation ---
def create_sequences(data, sequence_length):
    """Creates sequences and corresponding labels from time series data."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# --- 3. Model Definitions (Copied from t_lafs_demo.py for direct comparison) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

class EnhancedNN(nn.Module): # LSTM + Attention
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(EnhancedNN, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context = self.attention(lstm_out)
        return self.regressor(context)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # Transformer expects (Batch, Seq, Feature)
        x = self.input_layer(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] # Use output of the last sequence element
        return self.output_layer(x)

def train_pytorch_model(model, train_loader, epochs, lr, device):
    """Generic training function for a PyTorch model."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    print(f"   - Training {model.__class__.__name__} for {epochs} epochs...")
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

# --- 4. Main Execution ---
def main():
    print("="*80)
    print("🚀 Running Baseline Model Comparison")
    print("This script trains 6 models directly on raw sales data sequences.")
    print("="*80)

    # --- Configuration ---
    SEQUENCE_LENGTH = 7
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TRAIN_TEST_SPLIT = 0.8

    # --- Load and Prepare Data ---
    df = get_raw_sales_data()
    sales_data = df['sales'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data)
    X_seq, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    
    # --- Split Data ---
    train_size = int(len(X_seq) * TRAIN_TEST_SPLIT)
    
    # A. Data for Sequence Models (LSTM, Transformer)
    X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # B. Data for Tabular Models (LGBM, RF, XGB, SimpleNN)
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
    
    print(f"✅ Data prepared:")
    print(f"   - Sequence data shape (for LSTM/Transformer): {X_train_seq.shape}")
    print(f"   - Flattened data shape (for other models):  {X_train_flat.shape}")

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Train and Evaluate Tabular Models ---
    print("\n--- Training Tabular Models ---")
    tabular_models = {
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42)
    }
    for name, model in tabular_models.items():
        print(f"   - Training {name}...")
        model.fit(X_train_flat, y_train.ravel())
        preds_scaled = model.predict(X_test_flat)
        results[name] = scaler.inverse_transform(preds_scaled.reshape(-1, 1))

    # --- Train and Evaluate PyTorch Models ---
    print("\n--- Training PyTorch Models ---")
    # 1. Tabular NN Model
    X_train_flat_t = torch.FloatTensor(X_train_flat)
    y_train_t = torch.FloatTensor(y_train)
    train_loader_flat = DataLoader(TensorDataset(X_train_flat_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    
    nn_model = SimpleNN(input_size=SEQUENCE_LENGTH)
    nn_model = train_pytorch_model(nn_model, train_loader_flat, EPOCHS, LEARNING_RATE, device)
    nn_model.eval()
    with torch.no_grad():
        preds_scaled = nn_model(torch.FloatTensor(X_test_flat).to(device)).cpu().numpy()
    results["SimpleNN"] = scaler.inverse_transform(preds_scaled)

    # 2. Sequence NN Models
    X_train_seq_t = torch.FloatTensor(X_train_seq)
    train_loader_seq = DataLoader(TensorDataset(X_train_seq_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    
    sequence_models = {
        "EnhancedNN (LSTM+Attn)": EnhancedNN(input_size=1),
        "Transformer": TransformerModel(input_size=1)
    }
    for name, model_instance in sequence_models.items():
        model = train_pytorch_model(model_instance, train_loader_seq, EPOCHS, LEARNING_RATE, device)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.FloatTensor(X_test_seq).to(device)).cpu().numpy()
        results[name] = scaler.inverse_transform(preds_scaled)

    # --- Display Results ---
    print("\n\n" + "="*50)
    print("📊 BASELINE MODELS PERFORMANCE (NO FEATURE ENGINEERING)")
    print("="*50)
    print(f"{'Model':<25} | {'R² Score':>10} | {'MAE':>12} | {'RMSE':>12}")
    print("-"*68)

    y_test_actual = scaler.inverse_transform(y_test)
    for name, predictions in results.items():
        r2 = r2_score(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        print(f"{name:<25} | {r2:>10.4f} | {mae:>12,.2f} | {rmse:>12,.2f}")
    print("="*68)

if __name__ == "__main__":
    main() 