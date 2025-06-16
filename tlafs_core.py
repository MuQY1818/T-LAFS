import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
import google.generativeai as genai
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from datetime import datetime
import random
from pytorch_tabnet.tab_model import TabNetRegressor
from catboost import CatBoostRegressor
import re
from collections import defaultdict
from probe_forecaster_utils import train_probe_forecaster, generate_probe_features, ProbeForecaster, PositionalEncoding, AgentAttentionProbe
<<<<<<< HEAD:clp_probe_experiment.py
from mvse_tlafs_integration import generate_mvse_features_for_tlafs
# --- NEW: Import MVSE training components ---
from mvse_probe_integration import create_mvse_probe_features, train_mvse_probe_model, MVSEProbeForecaster
=======
from tlafs_utils import generate_mvse_features_for_tlafs
>>>>>>> feature/sonata-refactor:tlafs_core.py

warnings.filterwarnings('ignore')

# --- Global Variables & Setup ---
gemini_model = None
N_STABILITY_RUNS = 1

def setup_api_client():
    """Initializes the Google Gemini API client."""
    global gemini_model
    try:
        api_key = "sk-O4mZi7nZvCpp11x0UgbrIN5dr6jdNmTocD9ADso1S1ZWJzdL"
        base_url = "https://api.openai-proxy.org/google"
        genai.configure(
            api_key=api_key,
            transport="rest",
            client_options={"api_endpoint": base_url},
        )
        generation_config = {"response_mime_type": "application/json"}
        gemini_model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-05-20',
            generation_config=generation_config
        )
        genai.list_models() 
        print("✅ Gemini client initialized and connection successful.")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        exit()

# --- Data Handling ---
def get_time_series_data(dataset_type='min_daily_temps'):
    if dataset_type == 'total_cleaned':
        csv_path = 'data/total_cleaned.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    else: # Default to min_daily_temps
        csv_path = 'data/min_daily_temps.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
        
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Model Definitions ---
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
        self.attention = nn.Linear(hidden_size, 1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.bmm(lstm_out.transpose(1, 2), attn_weights).squeeze(2)
        return self.regressor(context)

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
    criterion, optimizer = nn.SmoothL1Loss(), optim.Adam(model.parameters(), lr=0.001)

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

# --- Autoencoder Models for Pre-training ---
class MaskedEncoder(nn.Module):
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
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        embedding = self.projection(last_hidden)
        return embedding

class MaskedDecoder(nn.Module):
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
        x_repeated = x_expanded.unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.gru(x_repeated)
        reconstruction = self.fc(outputs)
        return reconstruction

class MaskedTimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoder_hidden_dim: int, encoder_layers: int, decoder_hidden_dim: int, decoder_layers: int, final_embedding_dim: int, seq_len: int):
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

def visualize_autoencoder_reconstruction(model, data_loader, scaler, results_dir, mask_ratio, n_samples=3):
    model.eval()
    original_seqs_tensor = next(iter(data_loader))[0][:n_samples]
    
    torch.manual_seed(42)
    noise = torch.rand(original_seqs_tensor.shape[0], original_seqs_tensor.shape[1])
    unmasked_indices = noise > mask_ratio
    x_masked_input = original_seqs_tensor.clone()
    x_masked_input[~unmasked_indices] = 0

    with torch.no_grad():
        reconstructed_seqs_tensor = model(x_masked_input)
        
    original_seqs = original_seqs_tensor.cpu().numpy()
    reconstructed_seqs = reconstructed_seqs_tensor.cpu().numpy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(15, 5 * n_samples), sharex=True)
    if n_samples == 1: axes = [axes]
    fig.suptitle('Masked Autoencoder Pre-training: Original vs. Reconstructed', fontsize=16)

    for i in range(n_samples):
        original_unscaled = scaler.inverse_transform(original_seqs[i])
        reconstructed_unscaled = scaler.inverse_transform(reconstructed_seqs[i])
        
        time_steps = np.arange(original_unscaled.shape[0])
        axes[i].plot(time_steps, original_unscaled[:, 0], label='Original Data (Ground Truth)', color='dodgerblue', zorder=2)
        axes[i].plot(time_steps, reconstructed_unscaled[:, 0], label='Reconstructed by Model', color='orangered', linestyle='--', zorder=3)
        
        masked_time_steps = time_steps[~unmasked_indices[i]]
        for t in masked_time_steps:
             axes[i].axvspan(t - 0.5, t + 0.5, color='gray', alpha=0.2, zorder=1)
        
        axes[i].plot([], [], color='gray', alpha=0.2, linewidth=10, label=f'Masked Area (Input is Zero)')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].legend()

    plt.xlabel('Time Step in Window')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(results_dir, "autoencoder_reconstruction.png")
    plt.savefig(plot_path)
    print(f"✅ Autoencoder reconstruction plot saved to {plot_path}")
    plt.close(fig)

# --- Probe & Evaluation ---
def probe_feature_set(df: pd.DataFrame, target_col: str):
    """
    NEW PROBE: Multi-Model Fusion Probe.
    Evaluates a feature set's general quality by testing it on two diverse models:
    1. LightGBM (a tree-based model)
    2. SimpleNN (a neural network)
    The final score is a blend of their performances, rewarding universally good features.
    """
    # 1. Prepare Data
    df_feat = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[df_feat.index][target_col]
    X = df_feat

    if X.empty or y.empty or len(X) < 20: # Added length check for robustness
        print("  - ⚠️ Warning: Feature set is empty or too small. Returning poor score.")
        return {"primary_score": 0.0, "r2_lgbm": 0.0, "r2_nn": 0.0, "num_features": 0, "importances": None}

    # Use a simple time-series split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) < 1 or len(X_test) < 1:
        print("  - ⚠️ Warning: Not enough data for train/test split. Returning poor score.")
        return {"primary_score": 0.0, "r2_lgbm": 0.0, "r2_nn": 0.0, "num_features": X.shape[1], "importances": None}

    # --- Model 1: LightGBM ---
    lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=50, verbosity=-1, n_jobs=1)
    lgb_model.fit(X_train, y_train)
    preds_lgbm = lgb_model.predict(X_test)
    score_lgbm = r2_score(y_test, preds_lgbm)
    # --- NEW: Extract feature importances ---
    importances = pd.Series(lgb_model.feature_importances_, index=X_train.columns)

    # --- Model 2: SimpleNN ---
    scaler_x = MinMaxScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    nn_model = SimpleNN(X_train.shape[1])
    preds_nn_scaled = train_pytorch_model(nn_model, X_train_s, y_train_s, X_test_s)
    preds_nn = scaler_y.inverse_transform(preds_nn_scaled.reshape(-1, 1)).flatten()
    score_nn = r2_score(y_test, preds_nn)
    
    # --- Fusion Logic ---
    # We can use a simple average or a weighted average. Let's use a 50/50 blend.
    # We clip scores at 0 to prevent negative rewards from overly punishing the agent.
    primary_score = 0.5 * max(0.0, score_lgbm) + 0.5 * max(0.0, score_nn)

    return {
        "primary_score": primary_score,
        "r2_lgbm": score_lgbm,
        "r2_nn": score_nn,
        "num_features": X.shape[1],
        "importances": importances
    }

def visualize_final_predictions(dates, y_true, y_pred, best_model_name, probe_name, best_model_metrics, results_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    plt.plot(dates, y_true, label='Actual Values', color='dodgerblue', alpha=0.9)
    plt.plot(dates, y_pred, label=f'Predicted Values ({best_model_name})', color='orangered', linestyle='--')
    title = (f"Final Validation (Probe: {probe_name}) - Best Performing Model: {best_model_name}\n"
             f"R²: {best_model_metrics['r2']:.4f}  |  MAE: {best_model_metrics['mae']:.4f}  |  RMSE: {best_model_metrics['rmse']:.4f}")
    plt.title(title, fontsize=14)
    plt.legend()
    plot_path = os.path.join(results_dir, f"final_predictions_probe_{probe_name}.png")
    plt.savefig(plot_path)
    print(f"✅ Final predictions plot saved to {plot_path}")
    plt.show()

def save_results_to_json(results_data, probe_name, results_dir):
    """Saves the final results and summary to a JSON file in the specified directory."""
    file_path = os.path.join(results_dir, f"tlafs_results_probe_{probe_name}.json")
    
    # Custom handler for numpy types that the default JSON encoder can't handle.
    def json_converter(o):
        if isinstance(o, (np.floating, np.float64, np.float32)):
            if np.isinf(o) or np.isneginf(o): return str(o)
            if np.isnan(o): return None
            return float(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.bool_)):
            return bool(o)
        if isinstance(o, (np.ndarray)):
            return o.tolist()
        if isinstance(o, pd.Series):
            return o.to_dict()
        return str(o) if hasattr(o, '__str__') else f"<non-serializable: {type(o)}>"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False, default=json_converter)
    print(f"\n✅ Results and configuration saved to {file_path}")

def pretrain_mvse_probe(df, target_col='temp', hist_len=90, num_lags=14, epochs=100, patience=15):
    """
    NEW: Pre-trains the MVSEProbeForecaster model with early stopping.
    """
    print("\n🧠 Pre-training MVSE Probe Model for feature generation...")
    try:
        # 1. Create dataset
        hist_sequences, lag_features, targets, _, target_scaler = create_mvse_probe_features(
            df, target_col=target_col, hist_len=hist_len, num_lags=num_lags
        )

        if len(hist_sequences) == 0:
            print("  - ⚠️ Not enough data to pre-train MVSE model. Skipping.")
            return None, None

        # 2. Train model with more epochs and early stopping
        print(f"  - Training on {len(hist_sequences)} samples for up to {epochs} epochs...")
        model, best_loss = train_mvse_probe_model(
            hist_sequences, lag_features, targets,
            epochs=epochs,
            patience=patience, # Use early stopping
            mask_rate=0.3,
            batch_size=64
        )

        print(f"  - ✅ MVSE pre-training complete. Final validation loss: {best_loss:.6f}")
        return model, target_scaler
    except Exception as e:
        import traceback
        print(f"  - ❌ Failed to pre-train MVSE model: {e}\n{traceback.format_exc()}")
        return None, None

def evaluate_on_multiple_models(df: pd.DataFrame, target_col: str, probe_name: str):
    """
    Evaluates the final feature set on a variety of models.
    This now uses the static, correct execute_plan method from TLAFS_Algorithm.
    """
    print(f"Evaluating final feature set on various models (Probe was: {probe_name})...")
    
    # This plan is not executed, it's just for reference if needed.
    # The dataframe `df` is already the one with the best features.
    
    X = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[X.index][target_col]

    # For TabNet, we should identify categorical features
    # A simple heuristic: columns with a small number of unique values that are not float.
    categorical_cols = [col for col in X.columns if (X[col].dtype in ['object', 'category', 'int64'] and X[col].nunique() < 50) and col not in ['date', target_col]]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]
    cat_dims = [X[col].nunique() for col in categorical_cols]
    print(f"  - Identified {len(categorical_cols)} categorical features for TabNet: {categorical_cols}")

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'SVR': SVR(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'KNeighbors': KNeighborsRegressor(),
        'TabNet': TabNetRegressor(cat_idxs=categorical_indices, cat_dims=cat_dims, seed=42, verbose=0),
        'SimpleNN': SimpleNN(X.shape[1]),
        'EnhancedNN': EnhancedNN(X.shape[1]),
        'Transformer': TransformerModel(X.shape[1])
    }
    
    final_metrics = {}
    final_results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Create a validation set from the training set for early stopping in TabNet
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train.values, y_train.values.reshape(-1,1), test_size=0.2, random_state=42, shuffle=False)
    
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Use a separate scaler for the target to avoid data leakage issues if we were to scale y_test
    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    models_requiring_scaling = ['SVR', 'Ridge', 'Lasso', 'KNeighbors']

    for name, model in models.items():
        print(f"  - Evaluating {name}...")
        if name in ['SimpleNN', 'EnhancedNN', 'Transformer']:
            preds_scaled = train_pytorch_model(model, X_train_s, y_train_s, X_test_s)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        elif name == 'TabNet':
            model.fit(
                X_train=X_train_np, y_train=y_train_np,
                eval_set=[(X_val_np, y_val_np)],
                patience=15, max_epochs=100,
                batch_size=1024,
                eval_metric=['rmse']
            )
            preds = model.predict(X_test.values).flatten()
        elif name in models_requiring_scaling:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else: # LGBM, RF, XGB, CatBoost
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        final_metrics[name] = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
        final_results[name] = {
            "dates": X_test.index.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": preds.tolist()
        }
        print(f"    - {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        
    return final_metrics, final_results

class ExperienceReplayBuffer:
    """A buffer to store and sample past experiences for the LLM agent."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, adopted):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        experience = {
            "state": state, 
            "action": action, 
            "reward": reward,
            "adopted": adopted
        }
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, n_good=2, n_bad=1):
        if len(self.buffer) < 2: # Don't sample if buffer is too small
            return ""
            
        good_experiences = [exp for exp in self.buffer if exp['adopted']]
        bad_experiences = [exp for exp in self.buffer if not exp['adopted']]

        good_experiences.sort(key=lambda x: x.get('reward', 0), reverse=True)

        good_samples = random.sample(good_experiences, min(len(good_experiences), n_good))
        bad_samples = random.sample(bad_experiences, min(len(bad_experiences), n_bad))
        
        if not good_samples and not bad_samples:
            return ""

        prompt_str = "\n\n--- IN-CONTEXT LEARNING: EXAMPLES FROM PAST ATTEMPTS ---\n"
        prompt_str += "Learn from these past successes and failures to make a better plan.\n"

        if good_samples:
            prompt_str += "\n**Successful Plans (Adopted with High Reward):**\n"
            for exp in good_samples:
                r2 = exp['state'].get('R2 Score (raw)', -1.0)
                num_feats = exp['state'].get('Number of Features', 'N/A')
                # --- OPTIMIZATION: Summarize feature list in examples ---
                summarized_features = TLAFS_Algorithm.summarize_feature_list(exp['state']['Available Features'])
                plan = exp['action']
                reward = exp['reward']
                prompt_str += f"- PAST_CONTEXT: R²={r2:.3f}, #Feats={num_feats}, Feats={summarized_features}. PLAN: {plan}. OUTCOME: ADOPTED, reward={reward:.3f}.\n"
        
        if bad_samples:
            prompt_str += "\n**Failed Plans (Rejected):**\n"
            for exp in bad_samples:
                r2 = exp['state'].get('R2 Score (raw)', -1.0)
                num_feats = exp['state'].get('Number of Features', 'N/A')
                # --- OPTIMIZATION: Summarize feature list in examples ---
                summarized_features = TLAFS_Algorithm.summarize_feature_list(exp['state']['Available Features'])
                plan = exp['action']
                prompt_str += f"- PAST_CONTEXT: R²={r2:.3f}, #Feats={num_feats}, Feats={summarized_features}. PLAN: {plan}. OUTCOME: REJECTED.\n"
            
        return prompt_str

    def __len__(self):
        return len(self.buffer)

class TLAFS_Algorithm:
    """
    Time-series Language-augmented Feature Search (T-LAFS) Algorithm.
    This class orchestrates the automated feature engineering process, now framed as an RL problem.
    """
    def __init__(self, base_df, target_col, n_iterations=5, acceptance_threshold=0.01, results_dir="."):
        self.base_df = base_df
        self.target_col = target_col
        self.n_iterations = n_iterations
        self.acceptance_threshold = acceptance_threshold
        self.history = []
        self.best_score = -np.inf
        self.best_plan = []
        self.best_df = None
        self.results_dir = results_dir
        self.experience_buffer = ExperienceReplayBuffer(capacity=20)
        
        # --- NEW: Centralized model initialization and pre-training ---
        self._initialize_and_pretrain_models()

    def _initialize_and_pretrain_models(self):
        """A new private method to handle all model pre-training and setup."""
        print("\nEnriching data with time features for a smarter autoencoder...")
        self.base_df['dayofweek'] = self.base_df['date'].dt.dayofweek
        self.base_df['month'] = self.base_df['date'].dt.month
        self.base_df['weekofyear'] = self.base_df['date'].dt.isocalendar().week.astype(int)
        self.base_df['is_weekend'] = (self.base_df['date'].dt.dayofweek >= 5).astype(int)
        
        TLAFS_Algorithm.pretrain_cols_static = [self.target_col, 'dayofweek', 'month', 'weekofyear', 'is_weekend']
        TLAFS_Algorithm.target_col_static = self.target_col  # 添加缺失的类属性
        
        pretrained_models_dir = "pretrained_models"
        os.makedirs(pretrained_models_dir, exist_ok=True)
        
        print("\n🧠 Setting up PROBE FORECASTER model...")
        self.probe_model_path = os.path.join(pretrained_models_dir, "probe_forecaster.pth")
        self.probe_config = {
            'seq_len': 365, 'd_model': 64, 'nhead': 4, 'num_agents': 8,
            'num_lags': 14, 'epochs': 150, 'patience': 25, 'batch_size': 32,
            'learning_rate': 0.0005
        }
        if not os.path.exists(self.probe_model_path):
             print(f"  -> ProbeForecaster model not found. It will be trained if needed by dependent models.")
        else:
             print(f"  -> Found pre-trained ProbeForecaster model at {self.probe_model_path}")
        
        TLAFS_Algorithm.probe_config = self.probe_config
        TLAFS_Algorithm.probe_model_path = self.probe_model_path
        
        self.pretrain_all_embedders()

        print("\n🧠 Pre-training Meta-Forecast Models...")
        self.meta_forecast_models = {}
        meta_features = ['lag1', 'lag7', 'lag30']
        
        df_for_meta = self.base_df[[self.target_col]].copy()
        for lag in [1, 7, 30]:
            df_for_meta[f'lag{lag}'] = df_for_meta[self.target_col].shift(lag)
        df_for_meta.dropna(inplace=True)
        
        X_meta = df_for_meta[meta_features].values
        y_meta = df_for_meta[[self.target_col]].values
        
        train_size_meta = int(len(X_meta) * 0.8)
        X_train_meta, y_train_meta = X_meta[:train_size_meta], y_meta[:train_size_meta]
        
        scaler_x_meta = MinMaxScaler()
        X_train_meta_s = scaler_x_meta.fit_transform(X_train_meta)
        
        scaler_y_meta = MinMaxScaler()
        y_train_meta_s = scaler_y_meta.fit_transform(y_train_meta)
        
        meta_input_size = len(meta_features)
        models_to_train = {
            'SimpleNN_meta': SimpleNN(input_size=meta_input_size),
            'EnhancedNN_meta': EnhancedNN(input_size=meta_input_size)
        }
        
        for name, model in models_to_train.items():
            print(f"  - Training {name}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            dataset = TensorDataset(torch.FloatTensor(X_train_meta_s), torch.FloatTensor(y_train_meta_s))
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            criterion, optimizer = nn.SmoothL1Loss(), optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(30):
                for inputs, targets in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, targets.to(device))
                    loss.backward()
                    optimizer.step()
            model.eval()
            self.meta_forecast_models[name] = model.to('cpu')
        
        self.meta_scalers = {'x': scaler_x_meta, 'y': scaler_y_meta}
        TLAFS_Algorithm.meta_forecast_models = self.meta_forecast_models
        TLAFS_Algorithm.meta_scalers = self.meta_scalers

        # --- NEW: Pre-train the MVSE model ---
        self.mvse_probe_model, self.mvse_scaler = pretrain_mvse_probe(self.base_df, self.target_col)
        TLAFS_Algorithm.mvse_probe_model = self.mvse_probe_model
        TLAFS_Algorithm.mvse_scaler = self.mvse_scaler

    def pretrain_embedder(self, df_to_pretrain_on: pd.DataFrame, df_to_scale_on: pd.DataFrame, window_size: int, config: dict):
        """Pre-trains a masked autoencoder with validation and early stopping, and returns the trained ENCODER part."""
        
        print("  - 检查ProbeForecaster模型是否可用...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        probe_model_path = self.probe_model_path
        use_perceptual_loss = os.path.exists(probe_model_path)
        
        perceptual_model = None
        input_embed_layer = None
        pos_encoder = None
        perceptual_weight = 0.5 # Hyperparameter
            
        if use_perceptual_loss:
            print("  - 尝试加载ProbeForecaster以启用感知损失...")
            try:
                probe_config = self.probe_config
                perceptual_model = AgentAttentionProbe(
                    d_model=probe_config['d_model'], nhead=probe_config['nhead'],
                    num_agents=probe_config['num_agents']
                ).to(device)
                
                full_probe_forecaster_model = ProbeForecaster(
                    input_dim=3, d_model=probe_config['d_model'], nhead=probe_config['nhead'],
                    num_agents=probe_config['num_agents'], num_lags=probe_config['num_lags'], num_exog=2
                ).to(device)
                
                full_probe_forecaster_model.load_state_dict(torch.load(probe_model_path, map_location=device))
                
                perceptual_model.load_state_dict(full_probe_forecaster_model.attention_probe.state_dict())
                perceptual_model.eval()
                
                input_embed_layer = full_probe_forecaster_model.input_embedding.to(device)
                input_embed_layer.eval()

                pos_encoder = PositionalEncoding(d_model=probe_config['d_model']).to(device)
                print("  - ✅ ProbeForecaster加载成功，感知损失已启用。")

            except Exception as e:
                print(f"  - ⚠️ 警告：加载ProbeForecaster失败。原因: {e}")
                print("  - 将回退到仅使用基础重建损失。")
                use_perceptual_loss = False
        
        if not use_perceptual_loss:
            print("  - 将使用基础重建损失进行训练。")

        input_dim = df_to_pretrain_on.shape[1]
        print(f"  - 准备数据进行掩码自编码器预训练 (窗口: {window_size}, 输入维度: {input_dim})...")
        
        scaler = MinMaxScaler()
        scaler.fit(df_to_scale_on)
        series_scaled = scaler.transform(df_to_pretrain_on)
        
        sequences = [series_scaled[i:i+window_size] for i in range(len(series_scaled) - window_size + 1)]
        
        if not sequences:
            print("  - ⚠️ 警告：数据不足，无法进行预训练。返回未训练的编码器。")
            return MaskedEncoder(input_dim=input_dim, hidden_dim=config['encoder_hidden_dim'], num_layers=config['encoder_layers'], final_embedding_dim=config['final_embedding_dim']), scaler

        split_idx = int(len(sequences) * 0.8)
        train_sequences, val_sequences = (sequences[:split_idx], sequences[split_idx:]) if split_idx > 0 else (sequences, [])
        
        if not val_sequences:
            print("  - ⚠️ 警告：验证集为空，将不使用早停。")

        train_dataset = TensorDataset(torch.FloatTensor(np.array(train_sequences)))
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(np.array(val_sequences))), batch_size=config['batch_size'], shuffle=False) if val_sequences else None

        autoencoder = MaskedTimeSeriesAutoencoder(
            input_dim=input_dim, encoder_hidden_dim=config['encoder_hidden_dim'], encoder_layers=config['encoder_layers'],
            decoder_hidden_dim=config['decoder_hidden_dim'], decoder_layers=config['decoder_layers'],
            final_embedding_dim=config['final_embedding_dim'], seq_len=window_size
        ).to(device)
        criterion_recon = nn.SmoothL1Loss()
        criterion_perceptual = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'])

        print(f"  - 在 {len(train_dataset)} 个序列上预训练自编码器，最多 {config['epochs']} 个轮次...")
        best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None

        for epoch in range(config['epochs']):
            autoencoder.train()
            total_train_loss = 0
            for i, (batch,) in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                noise = torch.rand(batch.shape, device=device)
                unmasked_indices = noise > config['mask_ratio']
                x_masked_input = batch.clone()
                x_masked_input[~unmasked_indices] = 0
                
                reconstruction = autoencoder(x_masked_input)
                loss_mask = ~unmasked_indices
                loss_r = criterion_recon(reconstruction[loss_mask], batch[loss_mask])
                loss = loss_r

                if use_perceptual_loss and window_size >= 365:
                    with torch.no_grad():
                        original_seq_for_perceptual = batch[:, -365:, :3]
                        recon_seq_for_perceptual = torch.cat([reconstruction[:, -365:, 0:1], batch[:, -365:, 1:3]], dim=-1)
                        original_embedded = pos_encoder(input_embed_layer(original_seq_for_perceptual))
                        recon_embedded = pos_encoder(input_embed_layer(recon_seq_for_perceptual))
                        original_features = perceptual_model(original_embedded)
                        recon_features = perceptual_model(recon_embedded)
                        loss_p = criterion_perceptual(recon_features, original_features)
                        loss = (1 - perceptual_weight) * loss_r + perceptual_weight * loss_p
                
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / (i + 1) if train_loader else 0

            if not val_loader:
                if (epoch + 1) % 10 == 0: print(f"    Epoch [{epoch+1}/{config['epochs']}], Train Loss: {avg_train_loss:.6f}")
                continue

            autoencoder.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i, (batch,) in enumerate(val_loader):
                    batch = batch.to(device)
                    noise = torch.rand(batch.shape, device=device)
                    unmasked_indices = noise > config['mask_ratio']
                    x_masked_input = batch.clone(); x_masked_input[~unmasked_indices] = 0
                    reconstruction = autoencoder(x_masked_input)
                    loss_mask = ~unmasked_indices
                    loss_r_val = criterion_recon(reconstruction[loss_mask], batch[loss_mask])
                    loss = loss_r_val

                    if use_perceptual_loss and window_size >= 365:
                        original_seq_for_perceptual = batch[:, -365:, :3]
                        recon_seq_for_perceptual = torch.cat([reconstruction[:, -365:, 0:1], batch[:, -365:, 1:3]], dim=-1)
                        original_embedded = pos_encoder(input_embed_layer(original_seq_for_perceptual))
                        recon_embedded = pos_encoder(input_embed_layer(recon_seq_for_perceptual))
                        original_features = perceptual_model(original_embedded)
                        recon_features = perceptual_model(recon_embedded)
                        loss_p_val = criterion_perceptual(recon_features, original_features)
                        loss = (1 - perceptual_weight) * loss_r_val + perceptual_weight * loss_p_val
                        
                    if torch.isfinite(loss):
                        total_val_loss += loss.item()

            avg_val_loss = total_val_loss / (i + 1) if val_loader else 0
            print(f"    Epoch [{epoch+1}/{config['epochs']}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve, best_model_state = avg_val_loss, 0, autoencoder.state_dict()
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= config['patience']:
                print(f"    -- 早停触发于第 {epoch+1} 轮。 --")
                break
        
        if best_model_state:
            autoencoder.load_state_dict(best_model_state)
        
        visualize_autoencoder_reconstruction(autoencoder.to('cpu'), train_loader, scaler, self.results_dir, config['mask_ratio'])

        return autoencoder.encoder.to("cpu"), scaler

    def pretrain_all_embedders(self):
        """A new wrapper method to handle the logic that was in __init__."""
        print("\n🧠 Pre-training or LOADING MULTI-SCALE Masked Autoencoders...")
        
        pretrained_models_dir = "pretrained_models"
        os.makedirs(pretrained_models_dir, exist_ok=True)
        
        self.pretrained_encoders = {}
        self.embedder_scalers = {}
        embedding_window_sizes = [90, 365, 730]
        input_dim = len(TLAFS_Algorithm.pretrain_cols_static)

        pretrain_config = {
            'encoder_hidden_dim': 256, 'encoder_layers': 4,
            'decoder_hidden_dim': 128, 'decoder_layers': 2,
            'final_embedding_dim': 64, 
            'epochs': 100, 'batch_size': 64, 'patience': 15,
            'learning_rate': 0.001, 'mask_ratio': 0.4
        }

        df_for_pretraining = self.base_df[TLAFS_Algorithm.pretrain_cols_static]
        train_size = int(len(df_for_pretraining) * 0.8)
        df_for_scaling = df_for_pretraining.iloc[:train_size]
        
        for window_size in embedding_window_sizes:
            encoder_path = os.path.join(pretrained_models_dir, f"encoder_ws{window_size}.pth")
            scaler_path = os.path.join(pretrained_models_dir, f"scaler_ws{window_size}.joblib")

            if os.path.exists(encoder_path) and os.path.exists(scaler_path):
                print(f"\n--- Loading pre-trained model for window size: {window_size} days ---")
                encoder = MaskedEncoder(
                    input_dim=input_dim,
                    hidden_dim=pretrain_config['encoder_hidden_dim'],
                    num_layers=pretrain_config['encoder_layers'],
                    final_embedding_dim=pretrain_config['final_embedding_dim']
                )
                encoder.load_state_dict(torch.load(encoder_path))
                encoder.eval() 
                scaler = joblib.load(scaler_path)
                
                self.pretrained_encoders[window_size] = encoder
                self.embedder_scalers[window_size] = scaler
                print(f"   ✅ Loaded encoder and scaler from {pretrained_models_dir}")
            else:
                print(f"\n--- No pre-trained model found. Training for window size: {window_size} days ---")
                encoder, scaler = self.pretrain_embedder( # Call the method
                    df_for_pretraining,       
                    df_for_scaling,           
                    window_size=window_size,
                    config=pretrain_config
                )
                
                torch.save(encoder.state_dict(), encoder_path)
                joblib.dump(scaler, scaler_path)
                print(f"   ✅ Pre-training complete. Saved new encoder and scaler to {pretrained_models_dir}")
                
                self.pretrained_encoders[window_size] = encoder
                self.embedder_scalers[window_size] = scaler

        TLAFS_Algorithm.pretrained_encoders = self.pretrained_encoders
        TLAFS_Algorithm.embedder_scalers = self.embedder_scalers

    @staticmethod
    def summarize_feature_list(features: list) -> list:
        """Compresses a list of feature names into a more compact, readable format for LLMs."""
        
        groups = defaultdict(list)
        ungrouped = []

        # --- OPTIMIZATION: Added more patterns for summarization ---
        patterns = {
            "embed": re.compile(r"^(embed_)(\d+)(_.*)$"),
            "fourier": re.compile(r"^(fourier_(?:sin|cos)_)(\d+)(_.*)$"),
            "probe": re.compile(r"^(probe_feat_)(\d+)$"),
            "mvse": re.compile(r"^(mvse_feat_)(\d+)$"),
        }

        for f in features:
            matched = False
            for p_name, pattern in patterns.items():
                match = pattern.match(f)
                if match:
                    # Create a group key. For patterns with 3 groups like embed, use group 1 and 3. For others, use group 1.
                    key_part_3 = match.group(3) if len(match.groups()) > 2 else ''
                    group_key = f"{match.group(1)}*{key_part_3}"
                    groups[group_key].append(int(match.group(2)))
                    matched = True
                    break
            if not matched:
                ungrouped.append(f)

        # Reconstruct the list
        summarized_list = ungrouped
        for key, numbers in groups.items():
            if len(numbers) > 3:
                min_n, max_n = min(numbers), max(numbers)
                # Reconstruct name from key, e.g. 'embed_*_LE_Yearly' -> 'embed_0-31_LE_Yearly'
                summarized_name = key.replace('*', f'{min_n}-{max_n}')
                summarized_list.append(summarized_name)
            else:
                # Not enough to summarize, add them back individually
                for n in numbers:
                    summarized_list.append(key.replace('*', str(n)))
        
        return sorted(summarized_list)

    @staticmethod
    def execute_plan(df: pd.DataFrame, plan: list):
        """
        Executes a feature engineering plan on a given dataframe.
        This is the single, authoritative static method for all feature generation.
        It contains all non-leaky feature generation logic.
        """
        temp_df = df.copy()

        # --- FIX: Ensure base features for embeddings exist ---
        # The pre-trained embedder relies on a specific set of columns.
        # We must ensure they are present in the dataframe before any plan execution,
        # making this function robust for both search and re-evaluation modes.
        required_time_cols = ['dayofweek', 'month', 'weekofyear', 'is_weekend']
        if not all(col in temp_df.columns for col in required_time_cols):
            print("  - Ensuring base time features for embeddings are present...")
            if 'dayofweek' not in temp_df.columns:
                temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
            if 'month' not in temp_df.columns:
                temp_df['month'] = temp_df['date'].dt.month
            if 'weekofyear' not in temp_df.columns:
                temp_df['weekofyear'] = temp_df['date'].dt.isocalendar().week.astype(int)
            if 'is_weekend' not in temp_df.columns:
                temp_df['is_weekend'] = (temp_df['date'].dt.dayofweek >= 5).astype(int)
        
        target_col = TLAFS_Algorithm.target_col_static
        
        for step in plan:
            op = step.get("operation")
            # BUG FIX: Handle 'feature', 'base_feature', 'column', or 'target' to align with LLM.
            # --- NEW ROBUSTNESS FIX --- Also handle 'on' as a synonym for 'feature'
            feature = step.get("feature") or step.get("base_feature") or step.get("column") or step.get("target") or step.get("on")

            # A more robust way to generate a new column name
            def get_new_col_name(base_feature, id_suggestion):
                if base_feature and id_suggestion.startswith(base_feature):
                    return id_suggestion
                return f"{base_feature}_{id_suggestion}" if base_feature else id_suggestion

            try:
                # --- Basic Time-Series Features ---
                if op == "create_lag":
                    days = step.get("days", 1)
                    if isinstance(days, list) and days: days = int(days[0])
                    id = step.get("id", f"lag{days}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].shift(days).ffill().fillna(0)

                elif op == "create_diff":
                    periods = step.get("periods", 1)
                    if isinstance(periods, list) and periods: periods = int(periods[0])
                    id = step.get("id", f"diff{periods}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].diff(periods).shift(1).ffill().fillna(0)

                elif op in ["create_rolling_mean", "create_rolling_std", "create_rolling_skew", "create_rolling_kurt", "create_rolling_min", "create_rolling_max"]:
                    window = step.get("window", 7)
                    if isinstance(window, list) and window: window = int(window[0])
                    op_name = op.split('_')[2]
                    id = step.get("id", f"roll_{op_name}{window}")
                    new_col_name = get_new_col_name(feature, id)
                    roll_op = getattr(temp_df[feature].rolling(window=window), op_name)
                    temp_df[new_col_name] = roll_op().shift(1).ffill().fillna(0)

                elif op == "create_ewm":
                    span = step.get("span", 7)
                    if isinstance(span, list) and span: span = int(span[0])
                    id = step.get("id", f"ewm{span}")
                    new_col_name = get_new_col_name(feature, id)
                    temp_df[new_col_name] = temp_df[feature].ewm(span=span, adjust=False).mean().shift(1).ffill().fillna(0)

                # --- Calendar and Seasonality Features ---
                elif op == "create_time_features":
                    if pd.api.types.is_datetime64_any_dtype(temp_df[feature]):
                        default_parts = ["dayofweek", "month", "is_weekend"]
                        for part in step.get("extract", default_parts):
                            if part == "dayofweek": temp_df[f'{feature}_dayofweek'] = temp_df[feature].dt.dayofweek
                            elif part == "month": temp_df[f'{feature}_month'] = temp_df[feature].dt.month
                            elif part == "quarter": temp_df[f'{feature}_quarter'] = temp_df[feature].dt.quarter
                            elif part == "is_weekend": temp_df[f'{feature}_is_weekend'] = (temp_df[feature].dt.dayofweek >= 5).astype(int)
                            elif part == "dayofyear": temp_df[f'{feature}_dayofyear'] = temp_df[feature].dt.dayofyear
                            elif part == "weekofyear": temp_df[f'{feature}_weekofyear'] = temp_df[feature].dt.isocalendar().week.astype(int)
                    else:
                        print(f"  - ⚠️ Warning: Skipped time_features on non-datetime column '{feature}'.")

                elif op == "create_fourier_features":
                    # This operation is always based on the 'date' column for seasonality.
                    # We ignore the 'feature' parameter from the LLM to prevent errors.
                    if pd.api.types.is_datetime64_any_dtype(temp_df['date']):
                        # DEFENSIVE: Provide defaults for period and order to handle incomplete LLM plans
                        period = float(step.get("period", 365.25))
                        if isinstance(period, list) and period: period = float(period[0])
                        order = int(step.get("order", 4))
                        if isinstance(order, list) and order: order = int(order[0])
                        time_idx = (temp_df['date'] - temp_df['date'].min()).dt.days
                        for k in range(1, order + 1):
                            temp_df[f'fourier_sin_{k}_{int(period)}'] = np.sin(2 * np.pi * k * time_idx / period)
                            temp_df[f'fourier_cos_{k}_{int(period)}'] = np.cos(2 * np.pi * k * time_idx / period)
                    else:
                        print(f"  - ⚠️ Warning: Skipped fourier because 'date' column is not datetime.")

                # --- Learned & Interaction Features ---
                elif op == "create_learned_embedding":
                    # --- NEW: Multi-scale embedding logic ---
                    window = step.get("window", 90) # Default to 90 if not specified
                    if isinstance(window, list) and window: window = int(window[0])
                    embedder = TLAFS_Algorithm.pretrained_encoders.get(window)
                    scaler = TLAFS_Algorithm.embedder_scalers.get(window)

                    if embedder and scaler:
                        # --- BUG FIX: Ensure required columns exist RIGHT BEFORE they are used ---
                        if not all(col in temp_df.columns for col in required_time_cols):
                            print("  - Ensuring base time features for embeddings are present...")
                            if 'dayofweek' not in temp_df.columns: temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
                            if 'month' not in temp_df.columns: temp_df['month'] = temp_df['date'].dt.month
                            if 'weekofyear' not in temp_df.columns: temp_df['weekofyear'] = temp_df['date'].dt.isocalendar().week.astype(int)
                            if 'is_weekend' not in temp_df.columns: temp_df['is_weekend'] = (temp_df['date'].dt.dayofweek >= 5).astype(int)
                            
                        id = step.get("id", f"embed{window}")
                        pretrain_cols = TLAFS_Algorithm.pretrain_cols_static
                        
                        print(f"  - Generating multi-variate embedding (win:{window}) from {len(pretrain_cols)} features...")
                        
                        df_for_embedding = temp_df[pretrain_cols]
                        scaled_features = scaler.transform(df_for_embedding)
                        
                        sequences = np.array([scaled_features[i:i+window] for i in range(len(scaled_features) - window + 1)])
                        
                        if sequences.size == 0:
                            print(f"  - ⚠️ Not enough data for embedding with window {window}. Skipping.")
                            continue
                            
                        tensor = torch.FloatTensor(sequences)
                        with torch.no_grad():
                            # The new encoder directly outputs the low-dimensional embedding.
                            # No more complex pooling logic needed here.
                            embeddings = embedder(tensor).numpy()
                            
                        valid_indices = temp_df.index[window-1:]
                        # CRITICAL FIX: Use shape[1] (number of features) not shape[0] (number of sequences)
                        cols = [f"embed_{i}_{id}" for i in range(embeddings.shape[1])]
                        embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)
                        
                        # BUG FIX: Drop existing columns with the same names before joining to prevent overlap errors.
                        existing_cols_to_drop = [col for col in cols if col in temp_df.columns]
                        if existing_cols_to_drop:
                            temp_df.drop(columns=existing_cols_to_drop, inplace=True)

                        temp_df = temp_df.join(embed_df)
                        # LEAKAGE FIX: Shift the embedding features by 1 to ensure they only contain past information.
                        temp_df[cols] = temp_df[cols].shift(1).ffill().fillna(0)
                    else:
                        print(f"  - ⚠️ Embedder for window {window} not available. Skipping.")

                elif op == "create_interaction":
                    features_in_plan = step.get("features")
                    if not features_in_plan:
                        f1 = step.get("feature1", step.get("feature_a"))
                        f2 = step.get("feature2", step.get("feature_b"))
                        if f1 and f2:
                            features_in_plan = [f1, f2]

                    # --- FIX: Fuzzy matching to find actual feature names ---
                    actual_features = []
                    all_cols = temp_df.columns.tolist()
                    
                    if isinstance(features_in_plan, list):
                        for f_plan in features_in_plan:
                            # Correct common typos
                            if "weekkofyear" in f_plan:
                                f_plan = f_plan.replace("weekkofyear", "weekofyear")

                            if f_plan in all_cols:
                                actual_features.append(f_plan)
                            else:
                                # Try to find the closest match
                                try:
                                    # Use a library for fuzzy matching if available, otherwise simple check
                                    # For now, let's assume a simple check for rolling features might work
                                    # e.g., 'roll_mean7' in 'rolling_mean_7'
                                    simplified_plan = f_plan.replace("_", "").replace("roll", "rolling")
                                    
                                    best_match = None
                                    for col in all_cols:
                                        simplified_col = col.replace("_", "")
                                        if simplified_plan == simplified_col:
                                            best_match = col
                                            break # Found exact simplified match
                                    
                                    if best_match:
                                        print(f"  - ⓘ Fuzzy matched '{f_plan}' to '{best_match}'")
                                        actual_features.append(best_match)
                                    else:
                                        actual_features.append(f_plan) # Append original if no match found, will fail validation below

                                except Exception:
                                     actual_features.append(f_plan) # Fallback

                    if isinstance(actual_features, list) and len(actual_features)==2 and all(f in temp_df.columns for f in actual_features):
                        id = step.get('id', 'new')
                        new_col_name = f"{actual_features[0]}_x_{actual_features[1]}_{id}"
                        temp_df[new_col_name] = temp_df[actual_features[0]] * temp_df[actual_features[1]]
                    else:
                        print(f"  - ⚠️ Skipping interaction for invalid/missing features: {features_in_plan}")

                elif op == "create_forecast_feature":
                    model_name = step.get("model_name")
                    id = step.get("id", model_name)
                    
                    if model_name in TLAFS_Algorithm.meta_forecast_models:
                        print(f"  - Generating forecast feature from pre-trained model: {model_name}...")
                        model = TLAFS_Algorithm.meta_forecast_models[model_name]
                        scalers = TLAFS_Algorithm.meta_scalers
                        
                        # 1. Create the specific lag features this model needs from the target
                        meta_features = ['lag1', 'lag7', 'lag30']
                        df_for_pred = pd.DataFrame(index=temp_df.index)
                        for lag in [1, 7, 30]:
                            df_for_pred[f'lag{lag}'] = temp_df[target_col].shift(lag)
                        
                        # Keep track of original index to align series later
                        df_for_pred.dropna(inplace=True)

                        if not df_for_pred.empty:
                            # 2. Scale features using the saved scaler
                            X_pred_s = scalers['x'].transform(df_for_pred[meta_features].values)
                            
                            # 3. Predict
                            model.eval()
                            with torch.no_grad():
                                preds_s = model(torch.FloatTensor(X_pred_s)).numpy()
                            
                            # 4. Inverse transform predictions
                            preds = scalers['y'].inverse_transform(preds_s).flatten()
                            
                            # 5. Add to dataframe, aligning with the index before NaNs were dropped
                            new_col_name = id # The LLM is asked to provide a unique ID
                            pred_series = pd.Series(preds, index=df_for_pred.index)
                            temp_df[new_col_name] = pred_series
                            
                            # LEAKAGE FIX: Shift the forecast feature to ensure it's available at prediction time
                            temp_df[new_col_name] = temp_df[new_col_name].shift(1).ffill().fillna(0)
                        else:
                            print(f"  - ⚠️ Not enough data to generate forecast feature for {model_name}. Skipping.")
                    else:
                        print(f"  - ⚠️ Forecast model '{model_name}' not found. Skipping.")
                
                elif op == "create_probe_features":
                    print("  - Generating probe features from pre-trained model...")
                    config = TLAFS_Algorithm.probe_config
                    model_path = TLAFS_Algorithm.probe_model_path
                    # This function takes the df, adds features, and returns the new df
                    temp_df = generate_probe_features(temp_df, config, model_path)
                    print("  - ✅ Probe features generated.")

                elif op == "create_mvse_features":
                    # --- 防御性检查：避免重复添加 ---
                    if 'mvse_feat_0' in temp_df.columns:
                        print("  - ⚠️ MVSE features already exist. Skipping operation.")
                        continue
                    
                    # --- NEW: Use the pre-trained model ---
                    model = TLAFS_Algorithm.mvse_probe_model
                    scaler = TLAFS_Algorithm.mvse_scaler
                    
                    if model and scaler:
                        temp_df = generate_mvse_features_for_tlafs(temp_df, target_col, model, scaler)
                    else:
                        print("  - ⚠️ MVSE pre-trained model not available. Skipping operation.")

                elif op == "delete_feature":
                    feature_to_delete = step.get("feature")

                    if feature_to_delete:
                        # Original logic to delete a single named feature
                        if feature_to_delete in temp_df.columns and feature_to_delete not in ['date', target_col]:
                            temp_df.drop(columns=[feature_to_delete], inplace=True)
                            print(f"  - Successfully deleted feature: {feature_to_delete}")
                    else:
                        print("  - ⚠️ Warning: 'delete_feature' called without 'feature'. Skipping.")
                
            except Exception as e:
                import traceback
                print(f"  - ❌ ERROR executing step {step}. Error: {e}\n{traceback.format_exc()}")
        
        return temp_df
        
    def get_plan_from_llm(self, context_prompt, iteration_num, max_iterations):
        """
        Dynamically generates a system prompt based on the current iteration stage
        to guide the LLM through a curriculum of feature engineering.
        """
        
        # --- Curriculum Learning Stages ---
        stage = "advanced" # Default stage
        if (iteration_num / max_iterations) < 0.4:
            stage = "basic"

        # --- OPTIMIZATION: More concise tool definitions ---
        base_prompt = f"""You are a Data Scientist RL agent. Your goal is to create a feature engineering plan to maximize the Fusion R^2 score.
Your response MUST be a valid JSON list of operations: `[ {{"operation": "op_name", ...}}, ... ]`.
The target column is '{self.target_col}'.
"""
        basic_tools = """
# *** STAGE 1: BASIC FEATURE ENGINEERING ***
# GOAL: Create foundational features. Think about different windows/spans (e.g., 7, 14, 28).
# AVAILABLE TOOLS:
- Lag/Diff: {{"operation": "create_lag", "on": "...", "days": ...}} OR {{"operation": "create_diff", ...}}
- Rolling Stats: {{"operation": "create_rolling_{{mean|std|min|max}}", "on": "...", "window": ...}}
- Exp. WM: {{"operation": "create_ewm", "on": "...", "span": ...}}
- Fourier: {{"operation": "create_fourier_features", "period": 365.25, "order": 4}}
- Interaction: {{"operation": "create_interaction", "features": ["feat1", "feat2"]}}
- Delete: {{"operation": "delete_feature", "feature": "feature_name"}}
"""

        advanced_tools = """
# *** STAGE 2: ADVANCED & LEARNED FEATURES ***
# GOAL: Use powerful pre-trained models. Combine them with the best basic features.
# AVAILABLE TOOLS (all basic tools are still available):
- Learned Embeddings: {{"operation": "create_learned_embedding", "window": {{90|365|730}} }}
- Meta-Forecasts: {{"operation": "create_forecast_feature", "model_name": "{{SimpleNN_meta|EnhancedNN_meta}}"}}
- Attention Probe (High-Dim): {{"operation": "create_probe_features"}}
"""
        
        # --- NEW: Dynamically add MVSE tool if not already used ---
        # Check if any column in the best_df starts with 'mvse_feat_'
        mvse_already_exists = any(col.startswith('mvse_feat_') for col in self.best_df.columns)
        
        if not mvse_already_exists:
            advanced_tools += """- MVSE Probe (Efficient): {{"operation": "create_mvse_features"}} # RECOMMENDED
"""

        # --- Rules ---
        rules = """
*** RULES ***
- Be creative. Propose short plans (1-2 steps).
- IDs ("id": "...") are optional and auto-generated if omitted.
- For params like "window": {{90|365}}, CHOOSE ONE value (e.g., "window": 365).
- `create_learned_embedding` & `create_mvse_features` are very powerful. Try creating interactions with them.
"""

        system_prompt = base_prompt
        if stage == "basic":
            system_prompt += basic_tools
        else: # advanced stage
            system_prompt += basic_tools + advanced_tools

        system_prompt += rules
        
        try:
            # Use the global gemini_model configured in setup_api_client
            if gemini_model is None:
                raise Exception("Gemini model not initialized. Please run setup_api_client().")
            
            # Combine prompts for Gemini
            full_prompt_for_gemini = system_prompt + "\n\n" + context_prompt
            
            response = gemini_model.generate_content(full_prompt_for_gemini)
            plan_str = response.text

            parsed_json = json.loads(plan_str)

            # --- NEW: Robust plan parsing to handle single-dict or list responses ---
            # Case 1: The response is a dictionary with a "plan" key (legacy format)
            if isinstance(parsed_json, dict) and "plan" in parsed_json:
                plan = parsed_json.get("plan", [])
                return plan if isinstance(plan, list) else [plan]

            # Case 2: The response is already a list of operations (ideal format)
            elif isinstance(parsed_json, list):
                return parsed_json

            # Case 3: The response is a single operation dictionary (common for 1-step plans)
            elif isinstance(parsed_json, dict) and "operation" in parsed_json:
                return [parsed_json] # Wrap it in a list to make it iterable

            # Fallback for unexpected structure
            print(f"  - ⚠️ Warning: LLM returned an unexpected plan structure: {parsed_json}")
            return []
        except Exception as e:
            print(f"❌ Error calling Gemini or parsing plan: {e}")
            # Fallback plan updated to not use the target variable directly for rolling stats
            return [{"operation": "create_rolling_std", "feature": "month", "window": 3, "id": "fallback_err"}]

    def build_llm_context(self, probe_results, iteration_num):
        """Builds the main context dictionary for the LLM prompt."""
        # Note: This probe doesn't generate traditional feature importances.
        # We can pass the list of features being evaluated instead.
        available_features = list(self.best_df.drop(columns=['date', self.target_col]).columns)
        
        # --- NEW: Summarize feature list to save tokens ---
        summarized_features = TLAFS_Algorithm.summarize_feature_list(available_features)
        
        context = {
            "Iteration": f"{iteration_num + 1}/{self.n_iterations}",
            "Current Fusion Score": probe_results['primary_score'],
            "Historical Best Score": self.best_score,
            "R2 Score (LightGBM)": probe_results.get('r2_lgbm', 0.0),
            "R2 Score (SimpleNN)": probe_results.get('r2_nn', 0.0),
            "Number of Features": len(available_features),
            "Available Features": summarized_features,
        }
        
        # NEW: Add feature importances to the context
        importances = probe_results.get("importances")
        if importances is not None and not importances.empty:
            sorted_importances = importances.sort_values(ascending=False)
            # --- OPTIMIZATION: Reduce number of examples ---
            context["Top Features (High Importance)"] = sorted_importances.head(3).to_dict()
            context["Bottom Features (Low Importance)"] = sorted_importances.tail(3).to_dict()
            
        return context

    def format_prompt_for_llm(self, context_dict, in_context_examples_str):
        """Formats the context dictionary and examples into a string for the LLM."""
        prompt = "--- CURRENT STATE & TASK ---\n"
        prompt += "Your goal is to propose a feature engineering plan to improve the 'Current Fusion Score'.\n"
        prompt += "Analyze the feature importances below. Consider deleting low-importance features or creating interactions between high-importance ones.\n"
        for key, value in context_dict.items():
            if isinstance(value, (float, np.floating)):
                 prompt += f"- {key}: {value:.4f}\n"
            elif isinstance(value, dict):
                prompt += f"- {key}:\n"
                # Filter out zero-importance features for brevity
                filtered_dict = {k: v for k, v in value.items() if v > 0} if "Importance" in key else value
                for k, v in filtered_dict.items():
                    # --- OPTIMIZATION: Reduce float precision ---
                    prompt += f"  - {k}: {v:.4f}\n"
            else:
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nPropose a short, creative list of 1-2 operations to improve the score."
        prompt += in_context_examples_str
        return prompt

    def run(self):
        print(f"\n💡 Starting T-LAFS with Fusion Probe (RL Framework) ...\n")
        current_df = self.base_df.copy()
        
        # Start with a simple, universally useful feature to give the LLM a good baseline.
        initial_plan = [
            {"operation": "create_lag", "on": self.target_col, "days": 1, "id": "lag1"}
        ]
        current_df = self.execute_plan(current_df, initial_plan)
        current_plan = initial_plan
        
        print("\nEstablishing baseline score with initial feature set (lag1)...")
        try:
            probe_results = probe_feature_set(current_df, self.target_col)
            current_score = probe_results["primary_score"]
            self.last_probe_results = probe_results

            self.best_score = current_score
            self.best_df = current_df.copy()
            self.best_plan = current_plan.copy()
            print(f"  - Initial baseline score (Fusion R²): {self.best_score:.4f} | Features: {probe_results.get('num_features', -1)}")
        except Exception as e:
            import traceback
            print(f"  - ❌ ERROR during initial evaluation: {e}\n{traceback.format_exc()}")
            return None, None, -1

        for i in range(self.n_iterations):
            print(f"\n----- ITERATION {i+1}/{self.n_iterations} (Probe: Fusion R²) -----")
            
            last_results = self.last_probe_results
            print(f"  - Current Best Score (Fusion R²): {self.best_score:.4f} | Last Score: {last_results['primary_score']:.4f} | R2_LGBM: {last_results.get('r2_lgbm', -1):.3f} | R2_NN: {last_results.get('r2_nn', -1):.3f} | #Feats: {last_results.get('num_features', -1)}")

            print("\nStep 1: Strategist LLM is devising a new feature combo plan...")
            
            current_state_context = self.build_llm_context(last_results, i)
            in_context_examples = self.experience_buffer.sample(n_good=1, n_bad=1)
            full_prompt = self.format_prompt_for_llm(current_state_context, in_context_examples)
            plan_extension = self.get_plan_from_llm(full_prompt, i, self.n_iterations)
            
            if not plan_extension:
                self.history.append({"iteration": i + 1, "plan": [], "score": last_results['primary_score'], "adopted": False, "action": "noop"})
                continue
            
            print(f"✅ LLM Strategist proposed: {plan_extension}")

            print(f"\nStep 2: Probing the new feature combo plan...")
            # Always build upon the best known feature set
            df_with_new_features = self.execute_plan(self.best_df.copy(), plan_extension)
            
            new_probe_results = probe_feature_set(df_with_new_features, self.target_col)
            new_score = new_probe_results["primary_score"]

            print(f"  - Probe results: Fusion Score={new_score:.4f}, R2_LGBM: {new_probe_results.get('r2_lgbm', -1):.3f}, R2_NN: {new_probe_results.get('r2_nn', -1):.3f}, #Feats: {new_probe_results.get('num_features', -1)}")
            
            print(f"\nStep 3: Deciding whether to adopt the new plan...")
            is_adopted = new_score > (self.best_score - 0.005)
            
            reward = new_score - self.best_score
            self.experience_buffer.push(current_state_context, plan_extension, reward, is_adopted)
            
            if is_adopted:
                self.last_probe_results = new_probe_results
                print(f"  -> SUCCESS! Plan adopted. New score is {new_score:.4f}.")
                
                if new_score > self.best_score:
                    print(f"     -> And it's a NEW PEAK score, beating {self.best_score:.4f}!")
                    self.best_score = new_score
                    self.best_df = df_with_new_features.copy()
                    self.best_plan += plan_extension
            else:
                print(f"  -> PLAN REJECTED. Score {new_score:.4f} is not a significant improvement over best score {self.best_score:.4f}. Reverting.")
            
            self.history.append({"iteration": i + 1, "plan": plan_extension, "probe_results": new_probe_results, "adopted": is_adopted, "reward": reward})

        print("\n" + "="*80 + f"\n🏆 T-LAFS (Fusion Probe) Finished! 🏆")
        print(f"   - Best Primary Score Achieved (during search): {self.best_score:.4f}")
        
        return self.best_df, self.best_plan, self.best_score


def main():
    """Main function to run the DAP experiment."""
    # ===== 配置变量 =====
    DATASET_TYPE = 'min_daily_temps'  # 可选: 'min_daily_temps' 或 'total_cleaned'
    
    # 实验配置
    N_ITERATIONS = 10
    TARGET_COL = 'temp'
    
    # --- We will run a new search with our validated architecture ---
    USE_SAVED_PLAN = False 
    SAVED_PLAN_PATH = '' # Not used
    
    print("="*80)
    print(f"🚀 T-LAFS Experiment: Search with Fusion Probe")
    print("="*80)

    # --- 创建本次运行的专属结果目录 ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{run_timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"📂 All results for this run will be saved in: {results_dir}")

    # 初始化API客户端
    setup_api_client()
    
    # 加载数据
    base_df = get_time_series_data(DATASET_TYPE)
    best_df = None
    best_feature_plan = None
    best_score_during_search = "N/A (Re-evaluation mode)"

    if USE_SAVED_PLAN:
        print(f"\n" + "="*50)
        print(f"🔬 RE-EVALUATION MODE: Using plan from {SAVED_PLAN_PATH} 🔬")
        print("="*50)

        # We must initialize TLAFS to pre-train the encoders and set static variables
        print("\nStep 1: Initializing T-LAFS to pre-train encoders...")
        tlafs_alg = TLAFS_Algorithm(
            base_df=base_df.copy(),
            target_col=TARGET_COL,
            n_iterations=1, # Not used, but required
            results_dir=results_dir
        )
        print("   ✅ Encoders are ready.")

        # Load and execute the saved plan
        print("\nStep 2: Loading and executing saved feature plan...")
        with open(SAVED_PLAN_PATH, 'r') as f:
            results_data = json.load(f)
        best_feature_plan = results_data['best_feature_plan']
        best_df = TLAFS_Algorithm.execute_plan(base_df.copy(), best_feature_plan)
        print(f"   ✅ Plan executed. Final dataframe has {best_df.shape[1]} features.")
        # print(json.dumps(best_feature_plan, indent=2))

    else:
        # Original logic to run the full search
        print("\n" + "="*50)
        print(f"🚀 T-LAFS SEARCH MODE: Starting dynamic feature search... 🚀")
        print("="*50)
        tlafs = TLAFS_Algorithm(
            base_df=base_df,
            target_col=TARGET_COL,
            n_iterations=N_ITERATIONS,
            results_dir=results_dir
        )
        best_df, best_feature_plan, best_score_during_search = tlafs.run()


    # --- 最终验证和总结 ---
    if best_df is not None:
        probe_name_for_reporting = "FinalSearch_Fusion_Probe"
        print("\n" + "="*40)
        print(f"🔬 FINAL VALIDATION ON ALL MODELS ({probe_name_for_reporting}) 🔬")
        print("="*40)
        final_metrics, final_results = evaluate_on_multiple_models(
            best_df,
            TARGET_COL,
            probe_name_for_reporting
        )

        if final_metrics:
            # Find best model based on R^2 score
            best_final_model_name = max(final_metrics, key=lambda k: final_metrics[k]['r2'])
            best_final_metrics = final_metrics[best_final_model_name]
            best_result = final_results[best_final_model_name]

            print("\n" + "="*60)
            print(f"🏆 EXECUTIVE SUMMARY: T-LAFS '{probe_name_for_reporting}' STRATEGY 🏆")
            print("="*60)
            print(f"Feature engineering search was conducted by: '{probe_name_for_reporting}'")
            if not USE_SAVED_PLAN:
                print(f"   - Best Primary Score during search phase: {best_score_during_search:.4f}")
            print(f"\nBest feature plan discovered:")
            print(json.dumps(best_feature_plan, indent=2, ensure_ascii=False))
            print("\n------------------------------------------------------------")
            print(f"This feature set was then validated on a suite of specialist models.")
            print(f"🥇 Best Performing Specialist Model: '{best_final_model_name}'")
            print(f"🚀 Final Validated R² Score: {best_final_metrics['r2']:.4f}")
            print(f"   - MAE: {best_final_metrics['mae']:.4f}, RMSE: {best_final_metrics['rmse']:.4f}")
            print("\nConclusion: The T-LAFS framework successfully used the probe to find a")
            print("powerful feature set, which a specialist model then exploited to")
            print("achieve high-performance predictions.")
            print("="*60)

            visualize_final_predictions(
                dates=best_result['dates'],
                y_true=best_result['y_true'],
                y_pred=best_result['y_pred'],
                best_model_name=best_final_model_name,
                probe_name=probe_name_for_reporting,
                best_model_metrics=best_final_metrics,
                results_dir=results_dir
            )
            
            # --- 保存结果到JSON ---
            print("\n💾 Saving results to JSON...")
            results_to_save = {
                "run_mode": "re-evaluation" if USE_SAVED_PLAN else "search",
                "source_plan_file": SAVED_PLAN_PATH if USE_SAVED_PLAN else "N/A",
                "probe_model": probe_name_for_reporting,
                "best_score_during_search": best_score_during_search,
                "best_feature_plan": best_feature_plan,
                "final_features": [col for col in best_df.columns if col not in ['date', TARGET_COL]],
                "final_validation_scores": final_metrics,
                "best_final_model": {
                    "name": best_final_model_name,
                    "metrics": best_final_metrics
                },
                "run_history": tlafs.history if not USE_SAVED_PLAN else "N/A (Re-evaluation mode)"
            }
            save_results_to_json(results_to_save, probe_name_for_reporting, results_dir)

        else:
            print("\nCould not generate final validation scores.")

if __name__ == "__main__":
    main()
