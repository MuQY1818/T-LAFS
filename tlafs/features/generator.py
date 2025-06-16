import pandas as pd
import numpy as np
import torch

# 假设相关的模型和函数可以从框架的其他部分导入
# from ..models.autoencoder import MaskedEncoder
# from ..models.probe import ProbeForecaster
# from .mvse import generate_mvse_features_for_tlafs

# 导入新的MVSE特征生成器
from .mvse_features import create_mvse_features

# 在实际应用中，下面的TLAFS_Algorithm属性需要被正确设置或传入
# TLAFS_Algorithm.pretrained_encoders = {}
# TLAFS_Algorithm.embedder_scalers = {}
# TLAFS_Algorithm.meta_forecast_models = {}
# TLAFS_Algorithm.meta_scalers = {}
# TLAFS_Algorithm.probe_config = {}
# TLAFS_Algorithm.probe_model_path = ""
# TLAFS_Algorithm.pretrain_cols_static = []
# TLAFS_Algorithm.target_col_static = ""

def execute_plan(df: pd.DataFrame, plan: list, tlafs_params: dict):
    """
    在一个给定的DataFrame上执行特征工程计划。
    这是所有特征生成的单一、权威的静态方法。
    它包含了所有无泄漏的特征生成逻辑。
    """
    temp_df = df.copy()
    new_feature_name = None # 用于跟踪新生成的特征

    # 从传入的字典中获取必要的参数
    target_col = tlafs_params.get("target_col_static")
    pretrained_encoders = tlafs_params.get("pretrained_encoders", {})
    embedder_scalers = tlafs_params.get("embedder_scalers", {})
    meta_forecast_models = tlafs_params.get("meta_forecast_models", {})
    meta_scalers = tlafs_params.get("meta_scalers", {})
    # ... 其他需要的参数 ...

    required_time_cols = ['dayofweek', 'month', 'weekofyear', 'is_weekend']
    if not all(col in temp_df.columns for col in required_time_cols):
        # ... (确保基础时间特征存在的逻辑) ...
        pass
    
    # --- 核心修改：确保plan是一个列表 ---
    if not isinstance(plan, list):
        plan = [plan] # 如果plan是单个字典，将其包装在列表中

    executed_feature_names = []

    for step in plan:
        op = None
        try:
            op = step.get("function")
            args = step.get("args", {})
            
            # 用于记录当前步骤生成特征的临时变量
            step_feature_name = None

            # --- Macro functions ---
            if op == "create_time_features_macro":
                print("  - Executing macro: create_time_features_macro...")
                # These features don't create NaNs, so no fill needed.
                temp_df['year'] = pd.to_datetime(temp_df['date']).dt.year
                temp_df['month'] = pd.to_datetime(temp_df['date']).dt.month
                temp_df['day'] = pd.to_datetime(temp_df['date']).dt.day
                temp_df['dayofweek'] = pd.to_datetime(temp_df['date']).dt.dayofweek
                step_feature_name = "time_features_macro_set"
                print("  - ✅ Successfully generated time features macro set.")

            elif op == "create_lag_features_macro":
                print("  - Executing macro: create_lag_features_macro...")
                target = args.get("col", target_col)
                lags_to_create = [1, 2, 3, 7, 14]
                feature_names = []
                for lag in lags_to_create:
                    fname = f'{target}_lag_{lag}'
                    temp_df[fname] = temp_df[target].shift(lag)
                    feature_names.append(fname)
                # Fill NaNs for the newly created columns
                temp_df[feature_names] = temp_df[feature_names].ffill().fillna(0)
                step_feature_name = f"lag_features_macro_set_on_{target}"
                print(f"  - ✅ Successfully generated lag features macro set for '{target}'.")

            elif op == "create_rolling_features_macro":
                print("  - Executing macro: create_rolling_features_macro...")
                target = args.get("col", target_col)
                windows_to_create = [7, 14, 30]
                aggs_to_create = ['mean', 'std']
                feature_names = []
                for window in windows_to_create:
                    for agg in aggs_to_create:
                        fname = f'{target}_rolling_{agg}_{window}'
                        temp_df[fname] = temp_df[target].rolling(window=window).agg(agg).shift(1)
                        feature_names.append(fname)
                # Fill NaNs for the newly created columns
                temp_df[feature_names] = temp_df[feature_names].ffill().fillna(0)
                step_feature_name = f"rolling_features_macro_set_on_{target}"
                print(f"  - ✅ Successfully generated rolling features macro set for '{target}'.")

            elif op == "create_kitchen_sink_features_macro":
                print("  - 💥 Executing the 'Kitchen Sink' macro to generate a massive feature set...")
                target = args.get("col", target_col)
                
                # 1. Extensive Time Features
                date_col = pd.to_datetime(temp_df['date'])
                temp_df['year'] = date_col.dt.year
                temp_df['month'] = date_col.dt.month
                temp_df['day'] = date_col.dt.day
                temp_df['dayofweek'] = date_col.dt.dayofweek
                temp_df['dayofyear'] = date_col.dt.dayofyear
                temp_df['weekofyear'] = date_col.dt.isocalendar().week.astype(int)
                temp_df['quarter'] = date_col.dt.quarter
                temp_df['is_month_start'] = date_col.dt.is_month_start.astype(int)
                temp_df['is_month_end'] = date_col.dt.is_month_end.astype(int)
                temp_df['is_quarter_start'] = date_col.dt.is_quarter_start.astype(int)
                temp_df['is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
                temp_df['is_year_start'] = date_col.dt.is_year_start.astype(int)
                temp_df['is_year_end'] = date_col.dt.is_year_end.astype(int)

                # 2. Extensive Lag Features
                lags = list(range(1, 16)) + [21, 28, 35, 60, 90, 180, 365] # ~40 features
                for lag in lags:
                    temp_df[f'{target}_lag_{lag}'] = temp_df[target].shift(lag)

                # 3. Extensive Rolling Features
                windows = [7, 14, 28, 60, 90]
                aggs = ['mean', 'std', 'min', 'max', 'median'] # 5 aggs * 5 windows = 25 features
                for window in windows:
                    rolling = temp_df[target].rolling(window=window)
                    for agg in aggs:
                        temp_df[f'{target}_rolling_{agg}_{window}'] = rolling.agg(agg).shift(1)

                # 4. Fourier Features
                year_length = 365.25
                for k in range(1, 11): # 10 orders = 20 features
                    temp_df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * temp_df['dayofyear'] / year_length)
                    temp_df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * temp_df['dayofyear'] / year_length)

                # 5. Interaction Features (safe ones)
                # Create lags first to interact with
                if f'{target}_lag_1' not in temp_df.columns:
                     temp_df[f'{target}_lag_1'] = temp_df[target].shift(1)
                if f'{target}_lag_7' not in temp_df.columns:
                     temp_df[f'{target}_lag_7'] = temp_df[target].shift(7)

                time_features_for_interaction = ['dayofweek', 'month', 'weekofyear', 'quarter']
                for col in time_features_for_interaction:
                    temp_df[f"{col}_x_lag1"] = temp_df[col] * temp_df[f'{target}_lag_1']
                    temp_df[f"{col}_x_lag7"] = temp_df[col] * temp_df[f'{target}_lag_7']

                # Final cleanup
                temp_df.fillna(0, inplace=True)
                
                step_feature_name = "kitchen_sink_macro_set"
                num_features = len(temp_df.columns) - len(df.columns)
                print(f"  - ✅ Generated ~{num_features} new features with the Kitchen Sink macro.")

            # --- Basic time series features ---
            elif op == "create_lag_features":
                col = args.get("col")
                lags = args.get("lags", [1])
                for lag in lags:
                    step_feature_name = f"{col}_lag_{lag}"
                    temp_df[step_feature_name] = temp_df[col].shift(lag).ffill().fillna(0)

            elif op == "create_rolling_features":
                col = args.get("col")
                windows = args.get("windows", [7])
                aggs = args.get("aggs", ['mean'])
                for window in windows:
                    for agg in aggs:
                        step_feature_name = f"{col}_rolling_{agg}_{window}"
                        temp_df[step_feature_name] = temp_df[col].rolling(window=window).agg(agg).shift(1).ffill().fillna(0)

            elif op == "create_interaction_features":
                col1 = args.get("col1")
                col2 = args.get("col2")
                # --- Data leakage firewall ---
                if col1 == target_col or col2 == target_col:
                    print(f"  - 🛑 Data leakage warning: Cannot use target column ('{target_col}') directly. Skipping.")
                    continue
                
                if col1 in temp_df.columns and col2 in temp_df.columns:
                    step_feature_name = f"{col1}_x_{col2}"
                    temp_df[step_feature_name] = temp_df[col1] * temp_df[col2]
                else:
                    print(f"  - ⚠️ Columns not found: {col1} or {col2}. Skipping.")
                    continue
            
            elif op == "create_fourier_features":
                col = args.get("col")
                order = int(args.get("order", 1))
                if col != 'date' or col not in temp_df.columns:
                    print(f"  - ⚠️ Fourier features must use 'date' column. Skipping.")
                    continue
                
                print(f"  - Creating {order}th order Fourier features for '{col}'...")
                day_of_year = pd.to_datetime(temp_df[col]).dt.dayofyear
                year_length = 365.25
                
                for k in range(1, order + 1):
                    sin_col = f"fourier_sin_{k}"
                    cos_col = f"fourier_cos_{k}"
                    temp_df[sin_col] = np.sin(2 * np.pi * k * day_of_year / year_length)
                    temp_df[cos_col] = np.cos(2 * np.pi * k * day_of_year / year_length)
                
                step_feature_name = f"fourier_features_order_{order}"

            elif op == "delete_features":
                cols_to_delete = args.get("cols", [])
                if not isinstance(cols_to_delete, list):
                    cols_to_delete = [cols_to_delete]
                
                existing_cols = [c for c in cols_to_delete if c in temp_df.columns]
                if existing_cols:
                    print(f"  - Deleting features: {existing_cols}")
                    temp_df.drop(columns=existing_cols, inplace=True)
                    step_feature_name = f"deleted_{len(existing_cols)}_features"
                else:
                    print(f"  - ⚠️ Features not found: {cols_to_delete}. Skipping.")
                    continue

            # --- MVSE Probe Features ---
            elif op == "create_mvse_features":
                temp_df, step_feature_name = create_mvse_features(temp_df, target_col=target_col, **args)
            
            elif op == "create_embedding_features":
                col = args.get("col")
                window_size = args.get("window_size", 90)
                
                embedder = pretrained_encoders.get(window_size)
                scaler = embedder_scalers.get(window_size)

                if embedder and scaler:
                    pretrain_cols = tlafs_params.get("pretrain_cols_static", [])
                    
                    if not all(c in temp_df.columns for c in pretrain_cols):
                         temp_df['dayofweek'] = temp_df['date'].dt.dayofweek
                         temp_df['month'] = temp_df['date'].dt.month
                         temp_df['weekofyear'] = temp_df['date'].dt.isocalendar().week.astype(int)
                         temp_df['is_weekend'] = (temp_df['date'].dt.dayofweek >= 5).astype(int)

                    print(f"  - Generating multivariate embeddings from {len(pretrain_cols)} features (window:{window_size})...")
                    df_for_embedding = temp_df[pretrain_cols]
                    scaled_features = scaler.transform(df_for_embedding)
                    
                    sequences = np.array([scaled_features[i:i+window_size] for i in range(len(scaled_features) - window_size + 1)])
                    
                    if sequences.size == 0:
                        print(f"  - ⚠️ Insufficient data for window {window_size}. Skipping.")
                        continue
                        
                    tensor = torch.FloatTensor(sequences)
                    with torch.no_grad():
                        embeddings = embedder(tensor).numpy()
                        
                    valid_indices = temp_df.index[window_size-1:]
                    cols = [f"embed_{i}_win{window_size}" for i in range(embeddings.shape[1])]
                    step_feature_name = cols[0]
                    
                    embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)
                    
                    existing_cols_to_drop = [c for c in cols if c in temp_df.columns]
                    if existing_cols_to_drop:
                        temp_df.drop(columns=existing_cols_to_drop, inplace=True)

                    temp_df = temp_df.join(embed_df)
                    temp_df[cols] = temp_df[cols].shift(1).ffill().fillna(0)
                else:
                    print(f"  - ⚠️ Embedder not available for window {window_size}. Skipping.")
                    continue

            else:
                print(f"  - ⚠️ Unknown operation: {op}. Skipping.")
                continue
            
            if step_feature_name:
                executed_feature_names.append(step_feature_name)

        except Exception as e:
            import traceback
            print(f"  - ❌ Error executing step {step}, skipped. Error: {e}\n{traceback.format_exc()}")
            continue
    
    if not df.equals(temp_df) and executed_feature_names:
        final_name = ", ".join(executed_feature_names)
        return temp_df, final_name
    else:
        print("  - ⚠️ No new features generated after plan execution.")
        return None, None

    return temp_df 