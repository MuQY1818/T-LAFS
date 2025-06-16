"""
评估工具模块

提供模型评估和指标计算的工具函数。
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import lightgbm as lgb

from ..models.neural_models import TransformerModel, SimpleNN, EnhancedNN
# 假设其他模型和训练函数已在相应模块中定义
# from ..models.traditional import LightGBM, RandomForest, ...
from .training import train_pytorch_model

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含各种评估指标的字典
    """
    # 避免除以零
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def calculate_improvement(baseline_metrics: Dict[str, float], 
                         new_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    计算性能改进
    
    Args:
        baseline_metrics: 基准模型指标
        new_metrics: 新模型指标
        
    Returns:
        包含各项指标改进百分比的字典
    """
    improvements = {}
    for metric in baseline_metrics:
        if metric in new_metrics:
            baseline_value = baseline_metrics[metric]
            new_value = new_metrics[metric]
            if baseline_value != 0:
                improvement = ((new_value - baseline_value) / abs(baseline_value)) * 100
                improvements[f'{metric}_improvement'] = improvement
    
    return improvements 

def probe_feature_set(df: pd.DataFrame, target_col: str, features_to_probe: list = None):
    """
    专门针对LightGBM的快速探针。
    通过在LightGBM模型上测试来快速评估特征集的质量。
    
    Args:
        df (pd.DataFrame): 包含所有数据（特征+目标）的DataFrame。
        target_col (str): 目标列的名称。
        features_to_probe (list, optional): 要用于探测的特征列列表。
                                            如果为None，则使用df中除'date'和目标列外的所有列。
    """
    if features_to_probe is None:
        features_to_probe = [col for col in df.columns if col not in ['date', target_col]]

    # 确保所有要探测的特征都在DataFrame中
    missing_cols = [col for col in features_to_probe if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下用于探测的特征在DataFrame中不存在: {missing_cols}")

    df_feat = df[features_to_probe].dropna()
    y = df.loc[df_feat.index][target_col]
    X = df_feat

    if X.empty or y.empty or len(X) < 20:
        # 对于MAE/RMSE，返回一个很大的值表示差的性能
        return np.inf, {"mae_lgbm": np.inf, "num_features": 0}, pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(X_train) < 1 or len(X_test) < 1:
        return np.inf, {"mae_lgbm": np.inf, "num_features": X.shape[1]}, pd.DataFrame()

    # 使用 LightGBM 作为快速探针
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    preds = lgbm.predict(X_test)
    
    # 核心指标改为MAE
    score = mean_absolute_error(y_test, preds)

    primary_score = score
    
    # --- 新增：计算并返回特征重要性 ---
    importances_df = pd.DataFrame({
        'feature': X.columns,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # 统一返回格式为 (score, details_dict, importances_df)
    details = {
        "mae_lgbm": score, # 报告MAE
        "num_features": X.shape[1]
    }
    return primary_score, details, importances_df

def evaluate_on_multiple_models(df: pd.DataFrame, target_col: str):
    """
    在多种模型上评估最终的特征集。
    """
    print(f"在多种模型上评估最终特征集...")
    
    X = df.drop(columns=['date', target_col]).dropna()
    y = df.loc[X.index][target_col]

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
        'SimpleNN': SimpleNN(X.shape[1]),
        'EnhancedNN': EnhancedNN(X.shape[1]),
        'Transformer': TransformerModel(X.shape[1])
    }
    
    final_metrics = {}
    final_results = {}

    # 将数据分为训练+验证集 和 测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 从训练+验证集中分出训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=False
    )

    # 为PyTorch模型准备缩放器和数据
    scaler_x = MinMaxScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)
    X_test_s = scaler_x.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_val_s = scaler_y.transform(y_val.values.reshape(-1, 1))

    for name, model in models.items():
        print(f"  - 正在评估 {name}...")
        if isinstance(model, torch.nn.Module):
            preds_scaled = train_pytorch_model(
                model, X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, model_name=name
            )
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else: # 对于scikit-learn模型 (如LightGBM)
            # 对于非NN模型，我们使用完整的训练+验证集进行训练
            model.fit(X_train_val, y_train_val)
            preds = model.predict(X_test)
        
        # 使用我们更新过的evaluate_model函数
        metrics = evaluate_model(y_test, preds)

        final_metrics[name] = metrics
        final_results[name] = {
            "dates": X_test.index.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": preds.tolist()
        }
        print(f"    - {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
        
    return final_metrics, final_results 