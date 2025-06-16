"""
数据工具模块

提供数据处理和转换的工具函数。
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import acf

def analyze_dataset_characteristics(df: pd.DataFrame, target_col: str):
    """
    对时间序列数据集进行全面的自动化分析。
    """
    print("\n" + "="*40)
    print("🔬 正在执行自动化数据集分析...")
    print("="*40)
    
    results = {}
    
    # 1. 基本信息
    results['time_range_start'] = df['date'].min().strftime('%Y-%m-%d')
    results['time_range_end'] = df['date'].max().strftime('%Y-%m-%d')
    results['duration_days'] = (df['date'].max() - df['date'].min()).days
    results['total_records'] = len(df)
    
    # 2. 缺失值分析
    results['missing_values_count'] = int(df[target_col].isnull().sum())
    results['missing_values_percentage'] = float(df[target_col].isnull().mean() * 100)
    
    # 3. 目标变量统计
    target_series = df[target_col].dropna()
    results['target_mean'] = float(target_series.mean())
    results['target_std'] = float(target_series.std())
    results['target_min'] = float(target_series.min())
    results['target_max'] = float(target_series.max())
    results['target_skewness'] = float(target_series.skew())
    results['target_kurtosis'] = float(target_series.kurt())
    
    # 4. 平稳性检验 (ADF Test)
    try:
        adf_result = adfuller(target_series)
        results['adf_statistic'] = float(adf_result[0])
        results['adf_p_value'] = float(adf_result[1])
        results['is_stationary'] = bool(adf_result[1] < 0.05)
    except Exception as e:
        results['adf_test_error'] = str(e)
        results['is_stationary'] = 'Unknown'
        
    # 5. 季节性分析 (ACF)
    try:
        # 计算ACF，nlags可以设置为总数据点的一半，或者一个固定的较大值
        autocorr = acf(target_series, nlags=min(365*2, len(target_series)//2 - 1), fft=True)
        # 寻找除了lag 0之外的第一个显著峰值
        significant_peaks = np.where(autocorr > 2 / np.sqrt(len(target_series)))[0]
        peak_lags = significant_peaks[significant_peaks > 0]
        
        if len(peak_lags) > 0:
            # 常见的季节性周期
            common_periods = [7, 30, 90, 365]
            detected_seasonality = []
            for p in common_periods:
                # 检查在周期p附近是否存在峰值（例如，p-2 到 p+2 的范围内）
                if any(abs(lag - p) <= 2 for lag in peak_lags):
                    detected_seasonality.append(p)
            results['detected_seasonality_periods'] = detected_seasonality
        else:
            results['detected_seasonality_periods'] = []
    except Exception as e:
        results['seasonality_analysis_error'] = str(e)

    # 打印总结
    print("📋 数据集分析报告:")
    for key, value in results.items():
        print(f"  - {key}: {value}")
    print("="*40 + "\n")
    
    return results

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时间序列数据
    
    Args:
        data: 输入数据
        seq_length: 序列长度
        
    Returns:
        X: 输入序列
        y: 目标值
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def prepare_data_for_model(data: pd.DataFrame, target_col: str, seq_length: int, 
                          train_size: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备模型训练数据
    
    Args:
        data: 输入数据框
        target_col: 目标列名
        seq_length: 序列长度
        train_size: 训练集比例
        
    Returns:
        X_train: 训练集输入
        y_train: 训练集目标
        X_test: 测试集输入
        y_test: 测试集目标
    """
    # 提取目标列
    target_data = data[target_col].values
    
    # 创建序列
    X, y = create_sequences(target_data, seq_length)
    
    # 划分训练集和测试集
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test

def get_time_series_data(dataset_type='min_daily_temps'):
    """
    从CSV文件加载并准备时间序列数据。
    - 将列名重命名为 'date' 和 'temp'。
    - 将 'date' 列转换为datetime对象。
    - 按日期排序。
    """
    if dataset_type == 'total_cleaned':
        csv_path = 'data/total_cleaned.csv'
        df = pd.read_csv(csv_path)
        # 假设列名为 '日期' 和 '成交商品件数'
        df.rename(columns={'日期': 'date', '成交商品件数': 'temp'}, inplace=True)
    else: # 默认为 min_daily_temps
        csv_path = 'data/min_daily_temps.csv'
        df = pd.read_csv(csv_path)
        df.rename(columns={'Date': 'date', 'Temp': 'temp'}, inplace=True)
        
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ 数据加载自: {csv_path}, Shape: {df.shape}")
    return df 