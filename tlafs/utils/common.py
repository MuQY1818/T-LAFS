"""
Common utility functions for TLAFS
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series data
    
    Args:
        data: Input time series data
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (X, y) where X is the input sequences and y is the target values
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def visualize_predictions(dates: List[str], y_true: np.ndarray, y_pred: np.ndarray,
                         title: str, save_path: str):
    """
    Visualize model predictions
    
    Args:
        dates: List of dates
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='True', color='blue')
    plt.plot(dates, y_pred, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results(results: Dict[str, Any], save_path: str):
    """
    Save results to JSON file
    
    Args:
        results: Dictionary of results
        save_path: Path to save the results
    """
    def json_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(save_path, 'w') as f:
        json.dump(results, f, default=json_converter, indent=4)

def load_results(load_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file
    
    Args:
        load_path: Path to load the results from
        
    Returns:
        Dictionary of results
    """
    with open(load_path, 'r') as f:
        return json.load(f)

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create directory for experiment results
    
    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to the experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir 