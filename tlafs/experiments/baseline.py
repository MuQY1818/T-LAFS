"""
基准实验模块
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from tlafs.models.neural_models import SimpleNN, EnhancedNN, TransformerModel
from tlafs.utils.data_utils import create_sequences
from tlafs.utils.evaluation import evaluate_model
from tlafs.utils.visualization import plot_predictions, plot_model_comparison, plot_learning_curves
from tlafs.utils.file_utils import save_results

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'验证损失从 {self.val_loss_min:.6f} 下降到 {val_loss:.6f}，保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def get_time_series_data(dataset_type: str = 'min_daily_temps') -> pd.DataFrame:
    """
    获取时间序列数据
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        包含时间序列数据的DataFrame
    """
    print(f"\n正在加载数据集: {dataset_type}")
    if dataset_type == 'min_daily_temps':
        df = pd.read_csv('data/min_daily_temps.csv')
    elif dataset_type == 'total_cleaned':
        df = pd.read_csv('data/total_cleaned.csv')
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # 统一列名：第一列为 'Date'，第二列为 'Target'
    df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Target'}, inplace=True)
    print(f"数据集加载完成，列已重命名。形状: {df.shape}")
    return df

def train_pytorch_model(model, X_train, y_train, X_val, y_val, X_test, epochs=100, batch_size=32, lr=0.001, model_name='model'):
    """训练PyTorch模型并返回训练好的模型和预测结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 初始化早停
    checkpoint_path = f'checkpoints/{model_name}_best.pth'
    os.makedirs('checkpoints', exist_ok=True)
    early_stopping = EarlyStopping(patience=7, verbose=True)
    
    # 训练模型
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        with torch.no_grad():
            for X_batch, y_batch in val_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 早停检查
        early_stopping(avg_val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("触发早停机制")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor)
        y_pred = y_pred.cpu().numpy().squeeze()
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}, y_pred

def evaluate_on_multiple_models(data, target_col='Target'):
    """在多个模型上评估数据"""
    print(f"\n开始评估多个模型，目标列: {target_col}")
    print(f"数据形状: {data.shape}")
    
    # 准备数据
    X, y = create_sequences(data, seq_length=14)
    print(f"序列数据形状: X: {X.shape}, y: {y.shape}")
    
    # 分割数据
    train_size = int(len(X) * 0.7)  # 70% 训练
    val_size = int(len(X) * 0.15)   # 15% 验证
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"验证集形状: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    results = {
        'test_data': y_test  # 保存测试集数据
    }
    models = {
        'SimpleNN': SimpleNN(input_size=X.shape[1]),
        'EnhancedNN': EnhancedNN(input_size=X.shape[1]),
        'Transformer': TransformerModel(input_size=X.shape[1])
    }
    
    for name, model in models.items():
        print(f"\n开始评估模型: {name}")
        trained_model, history, y_pred = train_pytorch_model(
            model, X_train, y_train, X_val, y_val, X_test,
            model_name=name
        )
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"模型 {name} 的评估指标:")
        print(f"  mse: {mse:.4f}")
        print(f"  rmse: {rmse:.4f}")
        print(f"  mae: {mae:.4f}")
        print(f"  r2: {r2:.4f}")
        
        results[name] = {
            'model': trained_model,
            'predictions': y_pred,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'history': history
        }
    
    return results

def plot_results(results: Dict[str, Any], experiment_dir: str):
    """绘制实验结果
    
    Args:
        results: 包含所有模型结果的字典
        experiment_dir: 实验结果的保存目录
    """
    y_test = results.pop('test_data') # 提取并移除测试数据
    
    # 为每个模型绘制预测结果和学习曲线
    for model_name, model_results in results.items():
        # 绘制预测结果对比
        plot_predictions(
            y_true=y_test,
            y_pred=model_results['predictions'],
            title=f'{model_name} - 预测结果对比',
            save_path=os.path.join(experiment_dir, f'{model_name}_predictions.png')
        )
        
        # 绘制学习曲线
        plot_learning_curves(
            train_losses=model_results['history']['train_losses'],
            val_losses=model_results['history']['val_losses'],
            save_path=os.path.join(experiment_dir, f'{model_name}_learning_curves.png')
        )
        
    # 绘制模型性能对比
    metrics_dict = {model_name: res['metrics'] for model_name, res in results.items()}
    plot_model_comparison(
        metrics_dict,
        save_path=os.path.join(experiment_dir, 'model_comparison.png')
    )
    
    results['test_data'] = y_test # 将测试数据添加回来以便于其他可能的用途

def main():
    """主函数"""
    print("\n=== 开始运行基准实验 ===\n")
    
    # 加载数据
    df = get_time_series_data('min_daily_temps')
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join('results', f'baseline_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"创建实验目录: {experiment_dir}")
    
    # 运行实验
    results = evaluate_on_multiple_models(df['Target'], target_col='Target')
    
    # 保存结果
    results_file = os.path.join(experiment_dir, 'results.json')
    save_results(results, results_file)
    
    # 生成可视化
    plot_results(results, experiment_dir)
    
    print(f"\n实验完成，结果已保存到: {experiment_dir}")

if __name__ == '__main__':
    main()