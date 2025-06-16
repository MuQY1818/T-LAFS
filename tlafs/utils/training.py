import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import numpy as np

from .callbacks import EarlyStopping

def train_pytorch_model(model, X_train, y_train, X_val, y_val, X_test, 
                        epochs=100, batch_size=32, lr=0.001, model_name="model"):
    """
    一个通用的函数，用于训练PyTorch模型并返回预测结果，现在加入了早停机制。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 初始化早停
    checkpoint_path = f"checkpoints/{model_name}_best.pt"
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("早停机制触发")
            break

    # 加载最佳模型权重
    print(f"加载最佳模型: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 在测试集上进行最终预测
    model.eval()
    with torch.no_grad():
        preds_tensor = model(torch.FloatTensor(X_test).to(device))
        
    return preds_tensor.cpu().numpy().flatten() 