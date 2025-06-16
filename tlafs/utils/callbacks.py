import numpy as np
import torch
import os

class EarlyStopping:
    """
    当验证损失在给定的耐心周期内没有改善时，提前停止训练。
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 在停止训练前，要等待多少个轮次没有改善。
                            耐心值：7
            verbose (bool): 如果为True，则为每次验证损失改善时打印消息。
                            默认：False
            delta (float): 损失被认为是改善的最小变化量。
                           默认：0
            path (str): 保存模型的路径。
                        默认：'checkpoint.pt'
            trace_func (function): 用于打印跟踪消息的函数。
                                   默认：print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """当验证损失减少时保存模型。"""
        if self.verbose:
            self.trace_func(f'验证损失从 ({self.val_loss_min:.6f} --> {val_loss:.6f}) 下降。正在保存模型...')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 