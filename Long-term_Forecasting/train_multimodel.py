#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型时间序列预测训练脚本
支持使用 Hugging Face 统一接口的预训练模型进行时间序列预测的训练和评估
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import json
from pathlib import Path

# 导入自定义模块
from models.GPT4TS import MultiModelTS
from configs.multimodel_config import (
    get_model_config, print_config_summary, 
    get_preset_model_list, print_available_models
)


class TimeSeriesTrainer:
    """时间序列预测模型训练器
    
    支持任何基于 Transformer 的预训练模型的训练、验证和评估流程
    """
    
    def __init__(self, config, device):
        """初始化训练器
        
        Args:
            config: 模型配置对象
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = MultiModelTS(config, device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_synthetic_dataloader(self, split='train'):
        """创建合成数据的数据加载器
        
        Args:
            split (str): 数据分割类型 ('train', 'val', 'test')
            
        Returns:
            DataLoader: 数据加载器
        """
        # 根据分割类型设置数据大小
        if split == 'train':
            num_samples = 1000
        elif split == 'val':
            num_samples = 200
        else:  # test
            num_samples = 200
        
        # 创建合成时间序列数据
        data = []
        for i in range(num_samples):
            # 创建具有不同模式的时间序列
            t = np.linspace(0, 4*np.pi, self.config.seq_len + self.config.pred_len)
            
            series = []
            for v in range(7):  # 7个变量
                freq = 0.5 + v * 0.1 + np.random.normal(0, 0.05)
                phase = v * np.pi / 4 + np.random.normal(0, 0.1)
                noise = np.random.normal(0, 0.1, len(t))
                
                signal = (np.sin(freq * t + phase) + 
                         0.1 * t + 
                         0.3 * np.sin(2 * freq * t) + 
                         noise)
                series.append(signal)
            
            data.append(np.array(series).T)
        
        data = np.array(data)
        
        # 分割输入和标签
        x = data[:, :self.config.seq_len, :]
        y = data[:, self.config.seq_len:, :]
        
        # 转换为张量
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=(split == 'train'),
            num_workers=2 if split == 'train' else 1
        )
        
        return dataloader
    
    def train_epoch(self, train_loader):
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            y_pred = self.model(x, itr=batch_idx)
            loss = self.criterion(y_pred, y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f'    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            float: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x, itr=0)
                loss = self.criterion(y_pred, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, save_dir='./checkpoints'):
        """完整的训练流程
        
        Args:
            save_dir (str): 模型保存目录
        """
        print(f"🚀 开始训练模型: {self.config.model_name_or_path}")
        print("=" * 60)
        
        # 创建保存目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建数据加载器
        print("📊 创建数据加载器...")
        train_loader = self.create_synthetic_dataloader('train')
        val_loader = self.create_synthetic_dataloader('val')
        
        print(f"   - 训练样本: {len(train_loader.dataset)}")
        print(f"   - 验证样本: {len(val_loader.dataset)}")
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n📈 Epoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"   训练损失: {train_loss:.6f}")
            print(f"   验证损失: {val_loss:.6f}")
            print(f"   学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   用时: {epoch_time:.2f}s")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存最佳模型
                self.save_model(save_dir, epoch, is_best=True)
                print(f"   ✅ 新的最佳模型已保存 (验证损失: {val_loss:.6f})")
                
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"   ⏹️  早停触发 (耐心值: {self.config.patience})")
                    break
        
        print(f"\n🎉 训练完成!")
        print(f"   最佳验证损失: {self.best_val_loss:.6f}")
        
        # 保存训练历史
        self.save_training_history(save_dir)
    
    def save_model(self, save_dir, epoch, is_best=False):
        """保存模型
        
        Args:
            save_dir (str): 保存目录
            epoch (int): 当前epoch
            is_best (bool): 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # 为文件名创建安全的模型名称
        safe_model_name = self.config.model_name_or_path.replace('/', '_').replace('\\', '_')
        
        # 保存最新模型
        latest_path = os.path.join(save_dir, f'{safe_model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(save_dir, f'{safe_model_name}_best.pth')
            torch.save(checkpoint, best_path)
    
    def save_training_history(self, save_dir):
        """保存训练历史
        
        Args:
            save_dir (str): 保存目录
        """
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # 为文件名创建安全的模型名称
        safe_model_name = self.config.model_name_or_path.replace('/', '_').replace('\\', '_')
        
        history_path = os.path.join(save_dir, f'{safe_model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模型时间序列预测训练')
    parser.add_argument('--model', type=str, default='gpt2', 
                      help='要训练的模型 (预设模型键名、本地路径或 Hub 名称)')
    parser.add_argument('--experiment', type=str, default='quick_test',
                      choices=['quick_test', 'small_scale', 'full_scale'],
                      help='实验规模')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                      help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 使用设备: {device}")
    
    # 获取模型配置
    try:
        config = get_model_config(args.model, args.experiment)
        print(f"\n📋 模型配置:")
        print_config_summary(config)
    except Exception as e:
        print(f"❌ 配置错误: {str(e)}")
        return
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建训练器并开始训练
    try:
        trainer = TimeSeriesTrainer(config, device)
        trainer.train(args.save_dir)
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 