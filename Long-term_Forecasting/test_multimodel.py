#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多模型时间序列预测架构
使用 Hugging Face 统一接口测试不同的预训练模型进行时间序列预测
"""

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import time
import matplotlib.pyplot as plt
from models.GPT4TS import MultiModelTS
from configs.multimodel_config import get_model_config, get_preset_model_list, print_available_models


def create_synthetic_data(batch_size=32, seq_len=336, pred_len=96, num_variables=7):
    """创建合成的时间序列数据用于测试
    
    Args:
        batch_size (int): 批次大小
        seq_len (int): 输入序列长度
        pred_len (int): 预测长度
        num_variables (int): 变量数量
    
    Returns:
        tuple: (输入数据, 真实标签)
    """
    # 创建具有季节性和趋势的合成时间序列
    t = np.linspace(0, 4*np.pi, seq_len + pred_len)
    
    data = []
    for b in range(batch_size):
        series = []
        for v in range(num_variables):
            # 不同变量具有不同的频率和相位
            freq = 0.5 + v * 0.1
            phase = v * np.pi / 4
            noise = np.random.normal(0, 0.1, len(t))
            
            # 组合正弦波、线性趋势和噪声
            signal = (np.sin(freq * t + phase) + 
                     0.1 * t + 
                     0.3 * np.sin(2 * freq * t) + 
                     noise)
            series.append(signal)
        
        data.append(np.array(series).T)  # (seq_len + pred_len, num_variables)
    
    data = np.array(data)  # (batch_size, seq_len + pred_len, num_variables)
    
    # 分割输入和标签
    x = data[:, :seq_len, :]  # (batch_size, seq_len, num_variables)
    y = data[:, seq_len:, :]  # (batch_size, pred_len, num_variables)
    
    return torch.FloatTensor(x), torch.FloatTensor(y)


def test_single_model(model_key_or_path, device, x, y_true, experiment_type='quick_test'):
    """测试单个模型的性能
    
    Args:
        model_key_or_path (str): 模型键名或路径
        device (torch.device): 设备
        x (torch.Tensor): 输入数据
        y_true (torch.Tensor): 真实标签
        experiment_type (str): 实验类型
    
    Returns:
        dict: 包含预测结果和性能指标的字典
    """
    print(f"\n{'='*50}")
    print(f"测试模型: {model_key_or_path}")
    print(f"{'='*50}")
    
    try:
        # 获取模型配置
        config = get_model_config(model_key_or_path, experiment_type)
        
        # 创建模型
        start_time = time.time()
        model = MultiModelTS(config, device)
        model_creation_time = time.time() - start_time
        
        # 将数据移到设备
        x_device = x.to(device)
        y_true_device = y_true.to(device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            y_pred = model(x_device, itr=0)
            inference_time = time.time() - start_time
        
        # 计算性能指标
        mse = nn.MSELoss()(y_pred, y_true_device).item()
        mae = nn.L1Loss()(y_pred, y_true_device).item()
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建成功")
        print(f"📊 模型统计:")
        print(f"   - 模型路径: {config.model_name_or_path}")
        if hasattr(config, 'description'):
            print(f"   - 模型描述: {config.description}")
        print(f"   - 总参数数量: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 参数冻结比例: {(total_params-trainable_params)/total_params*100:.1f}%")
        print(f"⏱️  性能指标:")
        print(f"   - 模型创建时间: {model_creation_time:.3f}s")
        print(f"   - 推理时间: {inference_time:.3f}s")
        print(f"   - MSE Loss: {mse:.6f}")
        print(f"   - MAE Loss: {mae:.6f}")
        
        return {
            'model': model,
            'config': config,
            'prediction': y_pred.cpu(),
            'mse': mse,
            'mae': mae,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_creation_time': model_creation_time,
            'inference_time': inference_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ 模型 {model_key_or_path} 测试失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def visualize_predictions(results, x, y_true, save_path='predictions_comparison.png'):
    """可视化不同模型的预测结果
    
    Args:
        results (dict): 各模型的测试结果
        x (torch.Tensor): 输入数据
        y_true (torch.Tensor): 真实标签
        save_path (str): 保存路径
    """
    successful_models = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_models:
        print("没有成功的模型可供可视化")
        return
    
    num_models = len(successful_models)
    fig, axes = plt.subplots(num_models, 1, figsize=(15, 4*num_models))
    
    if num_models == 1:
        axes = [axes]
    
    # 只显示第一个样本的第一个变量
    sample_idx, var_idx = 0, 0
    
    for idx, (model_key, result) in enumerate(successful_models.items()):
        ax = axes[idx]
        
        # 准备数据
        input_seq = x[sample_idx, :, var_idx].numpy()
        true_pred = y_true[sample_idx, :, var_idx].numpy()
        model_pred = result['prediction'][sample_idx, :, var_idx].numpy()
        
        # 创建时间轴
        input_time = range(len(input_seq))
        pred_time = range(len(input_seq), len(input_seq) + len(true_pred))
        
        # 绘制
        ax.plot(input_time, input_seq, 'b-', label='历史数据', linewidth=2)
        ax.plot(pred_time, true_pred, 'g-', label='真实值', linewidth=2)
        ax.plot(pred_time, model_pred, 'r--', label=f'{model_key} 预测', linewidth=2)
        
        # 添加分界线
        ax.axvline(x=len(input_seq)-1, color='gray', linestyle=':', alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(f'{model_key} 模型预测结果 (MSE: {result["mse"]:.6f})', fontsize=14)
        ax.set_xlabel('时间步')
        ax.set_ylabel('数值')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 预测结果可视化已保存到: {save_path}")


def main():
    """主测试函数"""
    print("🚀 开始多模型时间序列预测测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 显示可用的预设模型
    print("\n")
    print_available_models()
    
    # 创建测试数据
    print(f"\n📊 创建合成测试数据...")
    x, y_true = create_synthetic_data(batch_size=4, seq_len=168, pred_len=24, num_variables=7)
    print(f"   - 输入形状: {x.shape}")
    print(f"   - 标签形状: {y_true.shape}")
    
    # 要测试的模型列表（从预设模型中选择较小的模型进行测试）
    models_to_test = [
        'gpt2',           # GPT-2 base
        'distilbert',     # DistilBERT (较小的BERT)
        'bert-base',      # BERT base
    ]
    
    # 可选：如果用户指定了本地模型路径，也可以测试
    # 例如: models_to_test.append('/path/to/local/model')
    
    print(f"\n🎯 将测试以下模型: {', '.join(models_to_test)}")
    
    # 测试所有模型
    results = {}
    experiment_type = 'quick_test'  # 使用快速测试配置
    
    for model_key in models_to_test:
        results[model_key] = test_single_model(
            model_key, device, x, y_true, experiment_type
        )
    
    # 打印总结
    print(f"\n{'='*60}")
    print("📋 测试总结")
    print(f"{'='*60}")
    
    successful_models = []
    failed_models = []
    
    for model_key, result in results.items():
        if result.get('success', False):
            successful_models.append((model_key, result))
            print(f"✅ {model_key}: MSE={result['mse']:.6f}, 参数={result['total_params']:,}")
        else:
            failed_models.append((model_key, result.get('error', 'Unknown error')))
            print(f"❌ {model_key}: {result.get('error', 'Unknown error')}")
    
    if successful_models:
        # 找出最佳模型
        best_model = min(successful_models, key=lambda x: x[1]['mse'])
        print(f"\n🏆 最佳模型: {best_model[0]} (MSE: {best_model[1]['mse']:.6f})")
        
        # 性能对比表格
        print(f"\n📊 性能对比:")
        print(f"{'模型':<15} {'MSE':<10} {'参数数量':<12} {'推理时间':<10}")
        print("-" * 50)
        for model_key, result in successful_models:
            print(f"{model_key:<15} {result['mse']:<10.6f} {result['total_params']:<12,} {result['inference_time']:<10.3f}s")
        
        # 可视化结果
        try:
            visualize_predictions(results, x, y_true)
        except Exception as e:
            print(f"⚠️  可视化失败: {str(e)}")
    
    if failed_models:
        print(f"\n⚠️  失败的模型:")
        for model_key, error in failed_models:
            print(f"   {model_key}: {error}")
        print(f"\n💡 提示:")
        print(f"   - 某些模型可能需要额外的依赖或权限")
        print(f"   - 可以尝试使用较小的模型或本地模型")
        print(f"   - 检查网络连接是否正常")
    
    print(f"\n🎉 测试完成!")


def test_custom_model(model_path):
    """测试自定义模型的便捷函数
    
    Args:
        model_path (str): 本地模型路径或 Hub 模型名称
    """
    print(f"🧪 测试自定义模型: {model_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    x, y_true = create_synthetic_data(batch_size=2, seq_len=168, pred_len=24, num_variables=7)
    
    # 测试模型
    result = test_single_model(model_path, device, x, y_true, 'quick_test')
    
    if result['success']:
        print(f"✅ 自定义模型测试成功!")
        print(f"   MSE: {result['mse']:.6f}")
        print(f"   总参数: {result['total_params']:,}")
    else:
        print(f"❌ 自定义模型测试失败: {result['error']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，测试指定的模型
        custom_model = sys.argv[1]
        test_custom_model(custom_model)
    else:
        # 否则运行标准测试
        main() 