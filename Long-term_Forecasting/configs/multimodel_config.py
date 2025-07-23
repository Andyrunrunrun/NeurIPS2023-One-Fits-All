#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型时间序列预测配置文件
使用 Hugging Face 统一接口，支持本地模型路径和 Hub 模型名称
"""

from types import SimpleNamespace
import os

# ============= 基础配置模板 =============
BASE_CONFIG = {
    # 数据相关配置
    'seq_len': 336,          # 输入序列长度
    'pred_len': 96,          # 预测序列长度
    'patch_size': 16,        # 每个patch的大小
    'stride': 8,             # patch的步长
    
    # 训练相关配置
    'batch_size': 32,        # 批次大小
    'learning_rate': 1e-4,   # 学习率
    'num_epochs': 10,        # 训练轮数
    'patience': 3,           # 早停耐心值
    
    # 模型相关配置
    'freeze': True,          # 是否冻结部分参数
    'dropout': 0.1,          # Dropout率
    'trust_remote_code': False,  # 是否信任远程代码
    
    # 其他配置
    'device': 'auto',        # 设备选择 ('auto', 'cpu', 'cuda')
    'seed': 42,              # 随机种子
}


# ============= 预设模型配置 =============

# GPT-2 系列模型配置
GPT2_MODELS = {
    'gpt2': {
        'model_name_or_path': 'gpt2',
        'description': 'GPT-2 base (124M 参数)',
        'd_model': 768,
        'model_layers': 6,
        'learning_rate': 1e-4,
        'batch_size': 32,
    },
    'gpt2-medium': {
        'model_name_or_path': 'gpt2-medium',  
        'description': 'GPT-2 medium (355M 参数)',
        'd_model': 1024,
        'model_layers': 4,
        'learning_rate': 5e-5,
        'batch_size': 16,
    },
    'gpt2-large': {
        'model_name_or_path': 'gpt2-large',
        'description': 'GPT-2 large (774M 参数)',
        'd_model': 1280,
        'model_layers': 3,
        'learning_rate': 2e-5,
        'batch_size': 8,
    },
}

# BERT 系列模型配置
BERT_MODELS = {
    'bert-base': {
        'model_name_or_path': 'bert-base-uncased',
        'description': 'BERT base uncased (110M 参数)',
        'd_model': 768,
        'model_layers': 6,
        'learning_rate': 2e-5,
        'batch_size': 16,
    },
    'bert-large': {
        'model_name_or_path': 'bert-large-uncased',
        'description': 'BERT large uncased (340M 参数)',
        'd_model': 1024,
        'model_layers': 4,
        'learning_rate': 1e-5,
        'batch_size': 8,
    },
    'distilbert': {
        'model_name_or_path': 'distilbert-base-uncased',
        'description': 'DistilBERT base (66M 参数)',
        'd_model': 768,
        'model_layers': 6,
        'learning_rate': 3e-5,
        'batch_size': 32,
    },
    'roberta-base': {
        'model_name_or_path': 'roberta-base',
        'description': 'RoBERTa base (125M 参数)',
        'd_model': 768,
        'model_layers': 6,
        'learning_rate': 2e-5,
        'batch_size': 16,
    },
}

# 大语言模型配置
LLM_MODELS = {
    'llama2-7b': {
        'model_name_or_path': 'meta-llama/Llama-2-7b-hf',
        'description': 'LLaMA 2 7B (7B 参数)',
        'd_model': 4096,
        'model_layers': 4,
        'learning_rate': 1e-5,
        'batch_size': 4,
        'trust_remote_code': True,
    },
    'qwen-7b': {
        'model_name_or_path': 'Qwen/Qwen-7B',
        'description': 'Qwen 7B (7B 参数)',
        'd_model': 4096,
        'model_layers': 4,
        'learning_rate': 1e-5,
        'batch_size': 4,
        'trust_remote_code': True,
    },
    'qwen-1.8b': {
        'model_name_or_path': 'Qwen/Qwen-1_8B',
        'description': 'Qwen 1.8B (1.8B 参数)',
        'd_model': 2048,
        'model_layers': 6,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'trust_remote_code': True,
    },
}

# 合并所有预设模型
PRESET_MODELS = {
    **GPT2_MODELS,
    **BERT_MODELS, 
    **LLM_MODELS
}


# ============= 实验配置组合 =============
EXPERIMENT_CONFIGS = {
    'quick_test': {
        # 快速测试配置（适用于所有模型）
        'seq_len': 168,      # 较短的序列
        'pred_len': 24,      # 较短的预测
        'num_epochs': 2,     # 少量轮数
        'batch_size': 4,     # 小批次
        'model_layers': 2,   # 少量层数
    },
    
    'small_scale': {
        # 小规模实验配置
        'seq_len': 336,
        'pred_len': 96,
        'num_epochs': 5,
        'batch_size': 8,
        'model_layers': 4,
    },
    
    'full_scale': {
        # 完整规模实验配置
        'seq_len': 720,      # 更长的序列
        'pred_len': 192,     # 更长的预测
        'num_epochs': 20,
        'batch_size': 16,
        'model_layers': 8,
    }
}


def get_model_config(model_key_or_path, experiment_type='small_scale'):
    """获取指定模型和实验类型的配置
    
    Args:
        model_key_or_path (str): 预设模型键名或模型路径
            - 预设模型: 'gpt2', 'bert-base', 'llama2-7b' 等
            - 本地路径: '/path/to/local/model'
            - Hub 名称: 'microsoft/DialoGPT-medium'
        experiment_type (str): 实验类型 ('quick_test', 'small_scale', 'full_scale')
    
    Returns:
        SimpleNamespace: 配置对象
    
    Examples:
        ```python
        # 使用预设模型
        config = get_model_config('gpt2', 'quick_test')
        
        # 使用本地模型路径
        config = get_model_config('/home/user/my_model', 'small_scale')
        
        # 使用 Hub 模型名称
        config = get_model_config('microsoft/DialoGPT-medium', 'full_scale')
        ```
    """
    # 获取实验配置
    if experiment_type in EXPERIMENT_CONFIGS:
        experiment_config = EXPERIMENT_CONFIGS[experiment_type]
    else:
        raise ValueError(f"不支持的实验类型: {experiment_type}")
    
    # 检查是否为预设模型
    if model_key_or_path in PRESET_MODELS:
        # 使用预设模型配置
        model_config = PRESET_MODELS[model_key_or_path].copy()
        print(f"📋 使用预设模型配置: {model_key_or_path}")
        print(f"📝 模型描述: {model_config.get('description', 'N/A')}")
    else:
        # 使用自定义路径或名称
        model_config = {
            'model_name_or_path': model_key_or_path,
            'description': f'自定义模型: {model_key_or_path}',
            'd_model': 768,  # 默认值，会在运行时自动检测
            'model_layers': 6,
            'learning_rate': 1e-4,
            'batch_size': 16,
        }
        print(f"📋 使用自定义模型: {model_key_or_path}")
        
        # 如果是本地路径，检查是否存在
        if os.path.exists(model_key_or_path):
            print(f"✅ 检测到本地模型路径: {model_key_or_path}")
        else:
            print(f"🌐 将尝试从 Hugging Face Hub 加载: {model_key_or_path}")
    
    # 合并基础配置、模型配置和实验配置
    final_config = {
        **BASE_CONFIG,
        **model_config,
        **experiment_config
    }
    
    return SimpleNamespace(**final_config)


def get_preset_model_list():
    """获取所有预设模型的列表
    
    Returns:
        dict: 按类别组织的模型列表
    """
    return {
        'GPT-2 系列': {k: v['description'] for k, v in GPT2_MODELS.items()},
        'BERT 系列': {k: v['description'] for k, v in BERT_MODELS.items()},
        '大语言模型': {k: v['description'] for k, v in LLM_MODELS.items()},
    }


def create_local_model_config(local_path, **kwargs):
    """为本地模型创建配置
    
    Args:
        local_path (str): 本地模型路径
        **kwargs: 其他配置参数
    
    Returns:
        SimpleNamespace: 配置对象
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"本地模型路径不存在: {local_path}")
    
    config_dict = {
        **BASE_CONFIG,
        'model_name_or_path': local_path,
        'description': f'本地模型: {local_path}',
        'd_model': 768,  # 默认值，会自动检测
        'model_layers': 6,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'trust_remote_code': True,  # 本地模型通常需要信任代码
        **kwargs  # 覆盖默认参数
    }
    
    return SimpleNamespace(**config_dict)


def print_config_summary(config):
    """打印配置摘要
    
    Args:
        config (SimpleNamespace): 配置对象
    """
    print(f"📊 模型配置摘要:")
    print(f"   模型路径: {config.model_name_or_path}")
    if hasattr(config, 'description'):
        print(f"   模型描述: {config.description}")
    print(f"   隐藏维度: {config.d_model}")
    print(f"   模型层数: {config.model_layers}")
    print(f"   序列长度: {config.seq_len}")
    print(f"   预测长度: {config.pred_len}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   训练轮数: {config.num_epochs}")
    print(f"   参数冻结: {config.freeze}")
    print(f"   信任远程代码: {config.trust_remote_code}")


def print_available_models():
    """打印所有可用的预设模型"""
    print("🎯 可用的预设模型:")
    print("=" * 60)
    
    model_categories = get_preset_model_list()
    
    for category, models in model_categories.items():
        print(f"\n📚 {category}:")
        for key, description in models.items():
            print(f"   {key:<15} - {description}")
    
    print(f"\n💡 使用方法:")
    print(f"   config = get_model_config('gpt2', 'quick_test')")
    print(f"   config = get_model_config('/path/to/local/model', 'small_scale')")


if __name__ == "__main__":
    print("=== 多模型时间序列预测配置系统 ===\n")
    
    # 显示可用模型
    print_available_models()
    
    print(f"\n" + "=" * 60)
    print("📋 配置示例:")
    print("=" * 60)
    
    # 演示不同的配置方式
    examples = [
        ('gpt2', 'quick_test'),
        ('bert-base', 'small_scale'), 
        ('distilbert', 'quick_test'),
    ]
    
    for model_key, exp_type in examples:
        print(f"\n--- {model_key.upper()} ({exp_type}) ---")
        try:
            config = get_model_config(model_key, exp_type)
            print_config_summary(config)
        except Exception as e:
            print(f"❌ 配置失败: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print("🔧 本地模型配置示例:")
    print("=" * 60)
    print("# 为本地模型创建配置")
    print("config = create_local_model_config(")
    print("    '/path/to/my/model',")
    print("    model_layers=4,")
    print("    batch_size=8")
    print(")")
    print("\n# 或者直接使用路径")
    print("config = get_model_config('/path/to/my/model', 'small_scale')") 