# 多模型时间序列预测架构

这是一个基于 Hugging Face 统一接口的时间序列预测框架，可以灵活地使用任何基于 Transformer 的预训练模型进行时间序列预测，包括本地模型和 Hub 模型。

## 🌟 主要特性

- **统一接口**: 使用 Hugging Face AutoModel 统一加载所有预训练模型
- **本地模型支持**: 支持从本地路径加载自定义训练的模型
- **丰富的预设模型**: 内置 GPT-2、BERT、DistilBERT、RoBERTa 等多种模型配置
- **灵活配置**: 通过简单的配置文件管理不同的实验设置
- **内存高效**: 智能的参数冻结策略，减少显存占用
- **完整工具链**: 包含训练、测试、可视化等完整工具

## 📁 文件结构

```
├── models/
│   └── GPT4TS.py              # 多模型架构实现 (MultiModelTS)
├── configs/
│   └── multimodel_config.py   # 配置系统
├── test_multimodel.py         # 模型测试脚本
├── train_multimodel.py        # 训练脚本
└── README_MultiModel.md       # 本文档
```

## 🚀 快速开始

### 1. 环境要求

```bash
# 基础依赖
pip install torch torchvision
pip install transformers>=4.21.0
pip install einops
pip install numpy matplotlib

# 可选：用于加速和量化
pip install accelerate
pip install bitsandbytes
```

### 2. 基础使用

```python
from models.GPT4TS import MultiModelTS
from configs.multimodel_config import get_model_config
import torch

# 使用预设模型
config = get_model_config('gpt2', 'quick_test')

# 或使用本地模型
# config = get_model_config('/path/to/local/model', 'small_scale')

# 或使用 Hub 模型
# config = get_model_config('microsoft/DialoGPT-medium', 'quick_test')

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModelTS(config, device)

# 准备数据
batch_size, seq_len, num_variables = 4, 168, 7
x = torch.randn(batch_size, seq_len, num_variables).to(device)

# 预测
predictions = model(x, itr=0)
print(f"预测结果形状: {predictions.shape}")
```

## 🎯 支持的模型

### 预设模型

| 类别       | 模型键名       | 描述         | 参数量 | 推荐用途         |
| ---------- | -------------- | ------------ | ------ | ---------------- |
| **GPT-2**  | `gpt2`         | GPT-2 base   | 124M   | 通用时间序列预测 |
|            | `gpt2-medium`  | GPT-2 medium | 355M   | 复杂模式识别     |
|            | `gpt2-large`   | GPT-2 large  | 774M   | 高精度预测       |
| **BERT**   | `bert-base`    | BERT base    | 110M   | 多变量关联分析   |
|            | `bert-large`   | BERT large   | 340M   | 深度特征提取     |
|            | `distilbert`   | DistilBERT   | 66M    | 轻量级部署       |
|            | `roberta-base` | RoBERTa base | 125M   | 鲁棒性预测       |
| **大模型** | `llama2-7b`    | LLaMA 2 7B   | 7B     | 强大的泛化能力   |
|            | `qwen-7b`      | Qwen 7B      | 7B     | 多语言支持       |
|            | `qwen-1.8b`    | Qwen 1.8B    | 1.8B   | 中等规模任务     |

### 自定义模型

```python
# 本地模型路径
config = get_model_config('/home/user/my_fine_tuned_model', 'small_scale')

# Hugging Face Hub 模型
config = get_model_config('microsoft/DialoGPT-medium', 'quick_test')

# 带自定义参数的本地模型
from configs.multimodel_config import create_local_model_config
config = create_local_model_config(
    '/path/to/local/model',
    model_layers=4,
    batch_size=8,
    trust_remote_code=True
)
```

## 🛠️ 配置系统

### 查看可用模型

```python
from configs.multimodel_config import print_available_models
print_available_models()
```

### 实验规模配置

| 规模          | 序列长度 | 预测长度 | 训练轮数 | 适用场景   |
| ------------- | -------- | -------- | -------- | ---------- |
| `quick_test`  | 168      | 24       | 2        | 快速验证   |
| `small_scale` | 336      | 96       | 5        | 小规模实验 |
| `full_scale`  | 720      | 192      | 20       | 完整训练   |

### 自定义配置

```python
from types import SimpleNamespace

# 完全自定义配置
custom_config = SimpleNamespace(
    model_name_or_path='bert-base-uncased',  # 模型路径
    seq_len=480,            # 输入序列长度
    pred_len=120,           # 预测长度
    patch_size=20,          # Patch大小
    stride=10,              # 步长
    model_layers=8,         # 使用层数
    batch_size=16,          # 批次大小
    learning_rate=2e-5,     # 学习率
    freeze=True,            # 参数冻结
    trust_remote_code=False # 是否信任远程代码
)
```

## 🧪 测试模型

### 运行预设模型测试

```bash
# 测试默认的预设模型
python test_multimodel.py

# 测试特定模型
python test_multimodel.py bert-base-uncased

# 测试本地模型
python test_multimodel.py /path/to/local/model
```

### 程序化测试

```python
from test_multimodel import test_custom_model

# 测试本地模型
test_custom_model('/path/to/my/model')

# 测试 Hub 模型  
test_custom_model('microsoft/DialoGPT-small')
```

### 测试输出示例

```
🚀 开始多模型时间序列预测测试
============================================================
🔧 使用设备: cuda

🎯 可用的预设模型:
============================================================

📚 GPT-2 系列:
   gpt2            - GPT-2 base (124M 参数)
   gpt2-medium     - GPT-2 medium (355M 参数)
   gpt2-large      - GPT-2 large (774M 参数)

📚 BERT 系列:
   bert-base       - BERT base uncased (110M 参数)
   bert-large      - BERT large uncased (340M 参数)
   distilbert      - DistilBERT base (66M 参数)
   roberta-base    - RoBERTa base (125M 参数)

==================================================
测试模型: gpt2
==================================================
📋 使用预设模型配置: gpt2
📝 模型描述: GPT-2 base (124M 参数)
🔄 正在加载模型: gpt2
✅ 成功加载模型: gpt2
📊 模型类型: GPT2Model
📏 检测到模型隐藏维度: 768 (来自 config.n_embd)
✅ 模型创建成功
📊 模型统计:
   - 模型路径: gpt2
   - 模型描述: GPT-2 base (124M 参数)
   - 总参数数量: 124,439,808
   - 可训练参数: 1,574,400
   - 参数冻结比例: 98.7%
⏱️  性能指标:
   - 模型创建时间: 2.341s
   - 推理时间: 0.145s
   - MSE Loss: 0.234567
   - MAE Loss: 0.123456

🏆 最佳模型: gpt2 (MSE: 0.234567)
📈 预测结果可视化已保存到: predictions_comparison.png
```

## 🏋️ 训练模型

### 查看可用模型

```bash
# 显示所有预设模型
python train_multimodel.py --list_models
```

### 基础训练

```bash
# 训练预设模型
python train_multimodel.py --model gpt2 --experiment quick_test
python train_multimodel.py --model bert-base --experiment small_scale
python train_multimodel.py --model distilbert --experiment full_scale

# 训练本地模型
python train_multimodel.py --model /path/to/local/model --experiment small_scale

# 训练 Hub 模型
python train_multimodel.py --model microsoft/DialoGPT-medium --experiment quick_test
```

### 高级训练选项

```bash
# 完整命令示例
python train_multimodel.py \
    --model bert-base-uncased \
    --experiment small_scale \
    --save_dir ./my_checkpoints \
    --device cuda
```

### 训练输出示例

```
📋 使用预设模型配置: gpt2
📝 模型描述: GPT-2 base (124M 参数)
🔄 正在加载模型: gpt2
✅ 成功加载模型: gpt2
📊 模型类型: GPT2Model
📏 检测到模型隐藏维度: 768 (来自 config.n_embd)
🔧 限制模型层数为: 2
✅ 成功限制 h 层数: 12 -> 2
🧊 应用参数冻结策略...
📊 参数冻结统计:
   - 总参数数: 24,515,584
   - 冻结参数: 23,592,960
   - 可训练参数: 922,624
   - 冻结比例: 96.2%

🚀 开始训练模型: gpt2
============================================================
📊 创建数据加载器...
   - 训练样本: 1000
   - 验证样本: 200

📈 Epoch 1/2
----------------------------------------
    Batch 0/250, Loss: 1.234567
   训练损失: 0.876543
   验证损失: 0.765432
   学习率: 1.00e-04
   用时: 12.34s
   ✅ 新的最佳模型已保存 (验证损失: 0.765432)

🎉 训练完成!
   最佳验证损失: 0.543210
```

## 📊 模型配置详解

### 智能参数检测

框架会自动检测模型的关键参数：

```python
# 自动检测隐藏维度
hidden_size_attrs = ['hidden_size', 'd_model', 'n_embd', 'dim', 'model_dim']

# 自动限制模型层数
layer_attrs = ['layers', 'layer', 'h', 'encoder.layer', 'transformer.h']
```

### 通用冻结策略

```python
# 只训练这些类型的参数
trainable_patterns = [
    'norm',           # 各种归一化层
    'ln',             # LayerNorm (GPT 风格)
    'layer_norm',     # LayerNorm (BERT 风格)
    'rmsnorm',        # RMSNorm (LLaMA 风格)
    'embed',          # 嵌入层
    'position',       # 位置编码
]
```

## 🎯 最佳实践

### 模型选择建议

| 任务特点       | 推荐模型      | 配置建议            |
| -------------- | ------------- | ------------------- |
| **快速原型**   | `distilbert`  | `quick_test`        |
| **标准预测**   | `gpt2`        | `small_scale`       |
| **高精度需求** | `bert-base`   | `full_scale`        |
| **资源受限**   | `distilbert`  | 减少 `batch_size`   |
| **复杂模式**   | `gpt2-medium` | 增加 `model_layers` |

### 本地模型使用

```bash
# 1. 准备本地模型目录
/path/to/my/model/
├── config.json
├── pytorch_model.bin
└── tokenizer.json

# 2. 创建配置
python -c "
from configs.multimodel_config import create_local_model_config
config = create_local_model_config('/path/to/my/model')
print(config.model_name_or_path)
"

# 3. 开始训练
python train_multimodel.py --model /path/to/my/model
```

### 性能优化

```python
# 减少显存占用
config.batch_size = 4
config.model_layers = 2
config.freeze = True

# 加速训练
config.num_epochs = 5
config.patience = 2

# 提高精度
config.model_layers = 8
config.seq_len = 720
```

## 🐛 常见问题

### Q1: 如何添加新的预设模型？

```python
# 在 configs/multimodel_config.py 中添加
NEW_MODELS = {
    'my-model': {
        'model_name_or_path': 'my-org/my-model',
        'description': '我的自定义模型',
        'd_model': 768,
        'model_layers': 6,
        'learning_rate': 1e-4,
        'batch_size': 16,
    }
}

# 然后添加到 PRESET_MODELS
PRESET_MODELS = {
    **GPT2_MODELS,
    **BERT_MODELS,
    **LLM_MODELS,
    **NEW_MODELS  # 添加这行
}
```

### Q2: 模型加载失败

```
❌ 模型加载失败: my-model
错误信息: Can't load tokenizer for 'my-model'
```

**解决方案:**
```python
# 设置 trust_remote_code=True
config = get_model_config('my-model', 'quick_test')
config.trust_remote_code = True

# 或者直接在预设中配置
'trust_remote_code': True
```

### Q3: 显存不足

```python
# 方案1: 减少批次大小
config.batch_size = 2

# 方案2: 减少模型层数
config.model_layers = 2

# 方案3: 使用梯度检查点
# 在模型训练时设置
torch.utils.checkpoint.checkpoint_sequential = True
```

### Q4: 本地模型路径问题

```bash
# Windows 路径
python train_multimodel.py --model "C:\Users\Name\model"

# Linux/Mac 路径  
python train_multimodel.py --model "/home/user/model"

# 相对路径
python train_multimodel.py --model "./models/my_model"
```

## 📈 扩展功能

### 自定义数据加载器

```python
def create_custom_dataloader(self, data_path):
    """加载真实数据"""
    # 加载你的时间序列数据
    data = np.load(data_path)
    # ... 数据预处理
    return dataloader
```

### 自定义损失函数

```python
class CustomLoss(nn.Module):
    def forward(self, pred, true):
        mse = nn.MSELoss()(pred, true)
        mae = nn.L1Loss()(pred, true)
        return mse + 0.1 * mae

# 在训练器中使用
self.criterion = CustomLoss()
```

### 集成其他模型

```python
# 在配置中添加新模型
'custom-transformer': {
    'model_name_or_path': 'your-org/custom-transformer',
    'description': '自定义 Transformer 模型',
    'trust_remote_code': True,
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **Hugging Face Transformers**: 提供统一的模型接口
- **PyTorch**: 深度学习框架
- **各个预训练模型的原作者们**

---

**快速上手命令:**

```bash
# 查看所有可用模型
python train_multimodel.py --list_models

# 快速测试
python test_multimodel.py

# 开始训练
python train_multimodel.py --model gpt2 --experiment quick_test
```

如有问题或建议，请提交 Issue 或联系维护者。祝您使用愉快！ 🎉 