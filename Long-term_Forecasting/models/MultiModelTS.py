import numpy as np
import torch
import torch.nn as nn
from torch import optim

# 使用 Hugging Face 的统一接口
from transformers import AutoModel, AutoConfig
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time


class MultiModelTS(nn.Module):
    """MultiModelTS: 支持多种预训练模型的时间序列预测架构
    
    这是一个灵活的时间序列预测框架，使用 Hugging Face 的统一接口支持各种预训练模型。
    通过将时间序列数据转换为 patch 表示，利用预训练模型强大的表征学习能力来进行长期时间序列预测。
    
    理论依据:
        - 基于 "Attention Is All You Need" 的 Transformer 架构
        - 借鉴 Vision Transformer (ViT) 的 patch 分割策略  
        - 利用预训练模型的领域知识迁移能力
        - 实现跨模态知识迁移：从语言建模到时间序列预测
    
    支持的模型类型:
        - 任何基于 Transformer 的预训练模型（通过 AutoModel 接口）
        - 本地模型路径或 Hugging Face Hub 模型名称
        - GPT-2, BERT, LLaMA, Qwen, RoBERTa, DistilBERT 等
    
    主要组件:
        - Patch 分割层: 将时间序列切分为固定大小的 patch
        - 线性投影层: 将 patch 映射到模型的隐藏维度
        - 预训练模型编码器: 提供强大的序列建模能力
        - 输出投影层: 将编码后的特征映射回预测长度
    
    输入张量形状:
        x: (batch_size, seq_len, num_variables) - 输入的多变量时间序列
    
    输出张量形状:
        outputs: (batch_size, pred_len, num_variables) - 预测的多变量时间序列
    
    使用示例:
        ```python
        import torch
        from types import SimpleNamespace
        
        # 配置参数 - 使用本地模型
        configs = SimpleNamespace(
            model_name_or_path='/path/to/local/model',  # 本地模型路径
            # 或者使用 Hub 模型名称
            # model_name_or_path='bert-base-uncased',
            patch_size=16,
            stride=8,
            seq_len=336,
            pred_len=96,
            d_model=768,
            model_layers=6,
            freeze=True,
            trust_remote_code=True  # 对于某些模型可能需要
        )
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiModelTS(configs, device)
        
        # 准备输入数据
        batch_size, seq_len, num_variables = 32, 336, 7
        x = torch.randn(batch_size, seq_len, num_variables).to(device)
        
        # 前向传播
        predictions = model(x, itr=0)  # 输出形状: (32, 96, 7)
        ```
    """
    
    def __init__(self, configs, device):
        """初始化 MultiModelTS 模型
        
        Args:
            configs: 模型配置对象，包含以下属性:
                model_name_or_path (str): 模型名称或本地路径
                patch_size (int): 每个 patch 的大小（时间步数）
                stride (int): patch 分割时的步长
                seq_len (int): 输入序列长度
                pred_len (int): 预测长度
                d_model (int): 模型的隐藏维度
                model_layers (int): 使用的模型层数
                freeze (bool): 是否冻结模型的部分参数
                trust_remote_code (bool, optional): 是否信任远程代码
                use_auth_token (bool/str, optional): Hugging Face 认证令牌
            device (torch.device): 运行设备 (CPU/GPU)
        """
        super(MultiModelTS, self).__init__()
        
        # (1) 保存核心配置参数
        self.model_name_or_path = configs.model_name_or_path
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        
        # (2) 计算 patch 数量：使用滑动窗口方式分割时间序列
        # 公式: (序列长度 - patch大小) // 步长 + 1
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        # (3) 创建填充层，确保最后一个 patch 能够完整提取
        # ReplicationPad1d(0, stride) 在序列末尾复制最后 stride 个元素
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1  # 因为填充后会多出一个 patch
        
        # (4) 加载预训练模型（使用统一的 AutoModel 接口）
        self.model = self._load_pretrained_model(configs)
        
        # (5) 获取模型的实际隐藏维度
        self.d_model = self._get_model_hidden_size()
        
        # (6) 限制使用的层数（如果需要）
        if hasattr(configs, 'model_layers') and configs.model_layers > 0:
            self._limit_model_layers(configs.model_layers)
        
        # (7) 创建输入投影层：patch_size -> d_model
        # 将每个 patch 映射到模型的隐藏维度空间
        self.in_layer = nn.Linear(configs.patch_size, self.d_model)
        
        # (8) 创建输出投影层：(d_model * patch_num) -> pred_len
        # 将所有 patch 的编码特征拼接后映射到预测长度
        self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
        
        # (9) 参数冻结策略
        if hasattr(configs, 'freeze') and configs.freeze:
            self._apply_freeze_strategy()

        # (10) 将所有模块移动到指定设备并设置为训练模式
        for layer in (self.model, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def _load_pretrained_model(self, configs):
        """使用 AutoModel 统一加载预训练模型
        
        Args:
            configs: 配置对象
            
        Returns:
            torch.nn.Module: 加载的预训练模型
        """
        print(f"🔄 正在加载模型: {configs.model_name_or_path}")
        
        try:
            # 准备加载参数
            load_kwargs = {
                'output_attentions': True,
                'output_hidden_states': True,
            }
            
            # 添加可选参数
            if hasattr(configs, 'trust_remote_code') and configs.trust_remote_code:
                load_kwargs['trust_remote_code'] = True
                
            if hasattr(configs, 'use_auth_token') and configs.use_auth_token:
                load_kwargs['use_auth_token'] = configs.use_auth_token
            
            # 尝试加载模型
            model = AutoModel.from_pretrained(
                configs.model_name_or_path,
                **load_kwargs
            )
            
            print(f"✅ 成功加载模型: {configs.model_name_or_path}")
            print(f"📊 模型类型: {model.__class__.__name__}")
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {configs.model_name_or_path}")
            print(f"错误信息: {str(e)}")
            print("💡 请检查模型路径是否正确，或者是否需要设置 trust_remote_code=True")
            raise e

    def _get_model_hidden_size(self):
        """获取模型的隐藏维度大小
        
        Returns:
            int: 隐藏维度大小
        """
        # 尝试从配置中获取隐藏维度
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # 常见的隐藏维度属性名
            hidden_size_attrs = ['hidden_size', 'd_model', 'n_embd', 'dim', 'model_dim']
            
            for attr in hidden_size_attrs:
                if hasattr(config, attr):
                    hidden_size = getattr(config, attr)
                    print(f"📏 检测到模型隐藏维度: {hidden_size} (来自 config.{attr})")
                    return hidden_size
        
        # 如果无法从配置获取，尝试推断
        print("⚠️  无法从配置获取隐藏维度，尝试推断...")
        
        # 创建一个小的测试输入来推断维度
        test_input = torch.randn(1, 8, 768)  # 假设的输入
        
        try:
            with torch.no_grad():
                output = self.model(inputs_embeds=test_input)
                if hasattr(output, 'last_hidden_state'):
                    hidden_size = output.last_hidden_state.shape[-1]
                    print(f"📏 通过推断得到隐藏维度: {hidden_size}")
                    return hidden_size
        except:
            pass
        
        # 默认值
        default_size = 768
        print(f"⚠️  无法确定隐藏维度，使用默认值: {default_size}")
        return default_size

    def _limit_model_layers(self, max_layers):
        """限制模型使用的层数
        
        Args:
            max_layers (int): 最大层数
        """
        print(f"🔧 限制模型层数为: {max_layers}")
        
        # 尝试不同的层属性名称
        layer_attrs = ['layers', 'layer', 'h', 'encoder.layer', 'transformer.h']
        
        for attr_path in layer_attrs:
            try:
                # 支持嵌套属性
                obj = self.model
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                
                if hasattr(obj, '__len__') and len(obj) > max_layers:
                    # 截断层数
                    if hasattr(obj, '__setitem__'):
                        # 对于 ModuleList
                        new_layers = obj[:max_layers]
                        obj.clear()
                        obj.extend(new_layers)
                    else:
                        # 对于其他类型，尝试直接赋值
                        setattr(obj, attr_path.split('.')[-1], obj[:max_layers])
                    
                    print(f"✅ 成功限制 {attr_path} 层数: {len(obj)} -> {max_layers}")
                    return
                    
            except (AttributeError, TypeError):
                continue
        
        print("⚠️  无法限制模型层数，将使用完整模型")

    def _apply_freeze_strategy(self):
        """应用通用的参数冻结策略
        
        冻结大部分参数，只训练归一化层和嵌入层相关的参数
        """
        print("🧊 应用参数冻结策略...")
        
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # 定义需要训练的参数模式
            trainable_patterns = [
                'norm',           # 各种归一化层
                'ln',             # LayerNorm (GPT 风格)
                'layer_norm',     # LayerNorm (BERT 风格)
                'rmsnorm',        # RMSNorm (LLaMA 风格)
                'embed',          # 嵌入层
                'position',       # 位置编码
                'wpe',            # 位置嵌入 (GPT 风格)
                'word_embed',     # 词嵌入
                'token_embed',    # Token嵌入
            ]
            
            # 检查参数名是否匹配可训练模式
            should_train = any(pattern in name.lower() for pattern in trainable_patterns)
            
            if should_train:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        freeze_ratio = frozen_params / total_params * 100
        print(f"📊 参数冻结统计:")
        print(f"   - 总参数数: {total_params:,}")
        print(f"   - 冻结参数: {frozen_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 冻结比例: {freeze_ratio:.1f}%")

    def forward(self, x, itr):
        """MultiModelTS 模型的前向传播过程
        
        实现完整的时间序列预测流程，支持任何基于 Transformer 的预训练模型。
        
        Args:
            x (torch.Tensor): 输入的多变量时间序列
                形状: (batch_size, seq_len, num_variables)
            itr (int): 当前迭代次数（用于调试，模型内部未使用）
        
        Returns:
            torch.Tensor: 预测的多变量时间序列
                形状: (batch_size, pred_len, num_variables)
        
        算法流程详解:
            1. 数据标准化（Z-score）
            2. 维度重排列，便于 patch 操作
            3. 序列填充和 patch 分割
            4. 预训练模型编码（统一接口）
            5. 预测生成和反标准化
        """
        # 获取输入张量的维度信息
        B, L, M = x.shape  # B=batch_size, L=seq_len, M=num_variables

        # ==================== (1) 数据标准化 ====================
        # 计算每个样本在时间维度上的均值，用于中心化
        # (B, L, M) -> (B, 1, M)
        means = x.mean(1, keepdim=True).detach()
        x = x - means  # 零均值化
        
        # 计算每个样本在时间维度上的标准差，用于标准化
        # (B, L, M) -> (B, 1, M)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev  # 标准化到单位方差

        # ==================== (2) 维度重排列 ====================
        # 将时间维度移到最后，便于后续的 patch 操作
        # (B, L, M) -> (B, M, L)
        x = rearrange(x, 'b l m -> b m l')

        # ==================== (3) Patch 分割 ====================
        # 在序列末尾填充，确保能提取完整的 patch
        # (B, M, L) -> (B, M, L + stride)
        x = self.padding_patch_layer(x)
        
        # 使用滑动窗口提取 patch
        # unfold(dimension=-1, size=patch_size, step=stride)
        # (B, M, L + stride) -> (B, M, patch_num, patch_size)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        # 重新排列维度，将 batch 和变量维度合并
        # (B, M, patch_num, patch_size) -> (B*M, patch_num, patch_size)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # ==================== (4) 线性投影到模型隐藏空间 ====================
        # 将每个 patch 映射到 d_model 维度
        # (B*M, patch_num, patch_size) -> (B*M, patch_num, d_model)
        outputs = self.in_layer(x)
        
        # ==================== (5) 预训练模型编码 ====================
        outputs = self._encode_with_model(outputs)

        # ==================== (6) 预测生成 ====================
        # 将所有 patch 的特征展平后进行最终预测
        # (B*M, patch_num, d_model) -> (B*M, patch_num * d_model) -> (B*M, pred_len)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        
        # 恢复原始的 batch 和变量维度
        # (B*M, pred_len) -> (B, pred_len, M)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        # ==================== (7) 反标准化 ====================
        # 恢复原始的尺度和均值
        outputs = outputs * stdev  # 恢复标准差
        outputs = outputs + means  # 恢复均值

        return outputs
    
    def _encode_with_model(self, x):
        """使用预训练模型进行编码（统一接口）
        
        Args:
            x (torch.Tensor): 输入的 patch 嵌入
                形状: (B*M, patch_num, d_model)
        
        Returns:
            torch.Tensor: 编码后的特征
                形状: (B*M, patch_num, d_model)
        """
        try:
            # 使用 inputs_embeds 参数进行编码
            outputs = self.model(inputs_embeds=x)
            
            # 尝试获取最后一层的隐藏状态
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                return outputs.hidden_states[-1]  # 取最后一层
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0]  # 通常第一个元素是主要输出
            else:
                # 如果都不行，直接返回 outputs
                return outputs
                
        except Exception as e:
            print(f"⚠️  模型编码过程中出现错误: {str(e)}")
            print("💡 尝试使用备用方法...")
            
            # 备用方法：如果模型不支持 inputs_embeds，尝试其他方式
            try:
                # 有些模型可能需要不同的输入参数
                outputs = self.model(hidden_states=x)
                if hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state
                return outputs[0] if isinstance(outputs, tuple) else outputs
            except:
                # 最后的备用方案：直接返回输入（相当于跳过编码）
                print("❌ 无法使用预训练模型编码，将跳过编码步骤")
                return x


# 为了向后兼容，保留原始类名作为别名
GPT4TS = MultiModelTS
