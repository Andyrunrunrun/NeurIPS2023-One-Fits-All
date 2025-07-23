import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class GPT4TS(nn.Module):
    """GPT4TS: 基于 GPT-2 架构的时间序列预测模型
    
    这是一个将预训练的 GPT-2 语言模型适配到时间序列长期预测任务的神经网络架构。
    该模型的核心思想是将连续的时间序列数据分割成固定大小的 patch（类似于 Vision Transformer 中的做法），
    然后将这些 patch 作为 token 输入到 GPT-2 模型中进行编码和预测。
    
    理论依据:
        - 基于 "Attention Is All You Need" 论文中的 Transformer 架构
        - 借鉴了 GPT-2 的自回归语言建模能力
        - 采用了 Vision Transformer (ViT) 中的 patch 分割策略
        - 实现了 "GPT4TS: Leveraging Pre-trained Language Model for Time Series Forecasting" 的核心思想
    
    主要组件:
        - Patch 分割层: 将时间序列切分为固定大小的 patch
        - 线性投影层: 将 patch 映射到 GPT-2 的隐藏维度
        - GPT-2 编码器: 提供强大的序列建模能力
        - 输出投影层: 将编码后的特征映射回预测长度
    
    输入张量形状:
        x: (batch_size, seq_len, num_variables) - 输入的多变量时间序列
    
    输出张量形状:
        outputs: (batch_size, pred_len, num_variables) - 预测的多变量时间序列
    
    使用示例:
        ```python
        import torch
        from types import SimpleNamespace
        
        # 配置参数
        configs = SimpleNamespace(
            is_gpt=True,
            patch_size=16,
            pretrain=True,
            stride=8,
            seq_len=336,
            pred_len=96,
            d_model=768,
            gpt_layers=6,
            freeze=True
        )
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GPT4TS(configs, device)
        
        # 准备输入数据
        batch_size, seq_len, num_variables = 32, 336, 7
        x = torch.randn(batch_size, seq_len, num_variables).to(device)
        
        # 前向传播
        predictions = model(x, itr=0)  # 输出形状: (32, 96, 7)
        ```
    """
    
    def __init__(self, configs, device):
        """初始化 GPT4TS 模型
        
        Args:
            configs: 模型配置对象，包含以下属性:
                is_gpt (bool): 是否使用 GPT-2 模型
                patch_size (int): 每个 patch 的大小（时间步数）
                pretrain (bool): 是否使用预训练的 GPT-2 权重
                stride (int): patch 分割时的步长
                seq_len (int): 输入序列长度
                pred_len (int): 预测长度
                d_model (int): GPT-2 的隐藏维度
                gpt_layers (int): 使用的 GPT-2 层数
                freeze (bool): 是否冻结 GPT-2 的部分参数
            device (torch.device): 运行设备 (CPU/GPU)
        """
        super(GPT4TS, self).__init__()
        
        # (1) 保存核心配置参数
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        
        # (2) 计算 patch 数量：使用滑动窗口方式分割时间序列
        # 公式: (序列长度 - patch大小) // 步长 + 1
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        # (3) 创建填充层，确保最后一个 patch 能够完整提取
        # ReplicationPad1d(0, stride) 在序列末尾复制最后 stride 个元素
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1  # 因为填充后会多出一个 patch
        
        # (4) 初始化 GPT-2 模型（如果启用）
        if configs.is_gpt:
            if configs.pretrain:
                # 加载预训练的 GPT-2 base 模型，启用注意力和隐藏状态输出
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            else:
                print("------------------no pretrain------------------")
                # 使用随机初始化的 GPT-2 配置
                self.gpt2 = GPT2Model(GPT2Config())
            
            # (5) 根据 gpt_layers 的正负决定保留哪些 Transformer 层
            # 正数: 保留前 gpt_layers 层；负数: 保留最后 |gpt_layers| 层
            if configs.gpt_layers >= 0:
                # (5.1) 保留前 gpt_layers 层
                self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
                # 形状变化: [全部层] -> [前 gpt_layers 层]
            else:
                # (5.2) 保留最后 |gpt_layers| 层
                self.gpt2.h = self.gpt2.h[configs.gpt_layers:]
                # 形状变化: [全部层] -> [后 |gpt_layers| 层]
            print("gpt2 = {}".format(self.gpt2))
        
        # (6) 创建输入投影层：patch_size -> d_model
        # 将每个 patch 映射到 GPT-2 的隐藏维度空间
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        
        # (7) 创建输出投影层：(d_model * patch_num) -> pred_len
        # 将所有 patch 的编码特征拼接后映射到预测长度
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        # (8) 参数冻结策略（仅在使用预训练模型时）
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:  # 只训练 LayerNorm 和位置编码
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # (9) 将所有模块移动到指定设备并设置为训练模式
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        """GPT4TS 模型的前向传播过程
        
        实现完整的时间序列预测流程，包括数据标准化、patch 分割、
        GPT-2 编码和预测生成。
        
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
            4. GPT-2 编码
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

        # ==================== (4) 线性投影到 GPT-2 隐藏空间 ====================
        # 将每个 patch 映射到 d_model 维度
        # (B*M, patch_num, patch_size) -> (B*M, patch_num, d_model)
        outputs = self.in_layer(x)
        
        # ==================== (5) GPT-2 编码（如果启用）====================
        if self.is_gpt:
            # 将 patch 嵌入作为输入传递给 GPT-2
            # inputs_embeds: (B*M, patch_num, d_model)
            # last_hidden_state: (B*M, patch_num, d_model)
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

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
