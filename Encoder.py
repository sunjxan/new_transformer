import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from SublayerConnection import SublayerConnection

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_first=True):
        """
        Transformer的单个编码器层。
        
        Args:
            d_model (int): 输入的特征维度（即词嵌入的维度）。
            num_heads (int): 多头注意力机制的头数。
            d_ff (int): 前馈网络中间层的维度。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        
        # 1. 多头自注意力层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 2. 前馈网络：两个线性层，中间用ReLU激活
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 3. 层归一化（LayerNorm） + Dropout层
        self.sublayer1 = SublayerConnection(d_model, dropout, norm_first)
        self.sublayer2 = SublayerConnection(d_model, dropout, norm_first)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入序列，shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): 掩码矩阵，shape: (batch_size, seq_len, seq_len)
        
        Returns:
            torch.Tensor: 编码器层的输出，shape: (batch_size, seq_len, d_model)
        """
        # ----------------- 步骤1：多头自注意力 -----------------
        # 输入x的shape: (batch_size, seq_len, d_model)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        
        # ----------------- 步骤2：前馈网络 -----------------
        x = self.sublayer2(x, self.ffn)
        
        return x  # 最终输出shape: (batch_size, seq_len, d_model)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, norm_first=True):
        """
        Transformer Encoder 模块
        Args:
            num_layers (int): Encoder 层数
            d_model (int): 输入/输出维度（模型总维度）
            num_heads (int): 多头注意力头数
            d_ff (int): 前馈网络中间层维度
            dropout (float): Dropout 概率
        """
        super().__init__()
        
        # 创建多个 Encoder 层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_first) 
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model) if norm_first else None  # 最终归一化层

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x (Tensor): 输入序列 (batch_size, seq_len, d_model)
            mask (Tensor): 掩码 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
        Returns:
            output (Tensor): 编码后的序列 (batch_size, seq_len, d_model)
        """
        # 逐层通过 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)  # 每层输出保持 (batch_size, seq_len, d_model)

        if self.norm:
            x = self.norm(x)  # 最终归一化

        return x
