import torch
import torch.nn as nn

from PositionwiseFeedForward import PositionwiseFeedForward
from MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
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
        
        # 3. 层归一化（LayerNorm）
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈后的归一化
        
        # 4. Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
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
        attn_output = self.self_attn(x, x, x, mask)  # 自注意力计算，输出shape同输入
        attn_output = self.dropout1(attn_output)      # 应用Dropout
        x = x + attn_output                           # 残差连接，shape不变
        x = self.norm1(x)                             # 层归一化，shape不变
        
        # ----------------- 步骤2：前馈网络 -----------------
        ff_output = self.ffn(x)          # 前馈网络，输出shape: (batch_size, seq_len, d_model)
        ff_output = self.dropout2(ff_output)  # 应用Dropout
        x = x + ff_output                # 残差连接，shape不变
        x = self.norm2(x)                # 层归一化，shape不变
        
        return x  # 最终输出shape: (batch_size, seq_len, d_model)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
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
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x (Tensor): 输入序列 (batch_size, seq_len, d_model)
            mask (Tensor): 掩码 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
        Returns:
            output (Tensor): 编码后的序列 (batch_size, seq_len, d_model)
        """
        # 1. 逐层通过 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)  # 每层输出保持 (batch_size, seq_len, d_model)

        # 2. 最终 LayerNorm（可选，部分实现在每层内部已包含）
        x = self.norm(x)

        return x
