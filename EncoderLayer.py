import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Transformer的前馈神经网络（FFN）模块。
        
        Args:
            d_model (int): 输入和输出的特征维度（即词嵌入的维度）。
            d_ff (int): 中间层的维度（通常比d_model大，如2048）。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        # 第一个线性层：将输入从d_model扩展到d_ff维度
        self.linear1 = nn.Linear(d_model, d_ff)
        # 第二个线性层：将中间层从d_ff恢复回d_model维度
        self.linear2 = nn.Linear(d_ff, d_model)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 输入序列，shape: (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 输出序列，shape: (batch_size, seq_len, d_model)
        """
        # 输入x的shape: (batch_size, seq_len, d_model)
        
        # 第一步：通过第一个线性层（扩展维度）
        x = self.linear1(x)
        # 此时x的shape: (batch_size, seq_len, d_ff)
        
        # 第二步：应用ReLU激活函数
        x = self.relu(x)
        # ReLU不改变shape，仍为(batch_size, seq_len, d_ff)
        
        # 第三步：应用Dropout
        x = self.dropout(x)
        # Dropout不改变shape，仍为(batch_size, seq_len, d_ff)
        
        # 第四步：通过第二个线性层（恢复维度）
        x = self.linear2(x)
        # 此时x的shape恢复为: (batch_size, seq_len, d_model)
        
        return x

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
        前向传播函数。
        
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