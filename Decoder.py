import torch
import torch.nn as nn

from PositionwiseFeedForward import PositionwiseFeedForward
from MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Transformer的单个解码器层。
        
        Args:
            d_model (int): 输入的特征维度（即词嵌入的维度）。
            num_heads (int): 多头注意力机制的头数。
            d_ff (int): 前馈网络中间层的维度。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()
        
        # 1. 带掩码的多头自注意力层（用于处理目标序列）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 2. 多头交叉注意力层（用于处理编码器输出）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 3. 前馈网络
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 4. 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 5. Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 目标序列输入，shape: (batch_size, tgt_seq_len, d_model)
            enc_output (torch.Tensor): 编码器输出，shape: (batch_size, src_seq_len, d_model)
            tgt_mask (torch.Tensor, optional): 目标序列掩码，shape: (batch_size, tgt_seq_len, tgt_seq_len)
            src_mask (torch.Tensor, optional): 源序列掩码，shape: (batch_size, tgt_seq_len, src_seq_len)
        
        Returns:
            torch.Tensor: 解码器层输出，shape: (batch_size, tgt_seq_len, d_model)
        """
        # ----------------- 步骤1：带掩码的自注意力 -----------------
        # 输入x的shape: (batch_size, tgt_seq_len, d_model)
        attn_output = self.self_attn(x, x, x, tgt_mask)  # 自注意力计算
        attn_output = self.dropout1(attn_output)          # Dropout
        x = x + attn_output                               # 残差连接
        x = self.norm1(x)                                 # 层归一化，shape不变
        
        # ----------------- 步骤2：交叉注意力（处理编码器输出） -----------------
        # enc_output的shape: (batch_size, src_seq_len, d_model)
        # 交叉注意力：query来自解码器，key和value来自编码器
        cross_attn_output = self.cross_attn(
            x, enc_output, enc_output, src_mask           # query, key, value, mask
        )
        cross_attn_output = self.dropout2(cross_attn_output)  # Dropout
        x = x + cross_attn_output                         # 残差连接
        x = self.norm2(x)                                 # 层归一化，shape不变
        
        # ----------------- 步骤3：前馈网络 -----------------
        ff_output = self.ffn(x)                           # 前馈网络
        ff_output = self.dropout3(ff_output)              # Dropout
        x = x + ff_output                                 # 残差连接
        x = self.norm3(x)                                 # 层归一化，shape不变
        
        return x  # 输出shape: (batch_size, tgt_seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Transformer Decoder 模块
        Args:
            num_layers (int): 解码器层数。
            d_model (int): 输入的特征维度。
            num_heads (int): 多头注意力的头数。
            d_ff (int): 前馈网络中间层维度。
            dropout (float): Dropout概率，默认为0.1。
        """
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)  # 最终归一化层
    
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        Args:
            x (torch.Tensor): 目标序列输入，shape: (batch_size, tgt_seq_len, d_model)
            enc_output (torch.Tensor): 编码器输出，shape: (batch_size, src_seq_len, d_model)
            tgt_mask (torch.Tensor, optional): 目标序列掩码，shape: (batch_size, tgt_seq_len, tgt_seq_len)
            src_mask (torch.Tensor, optional): 源序列掩码，shape: (batch_size, tgt_seq_len, src_seq_len)
        
        Returns:
            torch.Tensor: 解码器输出，shape: (batch_size, tgt_seq_len, d_model)
        """
        # 逐层传递输入
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        x = self.norm(x)  # 最终归一化

        return x
