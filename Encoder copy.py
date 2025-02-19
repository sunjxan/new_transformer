import torch
import torch.nn as nn

from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1, max_seq_len=512):
        """
        Transformer Encoder 模块
        Args:
            num_layers (int): Encoder 层数
            d_model (int): 输入/输出维度（模型总维度）
            num_heads (int): 多头注意力头数
            d_ff (int): 前馈网络中间层维度
            dropout (float): Dropout 概率
            max_seq_len (int): 最大序列长度（用于位置编码）
        """
        super().__init__()
        self.d_model = d_model
        
        # 位置编码（可学习的参数）
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # (1, max_seq_len, d_model)
        nn.init.normal_(self.position_embedding, mean=0, std=0.02)  # 初始化
        
        # 创建多个 Encoder 层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x (Tensor): 输入序列 (batch_size, seq_len, d_model)
            mask (Tensor): 掩码 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
        Returns:
            output (Tensor): 编码后的序列 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 添加位置编码
        # 截取位置编码的前 seq_len 个位置，并扩展匹配 batch_size
        pos_emb = self.position_embedding[:, :seq_len, :]  # (1, seq_len, d_model)
        pos_emb = pos_emb.expand(batch_size, -1, -1)      # (batch_size, seq_len, d_model)
        x = x + pos_emb  # (batch_size, seq_len, d_model)
        
        # 2. 应用 Dropout
        x = self.dropout(x)
        
        # 3. 逐层通过 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)  # 每层输出保持 (batch_size, seq_len, d_model)
        
        # 4. 最终 LayerNorm（可选，部分实现在每层内部已包含）
        output = self.layer_norm(x)
        
        return output