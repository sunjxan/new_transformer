import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        位置编码（Positional Encoding）模块。
        
        Args:
            d_model (int): 输入的特征维度（即词嵌入的维度）。
            max_len (int): 支持的最大序列长度（默认为5000）。
            dropout (float): Dropout概率（默认为0.1）。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # Dropout层

        # 初始化位置编码矩阵，shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引，shape: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 计算除数项，用于生成正弦和余弦的波长
        # div_term shape: (d_model // 2, )
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        # 填充位置编码矩阵的偶数列（正弦函数）
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2表示从0开始每隔一列取一列
        
        # 填充位置编码矩阵的奇数列（余弦函数）
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2表示从1开始每隔一列取一列
        
        # 扩展维度，使pe的shape变为 (1, max_len, d_model)，便于后续与输入相加
        pe = pe.unsqueeze(0)
        
        # 将位置编码矩阵注册为缓冲区（不参与梯度更新）
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 输入序列，shape: (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 添加位置编码后的序列，shape: (batch_size, seq_len, d_model)
        """
        # 从预计算的位置编码矩阵中截取与输入序列长度匹配的部分
        # self.pe[:, :x.size(1)] 的shape: (1, seq_len, d_model)
        # 通过广播机制，与输入x的shape (batch_size, seq_len, d_model) 相加
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        
        # 应用Dropout
        x = self.dropout(x)
        return x
