import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """
        缩放点积注意力机制
        Args:
            dropout (float): Dropout概率，默认为0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Args:
            Q: 查询张量, shape (batch_size, num_heads, seq_len_q, d_k)
            K: 键张量, shape (batch_size, num_heads, seq_len_k, d_k)
            V: 值张量, shape (batch_size, num_heads, seq_len_k, d_v)
            mask: 掩码张量, shape (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            注意力输出: shape (batch_size, num_heads, seq_len_q, d_v)
            注意力权重: shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 计算Q和K的点积得分
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Q·K^T
        # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)

        # 缩放操作：除以sqrt(d_k)防止梯度消失
        d_k = K.size(-1)  # 获取K的最后一个维度d_k
        scores = scores / math.sqrt(d_k)
        # scores shape保持不变: (batch_size, num_heads, seq_len_q, seq_len_k)

        # 应用掩码（如果需要）
        if mask is not None:
            # 将mask中为True/1的位置替换为极小的值（softmax后趋近于0）
            scores = scores.masked_fill(mask == 0, -1e9)  
            # mask需要能广播到scores的形状

        # 计算注意力权重（最后一维进行softmax）
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)

        attn_weights = self.dropout(attn_weights)

        # 将注意力权重应用到V上
        output = torch.matmul(attn_weights, V)
        # output shape: (batch_size, num_heads, seq_len_q, d_v)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制
        Args:
            d_model: 输入维度（总维度）
            num_heads: 注意力头的数量
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换矩阵（无偏置）
        self.W_q = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # (d_model, d_model)
        
        self.attn = ScaledDotProductAttention(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Args:
            q: 查询向量 (batch_size, seq_len_q, d_model)
            k: 键向量 (batch_size, seq_len_kv, d_model)
            v: 值向量 (batch_size, seq_len_kv, d_model)
            mask: 掩码 (batch_size, seq_len_q, seq_len_kv)
        Returns:
            输出: (batch_size, seq_len_q, d_model)
            注意力权重: (batch_size, num_heads, seq_len_q, seq_len_kv)
        """
        batch_size = Q.size(0)
        
        # 线性变换 + 分割多头 + 转置维度
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_kv, d_k)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len_kv, d_k)
        
        # 应用掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配多头 (batch_size, 1, seq_len_q, seq_len_kv) -> 广播到num_heads
            mask = mask.unsqueeze(1)

        output, attn_weights = self.attn(Q, K, V, mask)
        
        # 转置回维度 (batch_size, seq_len_q, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()
        
        # 拼接所有头 (batch_size, seq_len_q, d_model)
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(output)
        
        return output
