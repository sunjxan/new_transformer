import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制
        Args:
            d_model: 输入维度（总维度）
            num_heads: 注意力头的数量
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换矩阵（无偏置）
        self.W_q = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # (d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
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
        batch_size = q.size(0)
        
        # 线性变换 + 分割多头
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len_q, num_heads, d_k)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len_kv, num_heads, d_k)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len_kv, num_heads, d_k)
        
        # 转置维度以便矩阵计算 (batch_size, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数 (batch_size, num_heads, seq_len_q, seq_len_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 应用掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配多头 (batch_size, 1, seq_len_q, seq_len_kv) -> 广播到num_heads
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到value (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, v)
        
        # 转置回维度 (batch_size, seq_len_q, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()
        
        # 拼接所有头 (batch_size, seq_len_q, d_model)
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(output)
        
        return output, attn_weights