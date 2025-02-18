import torch
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力机制实现
    Args:
        query: 查询张量, shape (batch_size, seq_len_q, d_k)
        key: 键张量, shape (batch_size, seq_len_k, d_k)
        value: 值张量, shape (batch_size, seq_len_k, d_v)
        mask: 掩码张量, shape (batch_size, seq_len_q, seq_len_k) 或可广播形状

    Returns:
        注意力输出: shape (batch_size, seq_len_q, d_v)
        注意力权重: shape (batch_size, seq_len_q, seq_len_k)
    """
    # 计算query和key的点积得分
    scores = torch.matmul(query, key.transpose(-2, -1))  # Q·K^T
    # scores shape: (batch_size, seq_len_q, seq_len_k)

    # 缩放操作：除以sqrt(d_k)防止梯度消失
    d_k = query.size(-1)  # 获取query的最后一个维度d_k
    scores = scores / math.sqrt(d_k)
    # scores shape保持不变: (batch_size, seq_len_q, seq_len_k)

    # 应用掩码（如果需要）
    if mask is not None:
        # 将mask中为True/1的位置替换为极小的值（softmax后趋近于0）
        scores = scores.masked_fill(mask == 0, -1e9)  
        # mask需要能广播到scores的形状

    # 计算注意力权重（最后一维进行softmax）
    p_attn = torch.softmax(scores, dim=-1)
    # p_attn shape: (batch_size, seq_len_q, seq_len_k)

    # 将注意力权重应用到value上
    output = torch.matmul(p_attn, value)
    # output shape: (batch_size, seq_len_q, d_v)

    return output, p_attn