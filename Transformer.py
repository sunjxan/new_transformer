import math
import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # 词嵌入后缩放
        # 数值稳定性：在后续的注意力机制中，点积操作 Q·K^T 的结果会除以 sqrt(d_k)（其中 d_k = d_model）。在嵌入阶段提前乘以 sqrt(d_model)，可以保持数值量级的一致性。
        # 梯度控制：防止词嵌入的初始值过小，导致梯度消失。
        return self.embedding(x) * math.sqrt(self.d_model)

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
        前向传播
        
        Args:
            x (torch.Tensor): 输入序列，shape: (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 添加位置编码后的序列，shape: (batch_size, seq_len, d_model)
        """
        # 从预计算的位置编码矩阵中截取与输入序列长度匹配的部分
        # self.pe[:, :x.size(1)] 的shape: (1, seq_len, d_model)
        # 通过广播机制，与输入x的shape (batch_size, seq_len, d_model) 相加
        x = x + self.pe[:, :x.size(1)]
        
        # 应用Dropout
        x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        Transformer的最终输出生成器，将Decoder的输出映射到目标词汇表空间。
        
        Args:
            d_model (int): Decoder输出的特征维度。
            vocab_size (int): 目标词汇表的大小。
        """
        super().__init__()
        # 线性投影层：将Decoder输出的d_model维度映射到词汇表维度
        # 输入shape: (batch_size, seq_len, d_model)
        # 输出shape: (batch_size, seq_len, vocab_size)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_output):
        """
        前向传播
        
        Args:
            decoder_output (torch.Tensor): 
                Decoder的输出，shape: (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: 词汇表logits（未归一化的概率分数），shape: (batch_size, seq_len, vocab_size)
        """
        # decoder_output shape: (batch_size, seq_len, d_model)
        logits = self.proj(decoder_output)  # shape: (batch_size, seq_len, vocab_size)
        
        return logits  # 直接返回logits（更高效，避免重复计算Softmax）

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, max_seq_len=512, dropout=0.1):
        """
        Transformer 模型
        Args:
            src_vocab_size (int): 源语言词表大小
            tgt_vocab_size (int): 目标语言词表大小
            d_model (int): 模型维度（输入/输出维度）
            num_heads (int): 多头注意力头数
            num_encoder_layers (int): Encoder 层数
            num_decoder_layers (int): Decoder 层数
            d_ff (int): 前馈网络中间层维度
            max_seq_len (int): 最大序列长度
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        
        # 1. 词嵌入层
        self.src_embed = Embeddings(src_vocab_size, d_model)  # (src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)  # (tgt_vocab_size, d_model)
        
        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. 编码器和解码器
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # 4. 最终线性层
        self.generator = Generator(d_model, tgt_vocab_size)  # (d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        """
        编码
        Args:
            src (Tensor): 源序列 (batch_size, src_seq_len)
            src_mask (Tensor): 源序列掩码 (batch_size, src_seq_len, src_seq_len)
        Returns:
            memory (Tensor): 编码结果 (batch_size, src_seq_len, d_model)
        """
        # 1. 词嵌入
        src_emb = self.src_embed(src)  # (batch_size, src_seq_len, d_model)

        # 2. 位置编码
        src_emb = self.positional_encoding(src_emb)  # (batch_size, src_seq_len, d_model)
        
        # 3. 编码器处理
        memory = self.encoder(src_emb, src_mask)  # (batch_size, src_seq_len, d_model)
        
        return memory

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        """
        解码
        Args:
            memory (Tensor): 编码结果 (batch_size, src_seq_len, d_model)
            tgt (Tensor): 目标序列 (batch_size, tgt_seq_len)
            src_mask (Tensor): 源序列掩码 (batch_size, src_seq_len, src_seq_len)
            tgt_mask (Tensor): 目标序列掩码 (batch_size, tgt_seq_len, tgt_seq_len)
        Returns:
            decoder_output (Tensor): 解码结果 (batch_size, tgt_seq_len, d_model)
        """
        # 1. 词嵌入
        tgt_emb = self.tgt_embed(tgt)  # (batch_size, tgt_seq_len, d_model)

        # 2. 位置编码
        tgt_emb = self.positional_encoding(tgt_emb)  # (batch_size, tgt_seq_len, d_model)

        # 3. 解码器处理
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)  # (batch_size, tgt_seq_len, d_model)

        return decoder_output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        Args:
            src (Tensor): 源序列 (batch_size, src_seq_len)
            tgt (Tensor): 目标序列 (batch_size, tgt_seq_len)
            src_mask (Tensor): 源序列掩码 (batch_size, 1, src_seq_len)
            tgt_mask (Tensor): 目标序列掩码 (batch_size, tgt_seq_len, tgt_seq_len)
        Returns:
            output (Tensor): 输出概率分布 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. 编码
        memory = self.encode(src, src_mask)  # (batch_size, src_seq_len, d_model)

        # 2. 解码
        decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)  # (batch_size, tgt_seq_len, d_model)
        
        # 3. 输出层映射到词表
        output = self.generator(decoder_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        
        return output
    
    def init_parameters(self, init_type='xavier'):
        """
        初始化模型参数
        Args:
            init_type (str): 初始化类型，可选 'xavier'（默认）或 'kaiming'
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # 仅初始化矩阵权重，忽略偏置和LayerNorm参数
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                else:
                    raise ValueError(f"不支持的初始化类型: {init_type}")
            elif 'bias' in name:  # 偏置初始化为零
                nn.init.zeros_(param)
            # LayerNorm参数保持默认初始化（gamma=1, beta=0）

    @staticmethod
    def generate_src_mask(seq, pad_idx=0):
        """生成填充掩码（pad位置为False）"""
        return (seq != pad_idx).unsqueeze(-2)  # (batch_size, 1, seq_len)

    @staticmethod
    def generate_causal_mask(seq_len):
        """生成因果掩码（下三角为True）"""
        return torch.tril(torch.ones(seq_len, seq_len)) == 1  # (seq_len, seq_len)

    @staticmethod
    def generate_tgt_mask(seq, pad_idx=0):
        '''结合填充掩码和因果掩码得到目标序列掩码'''
        return Transformer.generate_src_mask(seq, pad_idx) & Transformer.generate_causal_mask(seq.size(-1)).to(seq.device)   # (batch_size, seq_len, seq_len)

'''
    计算模型参数量

    1. 嵌入层
    源语言嵌入：src_vocab_size × d_model
    目标语言嵌入：tgt_vocab_size × d_model

    2. 编码器（Encoder）
    每层包含：
        1个多头注意力（4个线性层，无偏置项）：4 × (d_model × d_model)
        前馈网络（2个线性层）：2 × d_model × d_ff + d_ff + d_model
        2个归一化层：2 × (d_model + d_model)
    最终归一化层（norm_first=True时）：d_model + d_model
    总参数量：
        num_encoder_layers × [4d² + 2d·d_ff + d_ff + 5d] + 2d

    3. 解码器（Decoder）
    每层包含：
        2个多头注意力（4个线性层，无偏置项）：2 × 4 × (d_model × d_model)
        前馈网络（2个线性层）：2 × d_model × d_ff + d_ff + d_model
        3个归一化层：3 × (d_model + d_model)
    最终归一化层（norm_first=True时）：d_model + d_model
    总参数量：
        num_decoder_layers × [8d² + 2d·d_ff + d_ff + 7d] + 2d

    4. 生成器（Generator）
    线性层：d_model × tgt_vocab_size + tgt_vocab_size
'''
def count_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
