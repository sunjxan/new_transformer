import torch
import torch.nn as nn

from PositionalEncoding import PositionalEncoding
from Encoder import Encoder
from Decoder import Decoder

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
        self.src_embed = nn.Embedding(src_vocab_size, d_model)  # (src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)  # (tgt_vocab_size, d_model)
        
        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. 编码器和解码器
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, max_seq_len)
        
        # 4. 最终线性层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)  # (d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        前向传播
        Args:
            src (Tensor): 源序列 (batch_size, src_seq_len)
            tgt (Tensor): 目标序列 (batch_size, tgt_seq_len)
            src_mask (Tensor): 源序列掩码 (batch_size, src_seq_len) 或 (batch_size, src_seq_len, src_seq_len)
            tgt_mask (Tensor): 目标序列掩码 (batch_size, tgt_seq_len) 或 (batch_size, tgt_seq_len, tgt_seq_len)
            memory_mask (Tensor): Encoder 到 Decoder 的掩码 (可选)
        Returns:
            output (Tensor): 输出概率分布 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. 词嵌入 + 位置编码
        src_emb = self.src_embed(src) * torch.sqrt(torch.tensor(self.d_model))  # (batch_size, src_seq_len, d_model)
        tgt_emb = self.tgt_embed(tgt) * torch.sqrt(torch.tensor(self.d_model))  # (batch_size, tgt_seq_len, d_model)
        src_emb = self.positional_encoding(src_emb)  # (batch_size, src_seq_len, d_model)
        tgt_emb = self.positional_encoding(tgt_emb)  # (batch_size, tgt_seq_len, d_model)
        
        # 2. 编码器处理
        memory = self.encoder(src_emb, src_mask)  # (batch_size, src_seq_len, d_model)
        
        # 3. 解码器处理
        decoder_output = self.decoder(
            tgt_emb, memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask  # 通常是 src_mask 的变体
        )  # (batch_size, tgt_seq_len, d_model)
        
        # 4. 输出层映射到词表
        output = self.output_layer(decoder_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        
        return output
