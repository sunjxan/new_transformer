import torch
from Transformer import Transformer

# 假设已经定义好Transformer类和相关组件

# 示例数据参数
batch_size = 10
src_seq_len = 120
tgt_seq_len = 100
pad_idx = 1  # 填充符的ID
src_vocab_size = 3000  # 根据实际情况修改
tgt_vocab_size = 5000  # 根据实际情况修改

# 创建模拟数据 (假设数据已经是LongTensor类型)
src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # (10, 120)
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # (10, 100)

# 初始化模型
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=7,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

# 初始化参数
model.init_parameters(init_type='xavier')

src_mask = Transformer.generate_padding_mask(src, pad_idx)
tgt_mask = Transformer.generate_padding_mask(tgt, pad_idx) & Transformer.generate_causal_mask(tgt_seq_len)

# 模型前向传播
output = model(
    src=src,                    # (10, 120)
    tgt=tgt,                    # (10, 100)
    src_mask=src_mask,          # (10, 1, 1, 120)
    tgt_mask=tgt_mask          # (10, 1, 100, 100)
)

print("输出形状:", output.shape)  # 应为 (10, 100, tgt_vocab_size)
