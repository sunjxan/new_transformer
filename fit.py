import torch
import torch.nn as nn

from data import create_dataloader
from Transformer import Transformer

train_loader, train_loader_size, src_vocab, tgt_vocab = create_dataloader(
    chinese_seq_len=8,
    english_seq_len=10,
    batch_size=3
)

# 初始化模型
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

# 初始化参数
model.init_parameters(init_type='xavier')

for src, tgt in train_loader:
    src_mask = Transformer.generate_src_mask(src)
    tgt_mask = Transformer.generate_tgt_mask(tgt[:, :-1])
    output = model(
        src=src,
        tgt=tgt[:, :-1],  # 解码器输入去尾
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding位置的损失
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    weight_decay=0.01
)

# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
