import torch
import torch.nn as nn
import torch.optim as optim

from data import get_vocabs, create_dataloader, chinese_tokenizer, decode_sequence
from Transformer import Transformer

src_vocab, tgt_vocab = get_vocabs()

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

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])  # 忽略padding位置的损失
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01
)

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

train_loader, train_loader_size = create_dataloader(src_vocab, tgt_vocab, 8, 10, 3,
    shuffle=True, drop_last=True)

model.train()

for _ in range(10):
    for src, tgt in train_loader:
        # 梯度清零
        optimizer.zero_grad()

        # 生成掩码
        src_mask = Transformer.generate_src_mask(src, src_vocab['<pad>'])
        tgt_mask = Transformer.generate_tgt_mask(tgt, tgt_vocab['<pad>'])

        # 前向传播
        output = model(
            src=src,
            tgt=tgt[:, :-1],  # 解码器输入去尾
            src_mask=src_mask,
            tgt_mask=tgt_mask[:, :-1, :-1]
        )
        
        # 计算损失
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1)  # 目标去头
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()

    scheduler.step()

val_loader, val_loader_size = create_dataloader(src_vocab, tgt_vocab, 8, 10, 3)

model.eval()

with torch.no_grad():
    for src, tgt in val_loader:
        # 生成掩码
        src_mask = Transformer.generate_src_mask(src, src_vocab['<pad>'])
        tgt_mask = Transformer.generate_tgt_mask(tgt, tgt_vocab['<pad>'])

        # 前向传播
        output = model(
            src=src,
            tgt=tgt[:, :-1],  # 解码器输入去尾
            src_mask=src_mask,
            tgt_mask=tgt_mask[:, :-1, :-1]
        )
        
        # 计算损失
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1)  # 目标去头
        )

def greedy_decode(model, src_sent, src_vocab, tgt_vocab, max_len=20):
    model.eval()
    
    tokens = chinese_tokenizer(src_sent)
    src = [src_vocab.get(t, src_vocab['<unk>']) for t in tokens]
    src = torch.LongTensor(src).unsqueeze(0)
    src_mask = Transformer.generate_src_mask(src, src_vocab['<pad>'])
    ys = torch.LongTensor([tgt_vocab['<sos>']]).unsqueeze(0)

    with torch.no_grad():
        memory = model.encode(src)
        for _ in range(max_len - 1):
            tgt_mask = Transformer.generate_src_mask(ys)
            decoder_output = model.decode(memory, ys, src_mask, tgt_mask)
            output = model.generator(decoder_output[:, -1])
            probs = torch.softmax(output, dim=-1)
            next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
            ys = torch.cat([ys, next_token], dim=-1)
            if next_token == tgt_vocab['<eos>']:
                break

    return ys[0].tolist()

src_sent = '我爱学习人工智能'
predictions = greedy_decode(model, src_sent, src_vocab, tgt_vocab)
print(decode_sequence(predictions, tgt_vocab))
