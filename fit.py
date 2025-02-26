import torch
import torch.nn as nn
import torch.optim as optim

from data import get_vocabs, create_dataloader, chinese_tokenizer
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

for _ in range(20):
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

# 预测结果解码
def decode_sequence(ids, vocab):
    idx2token = {v: k for k, v in vocab.items()}
    return ' '.join([idx2token.get(i, '<unk>')
                     for i in ids if i not in [vocab['<pad>'], vocab['<sos>'], vocab['<eos>']]])

def greedy_decode(model, src_sent, src_vocab, tgt_vocab, max_len=20):
    model.eval()

    tokens = chinese_tokenizer(src_sent)
    src = [src_vocab.get(t, src_vocab['<unk>']) for t in tokens]
    src = torch.LongTensor(src).unsqueeze(0)
    src_mask = Transformer.generate_src_mask(src, src_vocab['<pad>'])
    ys = torch.LongTensor([tgt_vocab["<sos>"]]).unsqueeze(0)

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

src_sent = '神经网络很强大'
predictions = greedy_decode(model, src_sent, src_vocab, tgt_vocab)
print(decode_sequence(predictions, tgt_vocab))


import math
from collections import deque
from typing import List, Tuple

def beam_search_decode(
    model,  # 具备predict方法的模型
    start_token: int,
    end_token: int,
    beam_width: int = 3,
    max_length: int = 20,
    alpha: float = 0.75  # 长度惩罚系数
) -> List[Tuple[List[int], float]]:
    """
    束搜索实现
    Args:
        model: 生成模型，需实现predict(input_seq)方法，返回logits
        start_token: 起始标记ID
        end_token: 结束标记ID
        beam_width: 束宽
        max_length: 最大生成长度
        alpha: 长度归一化系数（惩罚短句）
    Returns:
        List[ (sequence, normalized_score), ... ]
    """
    # 初始化：序列、分数、完成状态
    beam = [ ( [start_token], 0 ) ]  # (tokens, log_score)
    completed = []
    
    for step in range(max_length):
        candidates = []
        
        # 遍历当前所有候选
        for seq, score in beam:
            # 如果序列已结束，直接保留
            if seq[-1] == end_token:
                candidates.append( (seq, score) )
                continue
                
            # 获取下一个token的概率分布（对数概率）
            logits = model.predict(seq)
            log_probs = logits.log_softmax(dim=-1).cpu().numpy()
            
            # 取top k个候选
            top_k = log_probs.argsort()[-beam_width:][::-1]
            for token in top_k:
                new_seq = seq + [token]
                new_score = score + log_probs[token]
                candidates.append( (new_seq, new_score) )
        
        # 按分数排序，保留top beam_width个
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        beam = ordered[:beam_width]
        
        # 分离已完成的序列
        new_beam = []
        for seq, score in beam:
            if seq[-1] == end_token:
                completed.append( (seq, score/(len(seq)**alpha) )  # 长度归一化
            else:
                new_beam.append( (seq, score) )
        beam = new_beam
        
        if not beam:
            break  # 所有候选均已完成
            
    # 合并未完成的序列
    completed += [ (seq, score/(len(seq)**alpha)) for seq, score in beam ]
    
    # 按归一化分数排序返回
    return sorted(completed, key=lambda x: x[1], reverse=True)

class DummyModel:
    """模拟一个总是预测固定概率的模型"""
    def predict(self, seq):
        # 假设词汇表大小=5
        import torch
        if len(seq) < 3:
            return torch.log_softmax(torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25]), -1)
        else:
            return torch.log_softmax(torch.tensor([0.4, 0.1, 0.1, 0.3, 0.1]), -1)

model = DummyModel()
results = beam_search_decode(
    model, 
    start_token=0, 
    end_token=4,
    beam_width=2,
    max_length=5
)

# 输出结果示例
for seq, score in results:
    print(f"Sequence: {seq}, Score: {math.exp(score):.4f}")
