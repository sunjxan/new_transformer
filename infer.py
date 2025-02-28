import os
import torch

from data import create_vocabs, chinese_tokenizer, decode_sequence
from Transformer import Transformer

def process_data(model, sentence, tokenizer, vocab, max_len=128, device='cpu'):
    """处理输入数据并生成编码器输出"""
    tokens = tokenizer(sentence)
    src = [vocab.get(t, vocab['<unk>']) for t in tokens[:max_len]]
    src = torch.LongTensor(src).unsqueeze(0).to(device)
    src_mask = model.generate_src_mask(src, vocab['<pad>']).to(device)
    memory = model.encode(src, src_mask)
    return memory, src_mask

def get_probs(model, memory, ys, src_mask, tgt_vocab, device='cpu'):
    """获取下一个token的概率分布"""
    tgt_mask = model.generate_tgt_mask(ys, tgt_vocab['<pad>']).to(device)
    decoder_output = model.decode(ys, memory, tgt_mask, src_mask)
    output = model.generator(decoder_output[:, -1])
    return torch.log_softmax(output, dim=-1)

def greedy_decode(model, sentence, tokenizer, src_vocab, tgt_vocab, max_len=50, device='cpu'):
    model.eval()
    
    memory, src_mask = process_data(model, sentence, tokenizer, src_vocab, device=device)
    start_token, end_token = tgt_vocab['<sos>'], tgt_vocab['<eos>']
    ys = torch.LongTensor([start_token]).unsqueeze(0).to(device)
    
    for _ in range(max_len - 1):
        probs = get_probs(model, memory, ys, src_mask, tgt_vocab, device=device)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=-1)
        if next_token.item() == end_token:
            break

    return ys[0].cpu().tolist()

def normalize_scores(seq, score, len_penalty=0.75):
    penalty = ((5 + len(seq)) / (5 + 1)) ** len_penalty  # Google的GNMT公式
    return score / penalty

def beam_search_decode(model, sentence, tokenizer, src_vocab, tgt_vocab,
                       max_len=50, beam_width=5, len_penalty=0.75, device='cpu'):  # len_penalty长度惩罚系数
    model.eval()
    
    memory, src_mask = process_data(model, sentence, tokenizer, src_vocab, device=device)
    start_token, end_token = tgt_vocab['<sos>'], tgt_vocab['<eos>']

    # 初始化：序列、分数、完成状态
    beam = [ ( [start_token], 0.0 ) ]  # (tokens, score)
    completed = []
    
    for _ in range(max_len - 1):
        candidates = []
        
        # 遍历当前所有候选
        for seq, score in beam:
            # 如果序列已结束，直接保留
            if seq[-1] == end_token:
                candidates.append( (seq, score) )
                continue
                
            # 获取下一个token的概率分布
            ys = torch.LongTensor(seq).unsqueeze(0)
            probs = get_probs(model, memory, ys, src_mask, tgt_vocab, device=device)
            probs = probs.squeeze()
            ordered = torch.argsort(probs, dim=-1, descending=True)
            # 取top k个候选
            top_k = ordered[:beam_width].tolist()

            for token in top_k:
                new_seq = seq + [token]
                # log
                new_score = score + probs[token].item()
                candidates.append( (new_seq, new_score) )
        
        # 按分数排序，保留top beam_width个
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_width]
        
        # 分离已完成的序列
        new_beam = []
        for seq, score in candidates:
            if seq[-1] == end_token:
                completed.append( (seq, normalize_scores(seq, score, len_penalty)) )  # 长度归一化
            else:
                new_beam.append( (seq, score) )
        beam = new_beam
        
        if not beam:
            break  # 所有候选均已完成
            
    # 合并未完成的序列
    completed += [ (seq, normalize_scores(seq, score, len_penalty)) for seq, score in beam ]
    
    # 按归一化分数排序返回
    return sorted(completed, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    src_vocab, tgt_vocab = create_vocabs()

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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    sentence = input('请输入中文句子：\n')
    print('input:', sentence)

    predictions = greedy_decode(model, sentence, chinese_tokenizer, src_vocab, tgt_vocab, device=device)
    print('greedy decode:', decode_sequence(predictions, tgt_vocab))

    beam_search_result = beam_search_decode(model, sentence, chinese_tokenizer, src_vocab, tgt_vocab, device=device)
    print('beam search decode:', decode_sequence(beam_search_result[0][0], tgt_vocab))
