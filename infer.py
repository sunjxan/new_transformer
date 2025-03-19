import os
import torch

from data import create_chinese_tokenizer, create_english_tokenizer
from Transformer import Transformer

def process_data(model, text, tokenizer, device='cpu'):
    """处理输入数据并生成编码器输出"""
    tokens = tokenizer.tokenize(text)[:model.max_seq_len]
    src = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = model.generate_src_mask(src, tokenizer.pad_id())
    with torch.no_grad():
        memory = model.encode(src, src_mask)
    return memory, src_mask

def get_probs(model, memory, ys, src_mask, tokenizer):
    """获取下一个token的概率分布"""
    ys = ys[:, -model.max_seq_len:]
    tgt_mask = model.generate_tgt_mask(ys, tokenizer.pad_id())
    with torch.no_grad():
        decoder_output = model.decode(ys, memory, tgt_mask, src_mask)
        output = model.generator(decoder_output[:, -1])
    return torch.log_softmax(output, dim=-1)

def greedy_decode(model, text, src_tokenizer, tgt_tokenizer, max_len=50, device='cpu'):
    model.eval()
    
    memory, src_mask = process_data(model, text, src_tokenizer, device=device)
    
    start_token, end_token = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id()
    ys = torch.LongTensor([start_token]).unsqueeze(0).to(device)
    
    for _ in range(max_len - 1):
        probs = get_probs(model, memory, ys, src_mask, tgt_tokenizer)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    
    return ys[0].cpu().tolist()

def normalize_scores(seq, score, len_penalty=0.75):
    penalty = ((5 + len(seq)) / (5 + 1)) ** len_penalty  # Google的GNMT公式
    return score / penalty

def beam_search_decode(model, text, src_tokenizer, tgt_tokenizer,
                       max_len=50, beam_width=5, len_penalty=0.75, device='cpu'):  # len_penalty长度惩罚系数
    model.eval()
    
    memory, src_mask = process_data(model, text, src_tokenizer, device=device)
    
    # 初始化：序列、分数、完成状态
    start_token, end_token = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id()
    ys = torch.LongTensor([start_token]).unsqueeze(0).to(device)
    beam = [ ( ys, 0.0 ) ]  # (tokens, score)
    # 存储完整序列
    completed = []
    
    for _ in range(max_len - 1):
        candidates = []
        
        # 遍历当前所有候选
        for seq, score in beam:
            # 如果序列已结束，直接保留
            if seq[0, -1].item() == end_token:
                candidates.append( (seq, score) )
                continue
                
            # 获取下一个token的概率分布
            probs = get_probs(model, memory, seq, src_mask, tgt_tokenizer)
            top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, token], dim=-1)
                new_score = score + top_probs[0, i].item()
                candidates.append( (new_seq, new_score) )
        
        # 按分数排序，保留top beam_width个
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_width]
        
        # 分离已完成的序列
        new_beam = []
        for seq, score in candidates:
            if seq[0, -1].item() == end_token:
                completed.append( (seq[0].cpu().tolist(), normalize_scores(seq, score, len_penalty)) )  # 长度归一化
            else:
                new_beam.append( (seq, score) )
        beam = new_beam
        
        if not beam:
            break  # 所有候选均已完成
            
    # 合并未完成的序列
    completed += [ (seq[0].cpu().tolist(), normalize_scores(seq, score, len_penalty)) for seq, score in beam ]
    
    # 按归一化分数排序返回
    return sorted(completed, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    # 初始化模型和数据加载器
    chinese_tokenizer = create_chinese_tokenizer()
    english_tokenizer = create_english_tokenizer()
    
    # 创建模型
    model = Transformer(chinese_tokenizer.vocab_size(), english_tokenizer.vocab_size())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    while True:
        try:
            text = input('请输入中文句子: ').strip()
        except:
            print()
            exit()
        
        if text:
            break
    
    predictions = greedy_decode(model, text, chinese_tokenizer, english_tokenizer, device=device)
    print('\ngreedy decode:', english_tokenizer.detokenize(predictions))
    
    beam_search_result = beam_search_decode(model, text, chinese_tokenizer, english_tokenizer, device=device)
    print('\nbeam search decode:', english_tokenizer.detokenize(beam_search_result[0][0]))
