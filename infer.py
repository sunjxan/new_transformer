import os
import torch

from data import create_vocabs, chinese_tokenizer, decode_sequence
from Transformer import Transformer

def greedy_decode(model, sentence, src_vocab, tgt_vocab, max_len=20):
    model.eval()
    
    tokens = chinese_tokenizer(sentence)
    src = [src_vocab.get(t, src_vocab['<unk>']) for t in tokens]
    src = torch.LongTensor(src).unsqueeze(0)
    src_mask = model.generate_src_mask(src, src_vocab['<pad>'])
    memory = model.encode(src)

    ys = torch.LongTensor([tgt_vocab['<sos>']]).unsqueeze(0)
    for _ in range(max_len - 1):
        tgt_mask = model.generate_tgt_mask(ys, tgt_vocab['<pad>'])
        decoder_output = model.decode(memory, ys, src_mask, tgt_mask)
        output = model.generator(decoder_output[:, -1])
        probs = torch.softmax(output, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        ys = torch.cat([ys, next_token], dim=-1)
        if next_token == tgt_vocab['<eos>']:
            break

    return ys[0].tolist()

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
    predictions = greedy_decode(model, sentence, src_vocab, tgt_vocab)
    print('input:', sentence)
    print('output:', decode_sequence(predictions, tgt_vocab))
