import torch
from collections import defaultdict

import jieba

# 原始数据
sentences = [
    ['我爱学习人工智能', 'I love studying AI'],
    ['深度学习改变世界', ' Deep learning changed the world'],
    ['自然语言处理很强大', 'NLP is powerful'],
    ['神经网络非常复杂', 'Neural-networks are complex']
]

# 定义特殊标记
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

# 中英文处理函数
def chinese_tokenizer(text):
    """中文按字符分割"""
    return list(jieba.cut(text))

def english_tokenizer(text):
    """英文按空格分割并转小写"""
    return [token.strip().lower() for token in text.strip().split() if token]

# 构建词汇表
def build_vocab(sentences, tokenizer):
    """构建词汇表"""
    vocab = defaultdict(int)
    for sent in sentences:
        for token in tokenizer(sent):
            vocab[token] += 1
    return vocab

# 生成处理后的数据
def process_data(sentences, chinese_seq_len, english_seq_len):
    # 分离中英文
    chinese_sents = [pair[0] for pair in sentences]
    english_sents = [pair[1] for pair in sentences]

    # 构建词汇表
    zh_vocab = {token: i+len(SPECIAL_TOKENS) 
                for i, token in enumerate(sorted(build_vocab(chinese_sents, chinese_tokenizer)))}
    en_vocab = {token: i+len(SPECIAL_TOKENS) 
                for i, token in enumerate(sorted(build_vocab(english_sents, english_tokenizer)))}

    # 添加特殊标记
    for i, token in enumerate(SPECIAL_TOKENS):
        zh_vocab[token] = i
        en_vocab[token] = i

    # 处理中文数据
    zh_data = []
    for sent in chinese_sents:
        tokens = chinese_tokenizer(sent)[:chinese_seq_len]
        tokens = tokens + ['<pad>']*(chinese_seq_len - len(tokens))
        zh_data.append([zh_vocab.get(t, zh_vocab['<unk>']) for t in tokens])

    # 处理英文数据
    en_data = []
    for sent in english_sents:
        tokens = english_tokenizer(sent)[:english_seq_len]
        tokens = tokens + ['<pad>']*(english_seq_len - len(tokens))
        en_data.append([en_vocab.get(t, en_vocab['<unk>']) for t in tokens])

    return (
        torch.LongTensor(zh_data),
        torch.LongTensor(en_data),
        zh_vocab,
        en_vocab
    )

# 执行数据处理
src_tensor, tgt_tensor, zh_vocab, en_vocab = process_data(sentences, 3, 5)

# 打印结果
print("中文词汇表（部分）：", dict(list(zh_vocab.items())[:10]))
print("\n英文词汇表（部分）：", dict(list(en_vocab.items())[:10]))
print("\n源语言张量（shape {}）：\n{}".format(src_tensor.shape, src_tensor))
print("\n目标语言张量（shape {}）：\n{}".format(tgt_tensor.shape, tgt_tensor))

# 生成掩码
def generate_mask(tensor, pad_idx=0):
    """生成padding掩码"""
    return (tensor != pad_idx).unsqueeze(1).unsqueeze(2)

src_mask = generate_mask(src_tensor)
tgt_mask = generate_mask(tgt_tensor)

print("\n源语言掩码（shape {}）：\n{}".format(src_mask.shape, src_mask.squeeze()))
print("\n目标语言掩码（shape {}）：\n{}".format(tgt_mask.shape, tgt_mask.squeeze()))