import math
import torch
import jieba

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

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
    return list(jieba.cut(text.strip()))

def english_tokenizer(text):
    """英文按空格分割并转小写"""
    return [token.strip().lower() for token in text.strip().split() if token]

# 构建词汇表
def build_vocab(sentences, tokenizer):
    """构建词汇表"""
    counter = defaultdict(int)
    for sent in sentences:
        for token in tokenizer(sent):
            counter[token] += 1
    vocab = {token: i+len(SPECIAL_TOKENS) for i, token in enumerate(sorted(counter))}
    # 添加特殊标记
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i
    return vocab

class TranslationDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        if index >= 0 and index < self.__len__():
            return self.sentences[index]
        return

def collate_batch(batch, chinese_vocab, chinese_seq_len, english_vocab, english_seq_len):
    chinese_data = []
    english_data = []

    for chinese_sent, english_sent in batch:
        # 处理中文数据（源语言不加特殊标记）
        tokens = chinese_tokenizer(chinese_sent)
        tokens = tokens[:chinese_seq_len]
        tokens += ['<pad>'] * (chinese_seq_len - len(tokens))
        chinese_data.append([chinese_vocab.get(t, chinese_vocab['<unk>']) for t in tokens])
        # 处理英文数据（目标语言添加特殊标记）
        tokens = ['<sos>'] + english_tokenizer(english_sent) + ['<eos>'] 
        tokens = tokens[:english_seq_len]
        tokens += ['<pad>'] * (english_seq_len - len(tokens))
        english_data.append([english_vocab.get(t, english_vocab['<unk>']) for t in tokens])

    return torch.LongTensor(chinese_data), torch.LongTensor(english_data)

def create_dataloader(chinese_seq_len, english_seq_len, batch_size, shuffle=False, drop_last=False):
    # 分离中英文
    chinese_sents = [pair[0] for pair in sentences]
    english_sents = [pair[1] for pair in sentences]

    # 生成词汇表
    chinese_vocab = build_vocab(chinese_sents, chinese_tokenizer)
    english_vocab = build_vocab(english_sents, english_tokenizer)

    dataset = TranslationDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, chinese_vocab, chinese_seq_len, english_vocab, english_seq_len))
    if drop_last:
        loader_size = math.floor(len(dataset) / batch_size)
    else:
        loader_size = math.ceil(len(dataset) / batch_size)
    return loader, loader_size, chinese_vocab, english_vocab
