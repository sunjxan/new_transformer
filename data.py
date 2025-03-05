import torch
import jieba

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 原始数据
sentences = [
    ['我爱学习人工智能', 'I love studying AI'],
    ['深度学习改变世界', ' Deep learning changed the world'],
    ['自然语言处理很强大', 'NLP is powerful'],
    ['神经网络非常复杂', 'Neural-networks are complex']
]

def filter_examples(example):
    zh_len = len(example[0])
    en_len = len(example[1])
    
    return (
        1 < zh_len < 150 and   # 控制中英文长度
        1 < en_len < 200 and
        not any(c in example[0] for c in ['�', '�'])  # 过滤非法字符
    )

# 数据清洗
sentences = list(filter(filter_examples, sentences))

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
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))
    vocab = {token: i+len(SPECIAL_TOKENS) for i, token in enumerate(sorted(counter))}
    # 添加特殊标记
    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i
    return vocab

def create_vocabs():
    # 分离中英文
    chinese_sents = [pair[0] for pair in sentences]
    english_sents = [pair[1] for pair in sentences]
    
    # 生成词汇表
    chinese_vocab = build_vocab(chinese_sents, chinese_tokenizer)
    english_vocab = build_vocab(english_sents, english_tokenizer)
    
    return chinese_vocab, english_vocab

# 预测结果解码
def decode_sequence(ids, vocab):
    idx2token = {v: k for k, v in vocab.items()}
    return ' '.join([idx2token.get(i, '<unk>')
                     for i in ids if i not in [vocab['<pad>'], vocab['<sos>'], vocab['<eos>']]])

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

def collate_batch(batch, chinese_vocab, english_vocab, max_chinese_len=128, max_english_len=128):
    chinese_data = []
    english_data = []
    
    for chinese_sent, english_sent in batch:
        # 处理中文数据（源语言不加特殊标记）
        tokens = chinese_tokenizer(chinese_sent)
        tokens = [chinese_vocab.get(t, chinese_vocab['<unk>']) for t in tokens[:max_chinese_len]]
        chinese_data.append(torch.LongTensor(tokens))
        # 处理英文数据（目标语言添加特殊标记）
        tokens = ['<sos>'] + english_tokenizer(english_sent) + ['<eos>'] 
        tokens = [english_vocab.get(t, english_vocab['<unk>']) for t in tokens[:max_english_len]]
        english_data.append(torch.LongTensor(tokens))
    
    chinese_data = pad_sequence(chinese_data, batch_first=True, padding_value=chinese_vocab['<pad>'])
    english_data = pad_sequence(english_data, batch_first=True, padding_value=english_vocab['<pad>'])
    return chinese_data, english_data

def create_dataloader(chinese_vocab, english_vocab, batch_size, shuffle=False, drop_last=False, max_chinese_len=128, max_english_len=128):
    dataset = TranslationDataset(sentences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, chinese_vocab, english_vocab, max_chinese_len, max_english_len))
