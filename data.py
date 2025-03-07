import re
import jieba
import torch

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 原始数据
examples = [
    {'zh': '我爱学习人工智能', 'en': 'I love studying AI'},
    {'zh': '深度学习改变世界', 'en': ' Deep learning changed the world'},
    {'zh': '自然语言处理很强大', 'en': 'NLP is powerful'},
    {'zh': '神经网络非常复杂', 'en': 'Neural-networks are complex'}
]

def clean_text(example):
    pair = example.copy()
    for k, v in pair.items():
        # 过滤非法字符
        v = re.sub(r'[�◆★【】▲▼■●]', '', v)
        # 合并连续空格
        v = re.sub(r'\s+', ' ', v)
        pair[k] = v.strip()
    
    if 1 < len(pair['zh']) < 150 and 1 < len(pair['en']) < 200:
        return pair

# 数据清洗
texts = []
for example in examples:
    cleaned = clean_text(example)
    if cleaned:
        texts.append(cleaned)

class SimpleTokenizer:
    @staticmethod
    def pad_id():
        return 0
    
    @staticmethod
    def unk_id():
        return 1
    
    @staticmethod
    def bos_id():
        return 2
    
    @ staticmethod
    def eos_id():
        return 3
    
    @staticmethod
    def encode_as_pieces(text):
        return text.split()
    
    def __init__(self, texts):
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        # 构建词汇表
        counter = Counter()
        for text in texts:
            counter.update(self.__class__.encode_as_pieces(text))
        self.token2id = {}
        self.id2token = {}
        # 添加特殊标记
        for i, token in enumerate(special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token
        for i, token in enumerate(sorted(counter)):
            j = i + len(special_tokens)
            self.token2id[token] = j
            self.id2token[j] = token
    
    def vocab_size(self):
        return len(self.token2id)
    
    def __len__(self):
        return self.vocab_size()
    
    def piece_to_id(self, piece):
        return self.token2id.get(piece, self.__class__.unk_id())
    
    def id_to_piece(self, idx):
        if idx not in self.id2token:
            raise Exception("IndexError: Out of range: piece id is out of range.")
        return self.id2token[idx]
    
    def tokenize(self, text):
        return [self.piece_to_id(c) for c in self.__class__.encode_as_pieces(text)]
    
    def detokenize(self, ids):
        pieces = []
        for i in ids:
            if i == self.__class__.unk_id():
                pieces.append(' ⁇ ')
            elif i >= 4:
                pieces.append(self.id_to_piece(i))
        return ' '.join(pieces)

class ChineseTokenizer(SimpleTokenizer):
    @staticmethod
    def encode_as_pieces(text):
        return jieba.lcut(text.strip())
    
    def detokenize(self, ids):
        pieces = []
        for i in ids:
            if i == self.__class__.unk_id():
                pieces.append(' ⁇ ')
            elif i >= 4:
                pieces.append(self.id_to_piece(i))
        return ''.join(pieces)

class EnglishTokenizer(SimpleTokenizer):
    @staticmethod
    def encode_as_pieces(text):
        return [token.strip().lower() for token in text.strip().split() if token]

def create_chinese_tokenizer():
    return ChineseTokenizer([pair['zh'] for pair in texts])

def create_english_tokenizer():
    return EnglishTokenizer([pair['en'] for pair in texts])

class TranslationDataset(Dataset):
    def __init__(self, data, chinese_tokenizer, english_tokenizer):
        super().__init__()
        self.data = data
        self.chinese_tokenizer = chinese_tokenizer
        self.english_tokenizer = english_tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pair = self.data[idx]
        
        zh_tokens = self.chinese_tokenizer.tokenize(pair['zh'])
        en_tokens = self.english_tokenizer.tokenize(pair['en'])
        
        return {
            'zh': zh_tokens,
            'en': en_tokens
        }

def collate_batch(batch, chinese_tokenizer, english_tokenizer, max_len):
    src_batch = []
    tgt_batch = []
    
    for item in batch:
        src = item['zh']
        tgt = [english_tokenizer.bos_id()] + item['en'] + [english_tokenizer.eos_id()]
        src_batch.append(torch.LongTensor(src[:max_len]))
        tgt_batch.append(torch.LongTensor(tgt[:max_len]))
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=chinese_tokenizer.pad_id())
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=english_tokenizer.pad_id())
    return src_batch, tgt_batch

def create_dataloader(chinese_tokenizer, english_tokenizer, batch_size, max_len=512, shuffle=False, drop_last=False):
    dataset = TranslationDataset(texts, chinese_tokenizer, english_tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, chinese_tokenizer, english_tokenizer, max_len))
