import torch
import re
import jieba

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 原始数据
examples = [
    ['我爱学习人工智能', 'I love studying AI'],
    ['深度学习改变世界', ' Deep learning changed the world'],
    ['自然语言处理很强大', 'NLP is powerful'],
    ['神经网络非常复杂', 'Neural-networks are complex']
]

def clean_text(example):
    texts = example[:]
    for ix, text in enumerate(texts):
        # 过滤非法字符
        text = re.sub(r'[�◆★【】▲▼■●]', '', text)
        # 合并连续空格
        text = re.sub(r'\s+', ' ', text)
        texts[ix] = text.strip()
    
    if 1 < len(texts[0]) < 150 and 1 < len(texts[1]) < 200:
        return texts

# 数据清洗
sentences = []
for example in examples:
    cleaned = clean_text(example)
    if cleaned:
        sentences.append(cleaned)

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
        self._token2id = {}
        self._id2token = {}
        # 添加特殊标记
        for i, token in enumerate(special_tokens):
            self._token2id[token] = i
            self._id2token[i] = token
        for i, token in enumerate(sorted(counter)):
            j = i + len(special_tokens)
            self._token2id[token] = j
            self._id2token[j] = token
    
    def vocab_size(self):
        return len(self._token2id)
    
    def __len__(self):
        return self.vocab_size()
    
    def piece_to_id(self, piece):
        return self._token2id.get(piece, self.__class__.unk_id())
    
    def id_to_piece(self, idx):
        if idx not in self._id2token:
            raise Exception("IndexError: Out of range: piece id is out of range.")
        return self._id2token[idx]
    
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
        return list(jieba.cut(text.strip()))
    
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
