import torch
import torch.nn as nn
import torch.optim as optim # 导入优化器
from Transformer import Transformer

# 创建模拟数据 (假设数据已经是LongTensor类型)
# src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # (10, 120)
# tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))  # (10, 100)

# 模型前向传播
# output = model(
#     src=src,                    # (10, 120)
#     tgt=tgt,                    # (10, 100)
#     src_mask=src_mask,          # (10, 1, 1, 120)
#     tgt_mask=tgt_mask          # (10, 1, 100, 100)
# )

# print("输出形状:", output.shape)  # 应为 (10, 100, tgt_vocab_size)

sentences = [
    ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
    ['我 爱 学习 人工智能', 'I love studying AI'],
    ['深度学习 改变 世界', ' DL changed the world'],
    ['自然语言处理 很 强大', 'NLP is powerful'],
    ['神经网络 非常 复杂', 'Neural-networks are complex']
]

from collections import Counter # 导入Counter 类
# 定义TranslationCorpus类
class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        # 计算源语言和目标语言的最大句子长度，并分别加1和2以容纳填充符和特殊符号
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2
        # 创建源语言和目标语言的词汇表
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        # 创建索引到单词的映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
    # 定义创建词汇表的函数
    def create_vocabularies(self):
        # 统计源语言和目标语言的单词频率
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())        
        # 创建源语言和目标语言的词汇表，并为每个单词分配一个唯一的索引
        src_vocab = {'<pad>': 0, **{word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 
                     **{word: i+3 for i, word in enumerate(tgt_counter)}}        
        return src_vocab, tgt_vocab
    # 定义创建批次数据的函数
    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []
        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]
            # 将源语言和目标语言的句子转换为索引序列
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] for word in tgt_sentence.split()] + [self.tgt_vocab['<eos>']]            
            # 对源语言和目标语言的序列进行填充
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))            
            # 将处理好的序列添加到批次中
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']] * (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])        
          # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)            
        return input_batch, output_batch, target_batch

# 创建语料库类实例
corpus = TranslationCorpus(sentences)

# 示例数据参数
batch_size = 3
src_seq_len = corpus.src_len
tgt_seq_len = corpus.tgt_len - 1
pad_idx = 0  # 填充符的ID
src_vocab_size = len(corpus.src_vocab)  # 根据实际情况修改
tgt_vocab_size = len(corpus.tgt_vocab)  # 根据实际情况修改

# 初始化模型
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
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

criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器
epochs = 5 # 训练轮次
for epoch in range(epochs): # 训练100轮
    optimizer.zero_grad() # 梯度清零
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size) # 创建训练数据
    src_mask = Transformer.generate_padding_mask(enc_inputs, pad_idx)
    tgt_mask = Transformer.generate_padding_mask(dec_inputs, pad_idx) & Transformer.generate_causal_mask(tgt_seq_len)
    outputs = model(enc_inputs, dec_inputs, src_mask, tgt_mask) # 获取模型输出 
    loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1)) # 计算损失
    if (epoch + 1) % 1 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward()# 反向传播        
    optimizer.step()# 更新参数

# 创建一个大小为1的批次，目标语言序列dec_inputs在测试阶段，仅包含句子开始符号<sos>
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1,test_batch=True) 
src_mask = Transformer.generate_padding_mask(enc_inputs, pad_idx)
tgt_mask = Transformer.generate_padding_mask(dec_inputs, pad_idx) & Transformer.generate_causal_mask(tgt_seq_len)
predict = model(enc_inputs, dec_inputs, src_mask, tgt_mask) # 用模型进行翻译
predict = predict.view(-1, len(corpus.tgt_vocab)) # 将预测结果维度重塑
predict = predict.data.max(1, keepdim=True)[1] # 找到每个位置概率最大的单词的索引
# 解码预测的输出，将所预测的目标句子中的索引转换为单词
translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
# 将输入的源语言句子中的索引转换为单词
input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
print(input_sentence, '->', translated_sentence) # 打印原始句子和翻译后的句子
