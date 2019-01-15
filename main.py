import torch
import jieba
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn import preprocessing
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

MODEL_PATH = 'bert-model'
EPOCH = 100
jieba.load_userdict('bert-model/dict-traditional.txt')

# Load vocabularies
vocab = {}
with open('bert-model/TF.csv') as TF:
    for idx, line in enumerate(TF):
        term = line.split(',')[0]
        vocab[term] = idx

# BERT Model
class LanGen(nn.Module):
    def __init__(self):
        super(LanGen, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        self.model.embeddings.word_embeddings = nn.Embedding(len(vocab), 768)
        self.model.eval()
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(768, len(vocab), cutoffs=[782])
    
    def forward(self, x, target):
        encoder_layers, _ = self.model(x)
        out = self.adaptive_softmax(encoder_layers[-1].view(512, 768), target)
        return out

model = LanGen()
model.cuda()
optimizer = BertAdam(model.parameters(), lr=0.1)
data = []

# Tokenized input
with open('pair.csv') as PAIR:
    for line in tqdm(PAIR):
        [text, summary] = line.split(',')
        texts = []
        summaries = []
        paras = text.split('<newline>')
        for para in paras:
            texts.extend(list(jieba.cut(para)))
            texts.append('<newline>')

        summaries.extend(list(jieba.cut(summary)))

        texts.insert(0, '<SOS>')
        texts.append('<EOS>')
        texts.extend(['<PAD>'] * (512 - len(texts)))
        summaries.insert(0, '<SOS>')
        summaries.append('<EOS>')
        summaries.extend(['<PAD>'] * (512 - len(summaries)))

        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        data.append((idx_texts, idx_summaries))

# Training
for epoch in tqdm(range(EPOCH)):
    loss_sum = 0.0
    for idx_texts, idx_summaries in tqdm(data):
        optimizer.zero_grad()
        loss = model(torch.LongTensor([idx_summaries]).cuda(), torch.LongTensor(idx_texts).cuda()).loss
        loss.backward()
        loss_sum += loss.item() / 512
        optimizer.step()

    torch.save({
        'epoch': epoch + 1,
        'state': model.state_dict(),
        'loss': loss_sum / len(data),
        'optimizer': optimizer.state_dict()
    }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

    tqdm.write(f'epoch = {epoch + 1}, loss = {loss_sum / len(data)}')