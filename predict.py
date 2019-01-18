import torch
import jieba
from torch import nn
from pytorch_pretrained_bert import BertModel, BertAdam

MODEL_PATH = 'bert-model'
jieba.load_userdict('bert-model/dict-traditional.txt')

# Load vocabularies
vocab = {}
id2vocab = {}
with open('bert-model/TF.csv') as TF:
    for idx, line in enumerate(TF):
        term = line.split(',')[0]
        vocab[term] = idx
        id2vocab[idx] = term

# BERT Model
class LanGen(nn.Module):
    def __init__(self):
        super(LanGen, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        self.model.embeddings.word_embeddings = nn.Embedding(len(vocab), 768)
        self.model.encoder.layer = self.model.encoder.layer[:3]
        self.model.eval()
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(768, len(vocab), cutoffs=[782])
    
    def forward(self, x):
        encoder_layers, _ = self.model(x)
        out = self.adaptive_softmax.predict(encoder_layers[-1].view(512, 768))
        return out

model = LanGen()
checkpoint = torch.load('checkpoint/bert-LanGen-last.pt')
model.load_state_dict(checkpoint['state'])
print('Info of model:')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Loss: {checkpoint["loss"]}')

summary = '大理國無量山無量劍派的練武廳中，舉辦了五年一次的比武鬥劍大會，由無量劍的東、北、西三宗互相比試。此次是第九次大會。'
summary = list(jieba.cut(summary))
summary.insert(0, '<SOS>')
summary.append('<EOS>')
summary.extend(['<PAD>'] * (512 - len(summary)))
idx_summary = list(map(lambda x: vocab[x], summary[:512]))
out = model(torch.LongTensor([idx_summary])).tolist()
print(list(map(lambda x: id2vocab[x], out)))
