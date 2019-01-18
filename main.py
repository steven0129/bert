import torch
import jieba
import visdom
from torch import nn
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertAdam

vis = visdom.Visdom()
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
        self.model.encoder.layer = self.model.encoder.layer[:3]
        self.model.eval()
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(768, len(vocab), cutoffs=[782])
    
    def forward(self, x, target):
        encoder_layers, _ = self.model(x)
        out = self.adaptive_softmax(encoder_layers[-1].view(-1, 768), target)
        return out

model = LanGen()
model.cuda()
optimizer = torch.optim.Adam(model.parameters())
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
        summaries.insert(0, '<SOS>')
        summaries.append('<EOS>')
        summaries.extend(['<PAD>'] * (len(texts) - len(summaries)))

        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        data.append((idx_texts, idx_summaries))

# Training
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8) + 1:]
training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    
    for idx_texts, idx_summaries in tqdm(training_data):
        optimizer.zero_grad()
        loss = model(torch.LongTensor([idx_summaries]).cuda(), torch.LongTensor(idx_texts).cuda()).loss
        loss.backward()
        training_loss_sum += loss.item()
        optimizer.step()

    for idx_texts, idx_summaries in tqdm(testing_data):
        loss = model(torch.LongTensor([idx_summaries]).cuda(), torch.LongTensor(idx_texts).cuda()).loss
        testing_loss_sum += loss.item()

    torch.save({
        'epoch': epoch + 1,
        'state': model.state_dict(),
        'loss': training_loss_sum / len(training_data),
        'optimizer': optimizer.state_dict()
    }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

    torch.save({
        'epoch': epoch + 1,
        'state': model.state_dict(),
        'loss': training_loss_sum / len(training_data),
        'optimizer': optimizer.state_dict()
    }, f'checkpoint/bert-LanGen-last.pt')

    log = f'epoch = {epoch + 1}, training_loss = {training_loss_sum / len(training_data)}, testing_loss = {testing_loss_sum / len(testing_data)}'
    training_losses.append(training_loss_sum / len(training_data))
    testing_losses.append(testing_loss_sum / len(testing_data))
    
    vis.text('<b>LOG</b><br>' + log, win='log')
    vis.line(X=list(range(len(training_losses))), Y=training_losses, win='loss', name='training_loss')
    vis.line(X=list(range(len(testing_losses))), Y=testing_losses, win='loss', update='append', name='testing_loss')

    with open('log.txt', 'a+') as LOG:
        LOG.write(log + '\n')
    tqdm.write(log)
