import torch
import jieba
import visdom
import modeling
import random
import torch.utils.data as Data
import sys
from tqdm import tqdm
from gensim.models.fasttext import FastText
from torch.autograd import Variable
from torch import nn
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert.optimization import WarmupHalfLinearSchedule, WarmupCosineWithWarmupRestartsSchedule
from loss import FocalLoss


random.seed(0)
vis = visdom.Visdom()
EPOCH = 2000
jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

# Load vocabularies
word2vec = FastText.load_fasttext_format('bert-model/wordvec-large.dim1024')
vocab = {}
idx2vocab = {}
vec = []
with open('bert-model/TF.csv') as TF:
    print('建構詞向量...')
    for idx, line in enumerate(tqdm(TF)):
        term = line.split(',')[0]
        vocab[term] = idx
        idx2vocab[idx] = term
        vec.append(word2vec[term])

del word2vec

# BERT Model
model = modeling.BertNoEmbed(vocab=vocab, hidden_size=1024, enc_num_layer=3)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

model.apply(weight_init)
model.cuda()

# label_smoothing = modeling.LabelSmoothing(len(vocab), 0, 0.1)
# label_smoothing.cuda()
focal_loss = FocalLoss(class_num=len(vocab))
focal_loss.cuda()
SAVE_EVERY = 5
PENALTY_EPOCH = -1
DRAW_LEARNING_CURVE = False
data = []

# Tokenized input
print('Tokenization...')
with open('pair.csv') as PAIR:
    for line in tqdm(PAIR):
        [text, summary, _] = line.split(',')
        texts = []
        summaries = []
        paras = text.split('<newline>')
        for para in paras:
            texts.extend(list(jieba.cut(para)))
            texts.append('<newline>')

        summaries.extend(list(jieba.cut(summary)))

        texts = list(filter(lambda x: x != '\n', texts))
        summaries = list(filter(lambda x: x != '\n', summaries))

        texts.insert(0, '<SOS>')
        texts.append('<EOS>')
        summaries.insert(0, '<SOS>')
        summaries.append('<EOS>')
        
        idx_texts = list(map(lambda x: vocab[x], texts[:1500]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        wordvec_texts = list(map(lambda x: vec[x], idx_texts))
        wordvec_summaries = list(map(lambda x: vec[x], idx_summaries))
        
        data.append((idx_texts, wordvec_texts, wordvec_summaries))

# Training
random.Random(0).shuffle(data)
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8):]
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = BertAdam(model.parameters(), lr=0.001, weight_decay=0.1, schedule=WarmupHalfLinearSchedule(warmup=0.05, t_total=EPOCH * len(training_data)))
BATCH_SIZE = 1
training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    
    for index, (idx_texts, wordvec_texts, wordvec_summaries) in enumerate(tqdm(training_data)):
        optimizer.zero_grad()
        
        inputTensor = torch.FloatTensor(wordvec_summaries).cuda()
        tgtTensor = torch.LongTensor(idx_texts).cuda()
        
        non_masked_position = torch.ones(inputTensor.size(0)).cuda()
        masked_position = torch.zeros(tgtTensor.size(0) - inputTensor.size(0)).cuda()
        attn_masked = torch.cat((non_masked_position, masked_position))
        inputTensor = torch.cat((inputTensor, torch.zeros(tgtTensor.size(0) - inputTensor.size(0), inputTensor.size(1)).cuda()))
        output = model(inputTensor.unsqueeze(0), attn_masked.unsqueeze(0))
        # loss = label_smoothing(output, tgtTensor)
        loss = focal_loss(output, tgtTensor)
        training_loss_sum += loss.item()
        loss.backward()

        optimizer.step()

    for idx_texts, wordvec_texts, wordvec_summaries in tqdm(testing_data):
        inputTensor = torch.FloatTensor(wordvec_summaries).cuda()
        tgtTensor = torch.LongTensor(idx_texts).cuda()
        
        non_masked_position = torch.ones(inputTensor.size(0)).cuda()
        masked_position = torch.zeros(tgtTensor.size(0) - inputTensor.size(0)).cuda()
        attn_masked = torch.cat((non_masked_position, masked_position))
        inputTensor = torch.cat((inputTensor, torch.zeros(tgtTensor.size(0) - inputTensor.size(0), inputTensor.size(1)).cuda()))
        output = model(inputTensor.unsqueeze(0), attn_masked.unsqueeze(0))
        # loss = label_smoothing(output, tgtTensor)
        loss = focal_loss(output, tgtTensor)
        testing_loss_sum += loss.item()

    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'full_model': model,
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'full_model': model,
            'training_loss': training_loss_sum / len(training_data),
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
