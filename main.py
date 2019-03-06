import torch
import jieba
import visdom
import modeling
import random
import torch.utils.data as Data
from tqdm import tqdm
from gensim.models.fasttext import FastText
from torch.autograd import Variable
from torch import nn

random.seed(0)
vis = visdom.Visdom()
EPOCH = 1000
jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

# Load vocabularies
word2vec = FastText.load_fasttext_format('bert-model/wordvec-large')
vocab = {}
vec = []
with open('bert-model/TF.csv') as TF:
    print('建構詞向量...')
    for idx, line in enumerate(tqdm(TF)):
        term = line.split(',')[0]
        vocab[term] = idx
        vec.append(word2vec[term])

del word2vec

# BERT Model
model = modeling.LanGen(hidden_size=768)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
SAVE_EVERY = 50
PENALTY_EPOCH = -1
DRAW_LEARNING_CURVE = False
data = []

# Tokenized input
print('Tokenization...')
with open('pair+lcstcs10000.csv') as PAIR:
    for line in tqdm(PAIR):
        [text, summary] = line.split(',')
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
        summaries.extend(['<PAD>'] * (len(texts) - len(summaries)))
        
        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        data.append((idx_texts, idx_summaries))

# Training
sky_dragon_data = data[:1415]
lcstcs_data = data[1415:]
training_data = sky_dragon_data[:round(len(sky_dragon_data) * 0.8)]
testing_data = sky_dragon_data[round(len(sky_dragon_data) * 0.8):]
training_data.extend(lcstcs_data[:round(len(lcstcs_data) * 0.8)])
testing_data.extend(lcstcs_data[round(len(lcstcs_data) * 0.8):])

BATCH_SIZE = 1

training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    
    for batchIdx in tqdm(range(BATCH_SIZE, len(training_data), BATCH_SIZE)):
        batch = training_data[batchIdx - BATCH_SIZE : batchIdx]
        optimizer.zero_grad()
        training_batch_loss = 0.0
        training_loss = torch.tensor(0.0).cuda()
        for index, (idx_texts, idx_summaries) in enumerate(batch):
            inputTensor = torch.LongTensor([idx_summaries]).cuda()
            targetTensor = torch.LongTensor(idx_texts).cuda()
            loss = torch.mean(model(inputTensor, targetTensor))
            training_loss += loss
            training_batch_loss += loss.item()

        training_loss_sum += training_batch_loss
        training_loss.backward()
        optimizer.step()

    for idx_texts, idx_summaries in tqdm(testing_data):
        inputTensor = torch.LongTensor([idx_summaries]).cuda()
        targetTensor = torch.LongTensor(idx_texts).cuda()
        loss = torch.mean(model(inputTensor, targetTensor))
        testing_loss_sum += loss.item()

    # if (epoch + 1) % SAVE_EVERY == 0:
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'state': model.state_dict(),
    #         'testing_loss': testing_loss_sum / len(testing_data),
    #         'training_loss': training_loss_sum / len(training_data),
    #         'optimizer': optim.state_dict()
    #     }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

    # torch.save({
    #     'epoch': epoch + 1,
    #     'state': model.state_dict(),
    #     'testing_loss': testing_loss_sum / len(testing_data),
    #     'training_loss': training_loss_sum / len(training_data),
    #     'optimizer': optim.state_dict()
    # }, f'checkpoint/bert-LanGen-last.pt')

    log = f'epoch = {epoch + 1}, training_loss = {training_loss_sum / len(training_data)}, testing_loss = {testing_loss_sum / len(testing_data)}'
    training_losses.append(training_loss_sum / len(training_data))
    testing_losses.append(testing_loss_sum / len(testing_data))
    
    vis.text('<b>LOG</b><br>' + log, win='log')
    vis.line(X=list(range(len(training_losses))), Y=training_losses, win='loss', name='training_loss')
    vis.line(X=list(range(len(testing_losses))), Y=testing_losses, win='loss', update='append', name='testing_loss')
    # if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
    #     vis.line(X=list(range(len(learning_curve_training))), Y=learning_curve_training, win='Learning Curve', name='learning_curve_training')
    #     vis.line(X=list(range(len(learning_curve_testing))), Y=learning_curve_testing, win='Learning Curve', update='append', name='learning_curve_testing')

    with open('log.txt', 'a+') as LOG:
        LOG.write(log + '\n')
    tqdm.write(log)
