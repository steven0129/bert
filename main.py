import torch
import jieba
import visdom
import modeling
import sys
import random
import torch.utils.data as Data
from tqdm import tqdm
from loss import GANLoss
from gensim.models.fasttext import FastText
from torch.autograd import Variable
from torch import nn
from pytorch_pretrained_bert import BertForSequenceClassification, BertConfig

vis = visdom.Visdom()
EPOCH = 1000
jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

# Load vocabularies
word2vec = FastText.load_fasttext_format('bert-model/wordvec-large-new.dim768')
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
model = modeling.LanGen(vocab, vec, hidden_size=768)
model.load_state_dict(torch.load('bert-generator-base.pt')['state'])
model.cuda()
d_net = BertForSequenceClassification(BertConfig(vocab_size_or_config_json_file='bert-model/bert_config.json'), 2)
# d_net = modeling.TextCNN(embedding_dimension=len(vocab))
d_net.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_d = torch.optim.SGD(model.parameters(), lr=0.01)
label_smoothing = modeling.LabelSmoothing(len(vocab), 0, 0.1)
label_smoothing.cuda()
gan_loss = GANLoss()
gan_loss.cuda()
criterion = nn.BCELoss()
criterion.cuda()
G_STEP = 1
D_STEP = 1
SAVE_EVERY = 50
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
        summaries.extend(['<PAD>'] * (len(texts) - len(summaries)))
        
        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        data.append((idx_texts, idx_summaries))

# Training
training_data = data[:round(len(data))]
testing_data = []
training_losses = []
testing_losses = []
d_losses = []
g_losses = []

for epoch in tqdm(range(EPOCH)):
    REAL = torch.cuda.LongTensor([1])
    FAKE = torch.cuda.LongTensor([0])
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    d_loss_sum = 0.0
    g_loss_sum = 0.0
    learning_curve_training = []
    learning_curve_testing = []

    for index, (idx_texts, idx_summaries) in enumerate(tqdm(training_data)):
        # Discriminator Training
        optimizer_d.zero_grad()

        for _ in range(D_STEP):
            random.shuffle(idx_summaries)
            inputTensor = torch.LongTensor([idx_summaries]).cuda()
            noise, _ = model(inputTensor)
            d_loss = d_net(torch.unsqueeze(torch.argmax(noise, dim=1), 0).detach(), labels=FAKE) # Split generator from graph
            d_loss_sum += d_loss.item()
            d_loss.backward()

        inputTensor = torch.LongTensor([idx_texts]).cuda()
        d_loss = d_net(inputTensor, labels=REAL)
        d_loss_sum += d_loss.item()
        d_loss_sum = d_loss_sum
        d_loss.backward()
        
        optimizer_d.step()

        # Generator Training
        optimizer.zero_grad()

        for _ in range(G_STEP):
            random.shuffle(idx_summaries)
            inputTensor = torch.LongTensor([idx_summaries]).cuda()
            targetTensor = torch.LongTensor(idx_texts).cuda()
            noise, _ = model(inputTensor)
            output = d_net(torch.unsqueeze(torch.argmax(noise, dim=1), 0).detach()) # Judge noise to REAL
            rewards = torch.argmax(output, dim=1)
            g_loss = gan_loss(noise, targetTensor, rewards)
            g_loss_sum += g_loss.item()
            g_loss.backward()

        optimizer.step()

        tqdm.write(f'g_loss_sum = {g_loss_sum / ((index + 1) * G_STEP)}, d_loss_sum = {d_loss_sum / ((index + 1) * (D_STEP + 1))}')

        # if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
        #     learning_curve_training.append(training_loss_sum / (index + 1))
        #     testing_every_loss = []
        #     for idx_texts, idx_summaries in testing_data:
        #         inputTensor = torch.LongTensor([idx_summaries]).cuda()
        #         targetTensor = torch.LongTensor(idx_texts).cuda()
        #         output, _ = model(inputTensor)
        #         testing_every_loss.append(label_smoothing(output, targetTensor).item())
        #     learning_curve_testing.append(sum(testing_every_loss) / len(testing_every_loss))

    # for idx_texts, idx_summaries in tqdm(testing_data):
    #     inputTensor = torch.LongTensor([idx_summaries]).cuda()
    #     targetTensor = torch.LongTensor(idx_texts).cuda()
    #     output, _ = model(inputTensor)
    #     loss = label_smoothing(output, targetTensor)
    #     testing_loss_sum += loss.item()

    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            # 'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': g_loss_sum / (len(training_data) * G_STEP),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            # 'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': g_loss_sum / (len(training_data) * (D_STEP + 1)),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-last.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': d_net.state_dict(),
            'd_loss': d_loss_sum / (len(training_data) * (D_STEP + 1)),
            'g_loss': g_loss_sum / (len(training_data) * G_STEP),
            'optimizer_d': optimizer_d.state_dict()
        }, f'checkpoint/bert-Discriminator-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': d_net.state_dict(),
            'd_loss': d_loss_sum / (len(training_data) * (D_STEP + 1)),
            'g_loss': g_loss_sum / (len(training_data) * G_STEP),
            'optimizer_d': optimizer_d.state_dict()
        }, f'checkpoint/bert-Discriminator-last.pt')

    # log = f'epoch = {epoch + 1}, training_loss = {training_loss_sum / len(training_data)}'
    log = f'epoch = {epoch + 1}, g_loss = {g_loss_sum / (len(training_data) * G_STEP)}, d_loss = {d_loss_sum / (len(training_data) * (D_STEP + 1))}'
    g_losses.append(g_loss_sum / (len(training_data) * G_STEP))
    d_losses.append(d_loss_sum / (len(training_data) * (D_STEP + 1)))
    # testing_losses.append(testing_loss_sum / len(testing_data))
    
    vis.text('<b>LOG</b><br>' + log, win='log')
    # vis.line(X=list(range(len(training_losses))), Y=training_losses, win='loss', name='training_loss')
    # vis.line(X=list(range(len(testing_losses))), Y=testing_losses, win='loss', update='append', name='testing_loss')
    vis.line(X=list(range(len(d_losses))), Y=d_losses, win='gan_loss', name='d_loss')
    vis.line(X=list(range(len(g_losses))), Y=g_losses, win='gan_loss', update='append', name='g_loss')
    if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
        vis.line(X=list(range(len(learning_curve_training))), Y=learning_curve_training, win='Learning Curve', name='learning_curve_training')
        vis.line(X=list(range(len(learning_curve_testing))), Y=learning_curve_testing, win='Learning Curve', update='append', name='learning_curve_testing')

    with open('log.txt', 'a+') as LOG:
        LOG.write(log + '\n')
    tqdm.write(log)