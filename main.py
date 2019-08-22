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
from pytorch_pretrained_bert.optimization import BertAdam

random.seed(0)
vis = visdom.Visdom()
EPOCH = 1000
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
model.load_state_dict(torch.load('checkpoint/bert-LanGen-last.pt')['state'])
model.cuda()
d_net = modeling.TextCNNClassify(vocab, vec, num_labels=2)
d_net.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_d = torch.optim.SGD(d_net.parameters(), lr=0.01)

label_smoothing = modeling.LabelSmoothing(len(vocab), 0, 0.1)
label_smoothing.cuda()
gan_loss = GANLoss()
gan_loss.cuda()
G_STEP = 1
D_STEP = 3
D_PRE = 5
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
        
        idx_random_texts = list(map(lambda x: random.randint(0, x - 1), [len(vocab)] * len(texts)))
        idx_random_summaries = list(map(lambda x: random.randint(0, x - 1), [len(vocab)] * len(summaries)))
        idx_texts = list(map(lambda x: vocab[x], texts[:1500]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        wordvec_texts = list(map(lambda x: vec[x], idx_texts))
        wordvec_summaries = list(map(lambda x: vec[x], idx_summaries))
        wordvec_random_summaries = list(map(lambda x: vec[x], idx_random_summaries))
        
        data.append((idx_texts, idx_random_texts, idx_summaries, wordvec_random_summaries, wordvec_texts, wordvec_summaries))

# Training
random.Random(0).shuffle(data)
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8):]
d_summary_data = list(map(lambda x: (x[2], 0), training_data))
d_text_data = list(map(lambda x: (x[0], 1), training_data))
d_data = d_summary_data + d_text_data
random.Random(0).shuffle(d_data)
training_losses = []
testing_losses = []
d_losses = []
g_losses = []

# Pretrain Discriminator
loss_fct = nn.CrossEntropyLoss()

d_losses_pretrained = []
d_accuracy_pretrained = []

for epoch in tqdm(range(D_PRE)):
    TEXT = torch.cuda.LongTensor([1])
    SUMMARY = torch.cuda.LongTensor([0])
    d_loss_sum = 0.0
    d_correct = 0

    for index, (idxs, ans) in enumerate(tqdm(d_data)):
        optimizer_d.zero_grad()
        inputTensor = torch.LongTensor([idxs]).cuda()
        out = d_net(inputTensor)

        if ans == 0: # Summary
            d_correct += (torch.argmax(out.view(-1, d_net.num_labels), dim=1) == SUMMARY.item()).sum().item()
            d_loss = loss_fct(out.view(-1, d_net.num_labels), SUMMARY.view(-1))
        elif ans == 1: # TEXT
            d_correct += (torch.argmax(out.view(-1, d_net.num_labels), dim=1) == TEXT.item()).sum().item()
            d_loss = loss_fct(out.view(-1, d_net.num_labels), TEXT.view(-1))
        
        d_loss_sum += d_loss.item()
        d_loss.backward()
        optimizer_d.step()
            
    tqdm.write(f'd_loss_sum = {d_loss_sum / len(d_data)}, accuracy = {d_correct / len(d_data) * 100}%')

    d_losses_pretrained.append(d_loss_sum / len(d_data))
    d_accuracy_pretrained.append(d_correct / len(d_data))
    vis.line(X=list(range(len(d_losses_pretrained))), Y=d_losses_pretrained, win='d_loss_pretrained', name=f'd_loss')
    vis.line(X=list(range(len(d_accuracy_pretrained))), Y=d_accuracy_pretrained, win='d_accuracy_pretrained', name=f'd_accuracy')

torch.save({
    'epoch': epoch + 1,
    'state': d_net.state_dict(),
    'd_loss': d_loss_sum / len(d_data),
    'optimizer_d': optimizer_d.state_dict()
}, f'checkpoint/bert-Discriminator-pretrained.pt')

# Adversarial Training
for epoch in tqdm(range(EPOCH)):
    TEXT = torch.cuda.LongTensor([1])
    SUMMARY = torch.cuda.LongTensor([0])
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    d_loss_sum = 0.0
    g_loss_sum = 0.0
    learning_curve_training = []
    learning_curve_testing = []

    for index, (idx_texts, idx_random_texts, idx_summaries, wordvec_random_summaries, wordvec_texts, wordvec_summaries) in enumerate(tqdm(training_data)):
        # Discriminator Training
        for d_step in range(D_STEP):
            # x~summary: D(G(x)) --> SUMMARY
            optimizer_d.zero_grad()
            inputTensor = torch.FloatTensor(wordvec_summaries).cuda()
            tgtTensor = torch.LongTensor(idx_texts).cuda()
            
            non_masked_position = torch.ones(inputTensor.size(0)).cuda()
            masked_position = torch.zeros(tgtTensor.size(0) - inputTensor.size(0)).cuda()
            attn_masked = torch.cat((non_masked_position, masked_position))
            inputTensor = torch.cat((inputTensor, torch.zeros(tgtTensor.size(0) - inputTensor.size(0), inputTensor.size(1)).cuda()))
            
            noise = model(inputTensor.unsqueeze(0), attn_masked.unsqueeze(0))
            d_loss = d_net(torch.unsqueeze(torch.argmax(noise, dim=1), 0).detach(), labels=SUMMARY) # Split generator from graph
            d_loss_sum += d_loss.item()
            d_loss.backward()
            optimizer_d.step()

            # x~text: x --> TEXT
            optimizer_d.zero_grad()
            inputTensor = torch.LongTensor([idx_texts]).cuda()
            d_loss = d_net(inputTensor, labels=TEXT)
            d_loss_sum += d_loss.item()
            d_loss.backward()
            optimizer_d.step()

        # Generator Training
        for g_step in range(G_STEP):
            # x~summary: D(G(x)) --> TEXT
            optimizer.zero_grad()
            inputTensor = torch.FloatTensor(wordvec_summaries).cuda()
            tgtTensor = torch.LongTensor(idx_texts).cuda()
            
            non_masked_position = torch.ones(inputTensor.size(0)).cuda()
            masked_position = torch.zeros(tgtTensor.size(0) - inputTensor.size(0)).cuda()
            attn_masked = torch.cat((non_masked_position, masked_position))
            inputTensor = torch.cat((inputTensor, torch.zeros(tgtTensor.size(0) - inputTensor.size(0), inputTensor.size(1)).cuda()))
            
            noise = model(inputTensor.unsqueeze(0), attn_masked.unsqueeze(0))
            output = d_net(torch.unsqueeze(torch.argmax(noise, dim=1), 0)) # Judge noise to TEXT
            rewards = torch.argmax(output.detach(), dim=1)

            # GANLoss
            # if rewards=1, loss down --> TEXT
            # if rewards=0, loss up --> SUMMARY
            g_loss = gan_loss(noise, tgtTensor, rewards) 
            g_loss_sum += g_loss.item()
            g_loss.backward()
            optimizer.step()

        tqdm.write(f'g_loss_sum = {g_loss_sum / ((index + 1) * G_STEP)}, d_loss_sum = {d_loss_sum / ((index + 1) * (D_STEP * 2))}')

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
            'training_loss': g_loss_sum / (len(training_data) * G_STEP),
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
