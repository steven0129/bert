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
from elmoformanylangs import Embedder

random.seed(0)
vis = visdom.Visdom()
EPOCH = 5000
jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)
embedder = Embedder('bert-model/elmo/')
cosSim = nn.CosineSimilarity()

# Load vocabularies
vocab = {}
idx2vocab = {}
with open('bert-model/TF.csv') as TF:
    print('建構詞向量...')
    for idx, line in enumerate(tqdm(TF)):
        term = line.split(',')[0]
        vocab[term] = idx
        idx2vocab[idx] = term

# BERT Model
model = modeling.TransformerNoEmbed(vocab=vocab, hidden_size=1024, enc_num_layer=3, dec_num_layer=3)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
label_smoothing = modeling.LabelSmoothing(len(vocab), 0, 0.1)
label_smoothing.cuda()
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
        # summaries.extend(['<PAD>'] * (len(texts) - len(summaries)))
        
        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        wordvec_texts = embedder.sents2elmo([texts[:512]])[0]
        # idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        wordvec_summaries = embedder.sents2elmo([summaries[:512]])[0]
        data.append((idx_texts, wordvec_texts, wordvec_summaries))

# Training
random.Random(0).shuffle(data)
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8):]
BATCH_SIZE = 1
training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    
    for index, (idx_texts, wordvec_texts, wordvec_summaries) in enumerate(tqdm(training_data)):
        optimizer.zero_grad()
        inputTensor = torch.from_numpy(wordvec_summaries).cuda()
        tgtInputTensor = torch.from_numpy(wordvec_texts[:-1]).cuda()
        tgtTensor = torch.LongTensor(idx_texts[1:]).cuda()
        output = model(inputTensor.unsqueeze(0), tgtInputTensor.unsqueeze(0))
        loss = label_smoothing(output, tgtTensor)

        # Multi-head attention with disagreement regularization
        # disagreement = torch.zeros(1).cuda()
        # disagreement_idx = 0
        # for name, param in model.named_parameters():
        #     if name.endswith('.attention.self.value.weight'):
        #         summation = torch.zeros(1).cuda()
        #         for v1 in param.data:
        #             v1 = v1.cuda()
        #             v1 = v1.expand(param.data.size(0), param.data.size(1))
        #             v2 = param.data.cuda()
        #             summation += torch.mean(0.5 * (1 + cosSim(v1, v2)))

        #         disagreement +=  summation / param.data.size(0)
        #         disagreement_idx += 1
        
        # disagreement_avg = disagreement / disagreement_idx
        # tqdm.write(f'Average disagreement: {str(disagreement_avg.item())}')
        # loss_penalty = loss * disagreement_avg
        # loss_penalty.backward()
        loss.backward()
        training_loss_sum += loss.item()
        optimizer.step()

    for idx_texts, wordvec_summaries in tqdm(testing_data):
        inputTensor = torch.from_numpy(wordvec_summaries).cuda()
        targetTensor = torch.LongTensor(idx_texts).cuda()
        output, _ = model(inputTensor.unsqueeze(0))
        loss = label_smoothing(output, targetTensor)
        testing_loss_sum += loss.item()

    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint-generator-pretrain/bert-LanGen-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint-generator-pretrain/bert-LanGen-last.pt')

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
