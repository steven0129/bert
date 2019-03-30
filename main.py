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
        summaries.extend(['<PAD>'] * (len(texts) - len(summaries)))
        
        idx_texts = list(map(lambda x: vocab[x], texts[:512]))
        idx_summaries = list(map(lambda x: vocab[x], summaries[:512]))
        data.append((idx_texts, idx_summaries))

# Training
training_data = data[:round(len(data))]
testing_data = []
training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    learning_curve_training = []
    learning_curve_testing = []
    optim = optimizer
    
    for index, (idx_texts, idx_summaries) in enumerate(tqdm(training_data)):
        optim.zero_grad()
        inputTensor = torch.LongTensor([idx_summaries]).cuda()
        targetTensor = torch.LongTensor(idx_texts).cuda()
        output, _ = model(inputTensor)
        loss = label_smoothing(output, targetTensor)
        loss.backward()
        training_loss_sum += loss.item()
        optim.step()

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
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            # 'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-last.pt')

    log = f'epoch = {epoch + 1}, training_loss = {training_loss_sum / len(training_data)}'
    training_losses.append(training_loss_sum / len(training_data))
    # testing_losses.append(testing_loss_sum / len(testing_data))
    
    vis.text('<b>LOG</b><br>' + log, win='log')
    vis.line(X=list(range(len(training_losses))), Y=training_losses, win='loss', name='training_loss')
    # vis.line(X=list(range(len(testing_losses))), Y=testing_losses, win='loss', update='append', name='testing_loss')
    if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
        vis.line(X=list(range(len(learning_curve_training))), Y=learning_curve_training, win='Learning Curve', name='learning_curve_training')
        vis.line(X=list(range(len(learning_curve_testing))), Y=learning_curve_testing, win='Learning Curve', update='append', name='learning_curve_testing')

    with open('log.txt', 'a+') as LOG:
        LOG.write(log + '\n')
    tqdm.write(log)