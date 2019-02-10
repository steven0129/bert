import torch
import jieba
import visdom
import modeling
from tqdm import tqdm
from multiprocessing import Pool
from gensim.models.fasttext import FastText
from torch.autograd import Variable

vis = visdom.Visdom()
word2vec = FastText.load_fasttext_format('bert-model/wordvec-small')
EPOCH = 1000
UNSUPERVISED_EPOCH = 20
process = Pool()
jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

# Load vocabularies
vocab = {}
vec = []
with open('bert-model/TF.csv') as TF:
    for idx, line in enumerate(TF):
        term = line.split(',')[0]
        vocab[term] = idx
        vec.append(word2vec[term])

# BERT Model
model = modeling.LanGen(vocab, vec)
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
label_smoothing = modeling.LabelSmoothing(len(vocab), 0, 0.1)
label_smoothing.cuda()
DRAW_LEARNING_CURVE = True
SAVE_EVERY = 50
data = []
unsupervised_data = []

# Unsupervised Learning
unsupervised_losses = []
with open('bert-model/unsupervised.txt') as UN:
    for line in tqdm(UN):
        [word, context] = line.split(',')
        context = context.strip().split(' ')
        idx_word = [vocab[word]] + [vocab['<PAD>']] * (len(context) - 1)
        idx_context = list(map(lambda x: vocab[x], context))
        unsupervised_data.append((idx_word, idx_context))

for epoch in tqdm(range(UNSUPERVISED_EPOCH)):
    loss_sum = 0.0

    for idx, (idx_word, idx_context) in enumerate(tqdm(unsupervised_data)):
        optimizer.zero_grad()
        inputTensor = torch.LongTensor([idx_word]).cuda()
        targetTensor = torch.LongTensor(idx_context).cuda()
        output, _ = model(inputTensor)
        loss = label_smoothing(output, targetTensor)
        loss.backward()
        loss_sum += loss.item()
        if idx % 1000 == 0:
            tqdm.write(f'epoch = {epoch + 1}, 第{idx}筆loss: {loss_sum / (idx + 1)}')
        optimizer.step()

    unsupervised_losses.append(loss_sum / len(unsupervised_data))
    vis.line(X=list(range(len(unsupervised_losses))), Y=unsupervised_losses, win='unsupervised_loss', name='unsupervised_loss')

    torch.save({
        'epoch': epoch + 1,
        'state': model.state_dict(),
        'loss': loss_sum / len(unsupervised_data),
        'optimizer': optimizer.state_dict()
    }, f'checkpoint/unsupervised-bert-epoch{epoch + 1}.pt')

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
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8) + 1:]
training_losses = []
testing_losses = []

for epoch in tqdm(range(EPOCH)):
    training_loss_sum = 0.0
    testing_loss_sum = 0.0
    learning_curve_training = []
    learning_curve_testing = []
    
    for index, (idx_texts, idx_summaries) in enumerate(tqdm(training_data)):
        optimizer.zero_grad()
        inputTensor = torch.LongTensor([idx_summaries]).cuda()
        targetTensor = torch.LongTensor(idx_texts).cuda()
        output, _ = model(inputTensor)
        loss = label_smoothing(output, targetTensor)
        loss.backward()
        training_loss_sum += loss.item()
        optimizer.step()

        if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
            learning_curve_training.append(training_loss_sum / (index + 1))
            testing_every_loss = []
            for idx_texts, idx_summaries in testing_data:
                inputTensor = torch.LongTensor([idx_summaries]).cuda()
                targetTensor = torch.LongTensor(idx_texts).cuda()
                output, _ = model(inputTensor)
                testing_every_loss.append(label_smoothing(output, targetTensor).item())
            learning_curve_testing.append(sum(testing_every_loss) / len(testing_every_loss))

    for idx_texts, idx_summaries in tqdm(testing_data):
        inputTensor = torch.LongTensor([idx_summaries]).cuda()
        targetTensor = torch.LongTensor(idx_texts).cuda()
        output, _ = model(inputTensor)
        loss = label_smoothing(output, targetTensor)
        testing_loss_sum += loss.item()

    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-epoch{epoch + 1}.pt')

        torch.save({
            'epoch': epoch + 1,
            'state': model.state_dict(),
            'testing_loss': testing_loss_sum / len(testing_data),
            'training_loss': training_loss_sum / len(training_data),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint/bert-LanGen-last.pt')

    log = f'epoch = {epoch + 1}, training_loss = {training_loss_sum / len(training_data)}, testing_loss = {testing_loss_sum / len(testing_data)}'
    training_losses.append(training_loss_sum / len(training_data))
    testing_losses.append(testing_loss_sum / len(testing_data))
    
    vis.text('<b>LOG</b><br>' + log, win='log')
    vis.line(X=list(range(len(training_losses))), Y=training_losses, win='loss', name='training_loss')
    vis.line(X=list(range(len(testing_losses))), Y=testing_losses, win='loss', update='append', name='testing_loss')
    if (epoch + 1) % SAVE_EVERY == 0 and DRAW_LEARNING_CURVE:
        vis.line(X=list(range(len(learning_curve_training))), Y=learning_curve_training, win='Learning Curve', name='learning_curve_training')
        vis.line(X=list(range(len(learning_curve_testing))), Y=learning_curve_testing, win='Learning Curve', update='append', name='learning_curve_testing')

    with open('log.txt', 'a+') as LOG:
        LOG.write(log + '\n')
    tqdm.write(log)
