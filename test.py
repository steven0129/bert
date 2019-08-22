import modeling
import jieba
import torch
import csv
import random
from tqdm import tqdm
from gensim.models.fasttext import FastText
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pyltp import Postagger

jieba.load_userdict('bert-model/dict-traditional.txt')
jieba.suggest_freq('<newline>', True)

# Load vocabularies
print('建構詞向量...')
word2vec = FastText.load_fasttext_format('bert-model/wordvec-large.dim1024')
vocab = {}
id2vocab = {}
vec = []

with open('bert-model/TF.csv') as TF:
    for idx, line in enumerate(tqdm(TF)):
        term = line.split(',')[0]
        vocab[term] = idx
        id2vocab[idx] = term
        vec.append(word2vec[term])

del word2vec

POS = Postagger()
POS.load('bert-model/ltp_data_v3.4.0/pos.model')

# Tokenized input
print('Tokenization...')
data = []
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
        
        data.append((texts[:1500], summaries[:1500]))

random.Random(0).shuffle(data)
training_data = data[:round(len(data) * 0.8)]
testing_data = data[round(len(data) * 0.8):]

model = modeling.BertNoEmbed(vocab=vocab, hidden_size=1024, enc_num_layer=3)
model.eval()
model.cuda()
smoother = SmoothingFunction()
checkpoint = torch.load('checkpoint/bert-LanGen-last.pt')
model.load_state_dict(checkpoint['state'])
print('Info of model:')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Training Loss: {checkpoint["training_loss"]}')
#print(f'Testing Loss: {checkpoint["testing_loss"]}')

def eval_bleu(summaries, ans, mode='normal', random_seed=0, smoother=smoother.method1):
    summary = summaries
    seq_len = len(ans)
    if mode == 'random':
        random.Random(random_seed).shuffle(summary)
    elif mode == 'random_partially':
        pos = list(POS.postag(summary))
        pos_set = list(set(POS.postag(summary)))
        for p in pos_set:
            order_by_pos = [(i, x) for i, x in enumerate(summary) if pos[i]==p]
            [idxs, words] = zip(*order_by_pos)
            idxs = list(idxs)
            words = list(words)
            random.Random(0).shuffle(idxs)
            for i, idx in enumerate(idxs):
                summary[idx] = words[i]

    wordvec_summaries = list(map(lambda x: vec[vocab[x]], summary))
    inputTensor = torch.FloatTensor(wordvec_summaries).cuda()
    inputTensor = torch.cat((inputTensor, torch.zeros(seq_len - inputTensor.size(0), inputTensor.size(1)).cuda()))

    non_masked_position = torch.ones(inputTensor.size(0)).cuda()
    masked_position = torch.zeros(seq_len - inputTensor.size(0)).cuda()
    attn_masked = torch.cat((non_masked_position, masked_position))

    target = model.inference(inputTensor.unsqueeze(0).cuda(), attn_masked.unsqueeze(0).cuda())
    target = list(map(lambda x: id2vocab[x], target.tolist()))
    if '<EOS>' in target:
        EOS_pos = target.index('<EOS>')
        return sentence_bleu([ans], target[:EOS_pos + 1], smoothing_function=smoother)
    else:
        return sentence_bleu([ans], target, smoothing_function=smoother)

# evaluation
with open('eval_train.csv', 'w') as FILE:
    writer = csv.writer(FILE)
    writer.writerow(['normal', 'random_partially', 'random'])
    for texts, summaries in training_data:
        normal_bleu = eval_bleu(summaries, texts)
        random_partial_bleu = eval_bleu(summaries, texts, mode='random_partially')
        random_bleu = eval_bleu(summaries, texts, mode='random')
        writer.writerow([normal_bleu, random_partial_bleu, random_bleu])

with open('eval_test.csv', 'w') as FILE:
    writer = csv.writer(FILE)
    writer.writerow(['normal', 'random_partially', 'random'])
    for texts, summaries in testing_data:
        normal_bleu = eval_bleu(summaries, texts)
        random_partial_bleu = eval_bleu(summaries, texts, mode='random_partially')
        random_bleu = eval_bleu(summaries, texts, mode='random')
        writer.writerow([normal_bleu, random_partial_bleu, random_bleu])
