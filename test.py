import modeling
import jieba
import torch
import random
from tqdm import tqdm
from gensim.models.fasttext import FastText

text = '大理國無量山無量劍派的練武廳中，舉辦了五年一次的比武鬥劍大會，由無量劍的東、北、西三宗互相比試。此次是第九次大會。'
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

model = modeling.TransformerNoEmbed(vocab=vocab, hidden_size=1024, enc_num_layer=3, dec_num_layer=6)
model.eval()
model.cuda()
checkpoint = torch.load('checkpoint-generator-pretrain/bert-LanGen-last.pt')
model.load_state_dict(checkpoint['state'])
print('Info of model:')
print(f'Epoch: {checkpoint["epoch"]}')
print(f'Training Loss: {checkpoint["training_loss"]}')
print(f'Testing Loss: {checkpoint["testing_loss"]}')

summary = text
summary = list(jieba.cut(summary))
summary.insert(0, '<SOS>')
summary.append('<EOS>')
wordvec_summaries = list(map(lambda x: vec[vocab[x]], summary))
target = model.inference(torch.FloatTensor(wordvec_summaries).unsqueeze(0).cuda(), vec=vec)
result = ''.join(list(map(lambda x: id2vocab[x], target)))
print(result)