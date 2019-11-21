import torch
import jieba
from torch import nn
from pytorch_pretrained_bert import BertModel, BertAdam
from gensim.models.fasttext import FastText
from tqdm import tqdm
from pyltp import Postagger

model = torch.load('checkpoint-generator-pretrain/bert-LanGen-last.pt')['full_model']
MODEL_PATH = 'bert-model'
jieba.load_userdict('bert-model/dict-traditional.txt')
seq_len = 512

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

summary = '大理國無量山無量劍派的練武廳中，舉辦了五年一次的比武鬥劍大會，由無量劍的東、北、西三宗互相比試。此次是第九次大會。'
summary = list(jieba.cut(summary))
summary.insert(0, '<SOS>')
summary.append('<EOS>')
wordvec_summary = list(map(lambda x: vec[vocab[x]], summary[:512]))

inputTensor = torch.FloatTensor(wordvec_summary).cuda()
inputTensor = torch.cat((inputTensor, torch.zeros(seq_len - inputTensor.size(0), inputTensor.size(1)).cuda()))

non_masked_position = torch.ones(inputTensor.size(0)).cuda()
masked_position = torch.zeros(seq_len - inputTensor.size(0)).cuda()
attn_masked = torch.cat((non_masked_position, masked_position))

target = model.inference(inputTensor.unsqueeze(0).cuda(), attn_masked.unsqueeze(0).cuda())
target = list(map(lambda x: id2vocab[x], target.tolist()))
print(''.join(target))