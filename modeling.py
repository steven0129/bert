import torch
import torch.nn.functional as F
from torch import nn
from softmax import AdaptiveDSoftmaxWithLoss
from pytorch_pretrained_bert import BertModel
from torch.autograd import Variable

MODEL_PATH = 'bert-model'

class LanGen(nn.Module):
    def __init__(self, vocab, pretrained_vec, hidden_size):
        super(LanGen, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        weight = torch.FloatTensor(pretrained_vec)
        self.model.embeddings.word_embeddings = nn.Embedding.from_pretrained(embeddings=weight, freeze=True)
        self.model.encoder.layer = self.model.encoder.layer[:3]
        self.model.eval()
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, len(vocab), cutoffs=[994])
    
    def forward(self, x):
        encoder_layers, _ = self.model(x)
        out = self.adaptive_softmax.log_prob(encoder_layers[-1].view(-1, 768))
        return out, encoder_layers

class LanClassify(nn.Module):
    def __init__(self, vocab, pretrained_vec, hidden_size, num_labels, dropout=0.1):
        super(LanClassify, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        weight = torch.FloatTensor(pretrained_vec)
        self.model.embeddings.word_embeddings = nn.Embedding.from_pretrained(embeddings=weight, freeze=True)
        self.model.eval()
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, labels=None):
        _, pooled_output = self.model(x, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)