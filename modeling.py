import torch
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

class Discriminator(nn.Module):
    def __init__(self, pretrained_vec, hidden_size):
        super(Discriminator, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        weight = torch.FloatTensor(pretrained_vec)
        self.model.embeddings.word_embeddings = nn.Embedding.from_pretrained(embeddings=weight, freeze=True)
        self.model.encoder.layer = self.model.encoder.layer[:3]
        self.model.eval()
        self.output_layer = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        encoder_layers, _ = self.model(x)
        out = self.output_layer(encoder_layers[-1].view(-1, 768))
        out = self.sigmoid(out)
        out = torch.narrow(out, 0, 0, 1)
        return out, encoder_layers

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
