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

class TextCNN(nn.Module):
    def __init__(self, embedding_dimension):
        super(TextCNN, self).__init__()
        self.conv3 = nn.Conv2d(1, 1, (3, embedding_dimension))
        self.conv4 = nn.Conv2d(1, 1, (4, embedding_dimension))
        self.conv5 = nn.Conv2d(1, 1, (5, embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((512-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((512-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((512-5+1, 1))
        self.linear1 = nn.Linear(3, 2)

    def forward(self, x):
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, 2)

        return x

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