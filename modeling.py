import torch
from torch import nn
from softmax import AdaptiveDSoftmaxWithLoss
from pytorch_pretrained_bert import BertModel
from torch.autograd import Variable

MODEL_PATH = 'bert-model'

class LanGen(nn.Module):
    def __init__(self, hidden_size=300, max_len=512):
        super(LanGen, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        self.model.embeddings.word_embeddings = nn.DataParallel(self.model.embeddings.word_embeddings)
        self.model.embeddings.position_embeddings = nn.DataParallel(self.model.embeddings.position_embeddings)
        self.model.embeddings.token_type_embeddings = nn.DataParallel(self.model.embeddings.token_type_embeddings)
        self.model.eval()
        self.hidden_size = hidden_size
        self.adaptive_d_softmax = nn.DataParallel(AdaptiveDSoftmaxWithLoss(hidden_size=768, partition_dim=750))
        self.cuda()
    
    def forward(self, x, target):
        encoder_layers, _ = self.model(x)
        return self.adaptive_d_softmax(encoder_layers[-1].view(-1, self.hidden_size), target)