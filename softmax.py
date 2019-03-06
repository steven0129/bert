import torch
from torch import nn
from loss import NLLLoss
from torch.autograd import Variable

class AdaptiveDSoftmaxWithLoss(nn.Module):
    def __init__(self, hidden_size, partition_dim, cutoffs=13867, vocab_num=1289627):
        super(AdaptiveDSoftmaxWithLoss, self).__init__()
        self.partition1 = nn.Linear(partition_dim, cutoffs + 1)
        self.partition2 = nn.Linear(hidden_size - partition_dim, vocab_num)
        self.nll_loss = NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        self.cutoffs = cutoffs
        self.partition_dim = partition_dim

    def forward(self, x, target):
        x = Variable(x.float(), requires_grad=True)
        x = torch.index_select(x, 1, torch.cuda.LongTensor(list(range(self.partition_dim))))
        x = self.partition1(x)
        x = self.softmax(x)
        yt = self.one_hot(target, self.cutoffs).cuda()
        loss = self.nll_loss(x, yt)
        return loss

    def one_hot(self, x, max_index):
        max_vals = torch.cuda.LongTensor([max_index] * x.size(0))
        x = torch.where(x <= max_index, x, max_vals)
        return torch.zeros(x.size(0), max_index + 1).cuda().scatter_(1, x.view(-1, 1), 1.0)