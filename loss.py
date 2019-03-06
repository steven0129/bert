import torch
from torch import nn
from torch.autograd import Variable

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, yp, yt):
        term1 = yt * torch.log(yp)
        term2 = (torch.ones_like(yt) - yt) * torch.log(torch.ones_like(yp) - torch.log(yp))
        return torch.mean(term1 + term2, dim=1)