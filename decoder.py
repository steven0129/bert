import torch
import math
from torch import nn

class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, max_seq_len=512):
        super(PositionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        pe = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i in range(0, hidden_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/hidden_size)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/hidden_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_size)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].cuda()
        return x