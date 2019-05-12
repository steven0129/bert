import torch
import torch.nn.functional as F
import math
import copy
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

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.hidden_size)
        return self.out(concat)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)
            
        if dropout is not None:
            scores = dropout(scores)
                
        output = torch.matmul(scores, v)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__() 
        self.linear_1 = nn.Linear(hidden_size, d_ff)
        self.linear_2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.attn1 = MultiHeadAttention(heads, hidden_size)
        self.attn2 = MultiHeadAttention(heads, hidden_size)
        self.ff = FeedForward(hidden_size).cuda()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        tmp = self.norm1(x)
        x = x + self.dropout1(self.attn1(tmp, tmp, tmp, tgt_mask))
        tmp = self.norm2(x)
        x = x + self.dropout2(self.attn2(tmp, enc_output, enc_output, src_mask))
        tmp = self.norm3(x)
        x = x + self.dropout3(self.ff(tmp))
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layer, heads):
        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.pe = PositionalEncoder(hidden_size)
        self.layers = self.get_clones(DecoderLayer(hidden_size, heads), num_layer)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.pe(x)
        for i in range(self.num_layer):
            x = self.layers[i](x, enc_output, src_mask, tgt_mask)

        return self.norm(x)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])