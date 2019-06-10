import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from softmax import AdaptiveDSoftmaxWithLoss
from pytorch_pretrained_bert import BertModel, BertModelNoEmbed, BertConfig
from torch.autograd import Variable
from decoder import Decoder

MODEL_PATH = 'bert-model'

class TransformerNoEmbed(nn.Module):
    def __init__(self, vocab, hidden_size, enc_num_layer, dec_num_layer):
        super(TransformerNoEmbed, self).__init__()
        self.encoder = BertModelNoEmbed(config=BertConfig(vocab_size_or_config_json_file=len(vocab), hidden_size=hidden_size, num_hidden_layers=enc_num_layer, num_attention_heads=8, intermediate_size=3072, type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1))
        self.decoder = Decoder(hidden_size=hidden_size, num_layer=dec_num_layer, heads=8)
        self.hidden_size = hidden_size
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, len(vocab), cutoffs=[1000])

    def forward(self, x, tgt):
        seq_len = tgt.size(1)
        tgt_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype('uint8')
        tgt_mask = (torch.from_numpy(tgt_mask) == 0).cuda()
        encoder_layers, _ = self.encoder(x)
        encoder_output = encoder_layers[-1]
        dec_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)
        out = self.adaptive_softmax.log_prob(dec_output.view(-1, self.hidden_size))
        return out

    def inference(self, x, vec, SOS=48, EOS=49):
        max_len = 1024
        encoder_layers, _ = self.encoder(x)
        encoder_output = encoder_layers[-1]
        tgt = [SOS]
        
        for i in tqdm(range(1500)):
            seq_len = i+1
            tgt_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype('uint8')
            tgt_mask = (torch.from_numpy(tgt_mask) == 0).cuda()
            tgt_wordvec = list(map(lambda x: vec[x], tgt))
            dec_output = self.decoder(torch.cuda.FloatTensor([tgt_wordvec]), encoder_output, tgt_mask=tgt_mask)
            predicted_output = self.adaptive_softmax.predict(dec_output.view(-1, self.hidden_size)).tolist()
            next_word = predicted_output[-1]
            tgt.append(next_word)
            if next_word == EOS: break

        return tgt

class LanGenNoEmbed(nn.Module):
    def __init__(self, vocab, hidden_size, num_layer):
        super(LanGenNoEmbed, self).__init__()
        self.model = BertModelNoEmbed(config=BertConfig(vocab_size_or_config_json_file=len(vocab), hidden_size=hidden_size, num_hidden_layers=num_layer, num_attention_heads=8, intermediate_size=3072, type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1))
        self.model.encoder.layer = self.model.encoder.layer[:3]
        self.model.eval()
        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, len(vocab), cutoffs=[994])

    def forward(self, x):
        encoder_layers, _ = self.model(x)
        out = self.adaptive_softmax.log_prob(encoder_layers[-1].view(-1, 1024))
        return out, encoder_layers

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

class BertClassify(nn.Module):
    def __init__(self, vocab, pretrained_vec, num_labels, hidden_size=768, dropout=0.1):
        super(BertClassify, self).__init__()
        self.model = BertModel.from_pretrained(MODEL_PATH)
        weight = torch.FloatTensor(pretrained_vec)
        self.model.embeddings.word_embeddings = nn.Embedding.from_pretrained(embeddings=weight, freeze=True)
        self.model.eval()
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, labels=None):
        _, pooled_output = self.model(x, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        x = self.classifier(pooled_output)
        logits = self.softmax(x)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class TextCNNClassify(nn.Module):
    def __init__(self, vocab, pretrained_vec, num_labels, hidden_size=100, dropout=0.1):
        super(TextCNNClassify, self).__init__()
        weight = torch.FloatTensor(pretrained_vec)
        channel_num = 1
        filter_num = 100
        filter_sizes = [3, 4, 5]
        embedding_dim = 768

        self.embeddings = nn.Embedding.from_pretrained(embeddings=weight, freeze=True)
        self.num_labels = num_labels
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_num, filter_num, (size, embedding_dim)) for size in filter_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, num_labels)
        self.softmax = nn.Softmax()

    def forward(self, x, labels=None):
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        logits = self.softmax(x)
        
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
