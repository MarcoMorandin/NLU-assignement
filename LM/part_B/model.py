import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence

class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(emb_size, output_size, bias=False)
        self.output.weight = nn.Parameter(self.embedding.weight)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout <= 0.:
            return x

        max_batch_size = x.size(0)
        m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        return x


class LM_LSTM_VD(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VD, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.embedding_dropout = VariationalDropout(emb_dropout)

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(emb_size, output_size, bias=False)
        
        self.output_dropout = VariationalDropout(out_dropout)
        self.output.weight = nn.Parameter(self.embedding.weight)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop_emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(drop_emb)
        drop_lstm = self.out_dropout(lstm_out)
        output = self.output(drop_lstm).permute(0, 2, 1)
        return output