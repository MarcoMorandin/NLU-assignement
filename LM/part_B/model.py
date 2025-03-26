import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.output = nn.Linear(hidden_size, output_size)
        
        # Weight tying (same weights for embedding and output)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))  # Apply embedding dropout
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
    
class LM_LSTM_VD(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VD, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.output = nn.Linear(hidden_size, output_size)
        
        # Weight tying (same weights for embedding and output)
        self.output.weight = self.embedding.weight

        # Dropout layers
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm_dropout = nn.Dropout(out_dropout)

    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))  # Apply embedding dropout
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.lstm_dropout(lstm_out)  # Apply dropout to LSTM output
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

class AvSGD(optim.Optimizer):
    def __init__(self, params, lr=30, weight_decay=0, logging_interval=1, non_monotone_interval=5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AvSGD, self).__init__(params, defaults)
        self.logs = []
        self.T = 0
        self.logging_interval = logging_interval
        self.non_monotone_interval = non_monotone_interval

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                p.data.add_(-group['lr'], d_p)

        return loss

    def check_trigger(self, val_loss):
        """ Checks the validation loss and determines if we should start averaging """
        if self.T == 0:
            self.logs.append(val_loss)
            if len(self.logs) > self.non_monotone_interval:
                if val_loss > min(self.logs[:-self.non_monotone_interval]):
                    self.T = len(self.logs)