import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
# class LM_LSTM_WT(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
#         super(LM_LSTM_WT, self).__init__()
#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         # Ensure the last LSTM layer outputs `emb_size` (not `hidden_size`)
#         self.lstm = nn.LSTM(emb_size, emb_size, n_layers, bidirectional=False, batch_first=True)
#         self.output = nn.Linear(emb_size, output_size)
        
#         # Weight tying: output.weight = embedding.weight^T (requires transpose)
#         self.output.weight = nn.Parameter(self.embedding.weight)  # Proper weight tying

#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)
#         lstm_out, _ = self.lstm(emb)
#         output = self.output(lstm_out).permute(0, 2, 1)
#         return output
    
class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_WT, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # LSTM layers
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # Projection to match embedding size (if hidden_size != emb_size)
        self.proj = nn.Linear(hidden_size, emb_size) if hidden_size != emb_size else nn.Identity()
        
        # Output layer (weight tying)
        self.output = nn.Linear(emb_size, output_size, bias=False)  # Disable bias to match paper
        self.output.weight = nn.Parameter(self.embedding.weight)  # Proper weight tying

    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, seq_len)
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)  # (batch_size, seq_len, hidden_size)
        proj_out = self.proj(lstm_out)  # (batch_size, seq_len, emb_size)
        output = self.output(self.out_dropout(proj_out))  # (batch_size, seq_len, output_size)
        return output.permute(0, 2, 1)  # (batch_size, output_size, seq_len) for CrossEntropyLoss

class LM_LSTM_VD(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_VD, self).__init__()
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