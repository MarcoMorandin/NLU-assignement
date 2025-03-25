from functions import *
from utils import *
from model import *

from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

#TODO: implement logger
def main(point = "3"):
    #TODO: better organize params
    hid_size = 400
    emb_size = 400
    lr = 0.0001
    clip = 5
    
    cwd = os.getcwd()
    train_file = cwd + "/dataset/PennTreeBank/ptb.train.txt"
    dev_file = cwd + "/dataset/PennTreeBank/ptb.valid.txt"
    test_file = cwd + "/dataset/PennTreeBank/ptb.test.txt"

    if not os.path.exists(train_file) or not os.path.exists(dev_file) or not os.path.exists(test_file):
        print("Files not found") #error
        return
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}") #info
    
    train_data = read_file(train_file)
    dev_data = read_file(dev_file)
    test_data = read_file(test_file)
    
    lang = Lang(train_data, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    
    train_dataset = PennTreeBank(train_data, lang)
    dev_dataset = PennTreeBank(dev_data, lang)
    test_dataset = PennTreeBank(test_data, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))

    if point == "1":
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    else:
        model = LM_LSTM_DROPOUT(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    
    model.apply(init_weights)
    
    if point == "1" or point == "2":
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr)
        
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:
                break

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)

if __name__ == "__main__":
    main()