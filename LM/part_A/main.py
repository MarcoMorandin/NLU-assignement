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
import logging
from dataclasses import dataclass
from itertools import product


@dataclass
class Config:
    hid_size: int = 400
    emb_size: int = 400
    lr: float = 0.0008
    clip: float = 5
    batch_size: int = 128
    eval_batch_size: int = 128
    n_epochs: int = 100
    patience: int = 3

def setup_logging():
    logger = logging.getLogger('LM')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

def get_device(logger):
    if torch.cuda.is_available():
        logger.debug("CUDA is available. Using GPU")
        return torch.device("cuda")
    logger.debug("CUDA is not available. Using CPU")
    return torch.device("cpu")

def main(point = "3",  config=Config(), logger = None):
    device = get_device(logger)

    logger.debug(f"Starting training with configuration: {config}")
    if point == "1":
        logger.info("Model Variant: LSTM")
    elif point == "2":
        logger.info("Model Variant: LSTM with Dropout")
    elif point == "3":
        logger.info("Model Variant: LSTM with Dropout and AdamW")
    else:
        logger.error("Invalid Variant!")
        return
    
    # Data loading
    cwd = os.getcwd()
    data_paths = {
        'train': f"{cwd}/dataset/PennTreeBank/ptb.train.txt",
        'dev': f"{cwd}/dataset/PennTreeBank/ptb.valid.txt",
        'test': f"{cwd}/dataset/PennTreeBank/ptb.test.txt"
    }
    
    for split, path in data_paths.items():
        if not os.path.exists(path):
            logger.error(f"{split.capitalize()} file not found at: {path}")
            return
        logger.debug(f"Found {split} data at: {path}")

    # Dataset preparation
    train_data = read_file(data_paths['train'])
    lang = Lang(train_data, ["<pad>", "<eos>"])
    
    datasets = {
        'train': PennTreeBank(train_data, lang),
        'dev': PennTreeBank(read_file(data_paths['dev']), lang),
        'test': PennTreeBank(read_file(data_paths['test']), lang)
    }

    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device)
        ),
        'dev': DataLoader(
            datasets['dev'],
            batch_size=config.eval_batch_size,
            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device)
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=config.eval_batch_size,
            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device)
        )
    }
    

    model_class = LM_LSTM if point == "1" else LM_LSTM_DROPOUT
    model = model_class(
        config.emb_size,
        config.hid_size,
        len(lang.word2id),
        pad_index=lang.word2id["<pad>"]
    )
    model.to(device)
    model.apply(init_weights)
    
    if point in ["1", "2"]:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    
    for epoch in tqdm(range(config.n_epochs), desc="Training"):
        loss = train_loop(loaders['train'], optimizer, criterion, model, config.clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(loaders['dev'], criterion, model)
            
            logger.info(f"\nEpoch {epoch}: Train Loss: {loss_dev:.4f}, Val PPL: {ppl_dev:.4f}")
            losses_dev.append(np.asarray(loss_dev).mean())

            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                config.patience = 3
            else:
                config.patience -= 1
                
            if config.patience <= 0:
                logger.info("Early stopping triggered")
                break

    best_model.to(device)
    final_ppl,  _ = eval_loop(loaders['test'], criterion, best_model)    
    logger.info(f"Final Test PPL: {final_ppl:.4f}")
    return final_ppl

if __name__ == "__main__":
    logger = setup_logging()
    main(logger=logger)