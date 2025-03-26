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
from dataclasses import dataclass
from itertools import product
import json

@dataclass
class Config:
    hid_size: int = 500
    emb_size: int = 500
    lr: float = 3
    clip: float = 5
    batch_size: int = 128
    eval_batch_size: int = 128
    n_epochs: int = 100
    patience: int = 3


def main(point = "3",  config=Config(), logger = None, report_path = './report'):
    device = get_device(logger)
    
    logger.debug(f"Starting training with configuration: {config}")
    if point == "1":
        logger.info("Model Variant: LSTM with Weight Tying")
    elif point == "2":
        logger.info("Model Variant: LSTM with Weight Tying and Variational Dropout")
    elif point == "3":
        logger.info("Model Variant: LSTM with Weight Tying and Variational Dropout and AvSGD")
    else:
        logger.error("Invalid Variant!")
        return
    
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
    
    model_class = LM_LSTM_WT if point == "1" else LM_LSTM_VD
    
    model = model_class(
        config.emb_size,
        config.hid_size,
        len(lang.word2id),
        pad_index=lang.word2id["<pad>"]
    )
    
    model.to(device)
    model.apply(init_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
        
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppl_dev_list = []
    best_ppl = math.inf
    best_model = None
    
    for epoch in tqdm(range(config.n_epochs), desc="Training"):
        loss = train_loop(loaders['train'], optimizer, criterion_train, model, config.clip)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            
            ppl_dev, loss_dev = eval_loop(loaders['dev'], criterion_eval, model)
            ppl_dev_list.append(ppl_dev)
            
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
    final_ppl,  _ = eval_loop(loaders['test'], criterion_eval, best_model)    
    logger.info(f"Final Test PPL: {final_ppl:.4f}")
    
    report(best_model, sampled_epochs, losses_train, losses_dev, ppl_dev_list, final_ppl, point, config, report_path)


if __name__ == "__main__":
    import argparse
    logger = setup_logging()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('point', type=str,
                        help=f"Point to be run. Choose from:\n 1 - LSTM, 2 - LSTM with Dropout, 3 - LSTM with Dropout and AdamW",
                        default="3")
    
    parser.add_argument('path', type=str,
                        help="Path to save the plots and report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    
    main(point=args.point, logger=logger, report_path=args.path)
