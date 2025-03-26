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
import json


@dataclass
class Config:
    hid_size: int = 300
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

def report(best_model, epochs, loss_train, loss_dev, perplexity_list, final_ppl, point, path):
    generate_plot(
        epochs=epochs,
        data=[loss_train, loss_dev],
        labels=['Training Loss', 'Validation Loss'],
        title='Training and Validation Loss',
        xlabel='Epochs',
        ylabel='Loss',
        filename=os.path.join(path, 'loss_plot.png')
    )

    generate_plot(
        epochs=epochs,
        data=[perplexity_list],
        labels=['Validation Perplexity'],
        title='Validation Perplexity',
        xlabel='Epochs',
        ylabel='Perplexity',
        filename=os.path.join(path, 'ppl_plot.png')
    )
    
    report_data = {
        "number_epochs": len(epochs),
        "lr": Config.lr,
        "hidden_size": Config.hid_size,
        "emb_size": Config.emb_size,
        "clip": Config.clip,
        "batch_size": Config.batch_size,
        "eval_batch_size": Config.eval_batch_size,
        "point": point,
        "final_ppl": final_ppl
    }
    
    with open(os.path.join(path, 'report.json'), "w") as file:
        json.dump(report_data, file, indent=4)
    
    torch.save(best_model.state_dict(), os.path.join(path, "model.pt"))


def main(point = "3",  config=Config(), logger = None, report_path = './report'):
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
    
    report(best_model, sampled_epochs, losses_train, losses_dev, ppl_dev_list, final_ppl, point, report_path)

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