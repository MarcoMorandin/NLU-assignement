import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import os
import copy
import numpy as np
import math
import logging
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass
from functions import *
from utils import *
from model import *

@dataclass
class Config:
    hid_size: int = 200
    emb_size: int = 300
    lr: float = 0.0001
    clip: float = 5
    batch_size: int = 64
    eval_batch_size: int = 128
    n_epochs: int = 100
    patience: int = 3

def setup_logging():
    logger = logging.getLogger('LM')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_distributed(rank, world_size):
    """ Initializes PyTorch's distributed backend. """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size, point="3", config=Config()):
    setup_distributed(rank, world_size)  # Initialize Distributed Training
    logger = setup_logging()
    device = torch.device(f"cuda:{rank}")

    logger.debug(f"Using GPU {rank} out of {world_size}")
    
    # Load Data
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

    # Load Dataset
    train_data = read_file(data_paths['train'])
    lang = Lang(train_data, ["<pad>", "<eos>"])
    
    datasets = {
        'train': PennTreeBank(train_data, lang),
        'dev': PennTreeBank(read_file(data_paths['dev']), lang),
        'test': PennTreeBank(read_file(data_paths['test']), lang)
    }

    # **Use DistributedSampler for multi-GPU training**
    train_sampler = DistributedSampler(datasets['train'], num_replicas=world_size, rank=rank)
    
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=config.batch_size,
            sampler=train_sampler,
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

    # **Model Selection**
    model_class = LM_LSTM if point == "1" else LM_LSTM_DROPOUT
    model = model_class(
        config.emb_size,
        config.hid_size,
        len(lang.word2id),
        pad_index=lang.word2id["<pad>"]
    ).to(device)

    model.apply(init_weights)

    # **Use DistributedDataParallel**
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # **Optimizer Selection**
    optimizer = optim.AdamW(model.parameters(), lr=config.lr) if point == "3" else optim.SGD(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])

    # Training Loop
    patience = config.patience
    best_ppl = math.inf
    best_model = None

    for epoch in range(config.n_epochs):
        loaders['train'].sampler.set_epoch(epoch)  # Shuffle dataset across GPUs
        loss = train_loop(loaders['train'], optimizer, criterion, model, config.clip)

        if rank == 0:  # Only rank 0 logs and saves best model
            ppl_dev, loss_dev = eval_loop(loaders['dev'], criterion, model)
            logger.info(f"Epoch {epoch}: Train Loss: {loss_dev:.4f}, Val PPL: {ppl_dev:.4f}")

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = config.patience
            else:
                patience -= 1

            if patience <= 0:
                logger.info("Early stopping triggered")
                break

    if rank == 0:
        best_model.to(device)
        final_ppl, _ = eval_loop(loaders['test'], criterion, best_model)
        logger.info(f"Final Test PPL: {final_ppl:.4f}")

    dist.destroy_process_group()  # Cleanup distributed training

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)