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
        logger.debug("Model Variant: LSTM")
    elif point == "2":
        logger.debug("Model Variant: LSTM with Dropout")
    elif point == "3":
        logger.debug("Model Variant: LSTM with Dropout and AdamW")
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
    if(torch.cuda.device_count() > 1):
        logger.debug(f"Using {torch.cuda.device_count()} GPUs")
        model= nn.DataParallel(model)
    model.to(device)
    model.apply(init_weights)
    
    if point in ["1", "2"]:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    
    
    patience = 3
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
            
            logger.debug(f"Epoch {epoch}: Train Loss: {loss_dev:.4f}, Val PPL: {ppl_dev:.4f}")
            losses_dev.append(np.asarray(loss_dev).mean())

            if  ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0:
                logger.debug("Early stopping triggered")
                break

    best_model.to(device)
    final_ppl,  _ = eval_loop(loaders['test'], criterion, best_model)    
    logger.debug(f"Final Test PPL: {final_ppl:.4f}")
    return final_ppl

def random_search_phase(n_trials, logger):
    """Phase 1: Initial exploration with random search"""
    logger.info("Starting Phase 1: Random Search")
    param_dist = {
        'lr': lambda: 10 ** np.random.uniform(-4, -1),
        'hid_size': lambda: np.random.randint(100, 800),
        'emb_size': lambda: np.random.randint(100, 800),
        'batch_size': lambda: np.random.choice([8, 16, 32, 64])
    }
    
    results = []
    for trial in range(n_trials):
        config = Config(**{k: v() for k, v in param_dist.items()}, n_epochs=20)
        logger.info(f"Random Search Trial {trial + 1}/{n_trials}: {config}")
        test_ppl = main(point="3", config=config, logger=logger)
        results.append((config, test_ppl))
        logger.info(f"Trial {trial + 1} completed with Test PPL: {test_ppl:.4f}")
    
    results.sort(key=lambda x: x[1])
    best_config, best_ppl = results[0]
    logger.info(f"Random Search Best Config: {best_config}, PPL: {best_ppl:.4f}")
    return best_config, results

def grid_search_phase(best_config, logger):
    """Phase 2: Refinement with grid search"""
    logger.info("Starting Phase 2: Grid Search")
    param_grid = {
        'lr': [best_config.lr * f for f in [0.5, 1.0, 2.0]],
        'hid_size': [max(100, best_config.hid_size - 100), best_config.hid_size,
                    min(800, best_config.hid_size + 100)],
        'emb_size': [max(100, best_config.emb_size - 100), best_config.emb_size,
                    min(800, best_config.emb_size + 100)],
        'batch_size': [max(8, best_config.batch_size - 8), best_config.batch_size,
                      min(64, best_config.batch_size + 8)]
    }
    
    best_ppl = float('inf')
    best_refined_config = None
    
    for params in product(*param_grid.values()):
        config = Config(**dict(zip(param_grid.keys(), params)), n_epochs=50)
        logger.info(f"Grid Search Trial: {config}")
        test_ppl = main(point="3", config=config, logger=logger)
        logger.info(f"Grid Search Trial completed with Test PPL: {test_ppl:.4f}")
        
        if test_ppl < best_ppl:
            best_ppl = test_ppl
            best_refined_config = config
            logger.info(f"New best grid config: {best_refined_config}, PPL: {best_ppl:.4f}")
    
    logger.info(f"Grid Search Best Config: {best_refined_config}, PPL: {best_ppl:.4f}")
    return best_refined_config

def validation_phase(best_config, logger):
    """Phase 3: Final validation"""
    logger.info("Starting Phase 3: Final Validation")
    config = Config(
        lr=best_config.lr,
        hid_size=best_config.hid_size,
        emb_size=best_config.emb_size,
        batch_size=best_config.batch_size,
        n_epochs=100
    )
    logger.info(f"Validating with final config: {config}")
    final_ppl = main(point="3", config=config, logger=logger)
    logger.info(f"Final Validation Test PPL: {final_ppl:.4f}")
    return config, final_ppl

def practical_workflow():
    logger = setup_logging()
    logger.info("Starting Practical Workflow for Hyperparameter Tuning")
    
    # Phase 1: Random Search (10 trials)
    best_random_config, random_results = random_search_phase(n_trials=10, logger=logger)
    
    # Phase 2: Grid Search around best random config
    best_grid_config = grid_search_phase(best_random_config, logger)
    
    # Phase 3: Final Validation
    final_config, final_ppl = validation_phase(best_grid_config, logger)
    
    logger.info("Workflow Complete")
    logger.info(f"Final Best Configuration: {final_config}")
    logger.info(f"Final Test PPL: {final_ppl:.4f}")
    
    return final_config, final_ppl

if __name__ == "__main__":
    final_config, final_ppl = practical_workflow()