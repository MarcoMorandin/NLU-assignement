import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
import os
import json

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)     
                    

def generate_plot(epochs, data, labels, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for d, label in zip(data, labels):
        plt.plot(epochs, d, label=label, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def report(best_model, epochs, loss_train, loss_dev, perplexity_list, final_ppl, point, Config, path):
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