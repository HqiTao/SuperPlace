import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) if max_val - min_val else 0 for x in data]

def domain_awareness(args, model, train_dl, optimizer, scaler, scheduler, miner, criterion):

    gradients = {}
    wg = {}
    weight = {}

    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name and "aggregation" not in name:
            layer_name = '.'.join(name.split('.')[4:-1])
            gradients[layer_name] = 0
            wg[layer_name] = 0
            weight[layer_name] = 0

    epoch_losses=[]

    model.train()
    for places, labels in tqdm(train_dl, ncols=100):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            features = model(images)
            miner_outputs = miner(features, labels)
            loss = criterion(features, labels, miner_outputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, features, miner_outputs, images, labels

        for name, param in model.named_parameters():
            if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name and "aggregation" not in name:
                layer_name = '.'.join(name.split('.')[4:-1])
                weight[layer_name] += param.abs().detach()
                gradients[layer_name] += param.grad.abs().detach()
                wg[layer_name] += (param.grad / param).abs().detach()

    for name in gradients:
        gradients[name] /= len(train_dl)
        weight[name] /= len(train_dl)
        wg[name] /= len(train_dl)

    param_names = [name for name in gradients]
    avg_gradients = [gradients[name].mean().item() for name in gradients]
    avg_weight = [weight[name].mean().item() for name in weight]
    avg_wg = [wg[name].mean().item() for name in wg]

    norm_avg_gradients = normalize(avg_gradients)
    norm_avg_weight = normalize(avg_weight)
    norm_avg_wg = normalize(avg_wg)

    plt.figure(figsize=(24, 10))
    plt.plot(param_names, norm_avg_gradients, marker='o', color='blue', label='Average Gradient')
    plt.plot(param_names, norm_avg_weight, marker='^', color='green', label='Average Weight')
    plt.plot(param_names, norm_avg_wg, marker='s', color='red', label='Average Gradient/Weight')

    plt.xticks(rotation=90)
    plt.xlabel('Parameter Name')
    plt.ylabel('Value')
    plt.title('Importance of Network Parameters')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "demo.png"))

