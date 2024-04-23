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

def task_awareness(args, model, train_dl, optimizer, scaler, scheduler, miner, criterion):

    gradients = {}
    w_mul_g = {}
    weights = {}

    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name and "aggregation" not in name:
            layer_name = '.'.join(name.split('.')[4:-1])
            gradients[layer_name] = 0
            w_mul_g[layer_name] = 0
            weights[layer_name] = 0

    epoch_losses=[]

    C = 768
    cumulative_channel_losses = torch.zeros(C)

    model.eval()
    for places, labels in tqdm(train_dl, ncols=100):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        
        with torch.cuda.amp.autocast():

            features = model(images) # features [B, C, D]
            channel_losses = torch.zeros(features.size(1))
            for i in range((features.size(1))):
                miner_outputs = miner(features[:, i, :], labels)
                loss = criterion(features[:, i, :], labels, miner_outputs)
                channel_losses[i] = loss.item()

        cumulative_channel_losses += channel_losses

        del channel_losses, features, miner_outputs, images, labels

    norm_cumulative_channel_losses = normalize(cumulative_channel_losses.numpy())
    plt.plot(norm_cumulative_channel_losses)
    plt.title("Cumulative Loss per Channel")
    plt.xlabel("Channel")
    plt.ylabel("Cumulative Loss")
    plt.savefig("channel_importance.png")

    indices_of_smallest = sorted(range(len(norm_cumulative_channel_losses)), key=lambda x: norm_cumulative_channel_losses[x])[:2]

    print("Indices of the 20 smallest normalized values:", indices_of_smallest)
    np.save('indices_of_smallest.npy', indices_of_smallest)
