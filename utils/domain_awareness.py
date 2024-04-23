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
    w_mul_g = {}
    weights = {}

    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name and "aggregation" not in name:
            layer_name = '.'.join(name.split('.')[4:-1])
            gradients[layer_name] = 0
            w_mul_g[layer_name] = 0
            weights[layer_name] = 0

    epoch_losses=[]

    model.eval()
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

        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, features, miner_outputs, images, labels

        for name, param in model.named_parameters():
            if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name and "aggregation" not in name:
                layer_name = '.'.join(name.split('.')[4:-1])
                weights[layer_name] += param.data.abs().detach()
                gradients[layer_name] += param.grad.abs().detach()
                w_mul_g[layer_name] += (torch.mul(param.grad, param.data)).abs().detach()

    param_names = [name for name in gradients]
    avg_gradients = [gradients[name].mean().item() for name in gradients]
    avg_weights = [weights[name].mean().item() for name in weights]
    avg_w_mul_g = [w_mul_g[name].mean().item() for name in w_mul_g]

    norm_avg_gradients = normalize(avg_gradients)
    norm_avg_weights = normalize(avg_weights)
    norm_avg_w_mul_g = normalize(avg_w_mul_g)

    data_to_save = {"param_names": param_names, "norm_avg_gradients": norm_avg_gradients}
    np.save(os.path.join("logs", args.backbone + "_" + args.aggregation, "gsv_cities", "norm_avg_gradients.npy"), data_to_save)

    plt.figure(figsize=(24, 10))
    plt.plot(param_names, norm_avg_gradients, marker='o', color='blue', label='Average Gradient')
    plt.plot(param_names, norm_avg_weights, marker='^', color='green', label='Average Weight')
    plt.plot(param_names, norm_avg_w_mul_g, marker='s', color='red', label='Average Weight mul Gradient')

    plt.xticks(rotation=90)
    plt.xlabel('Parameter Name')
    plt.ylabel('Value')
    plt.title('Importance of Network Parameters')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "awareness.png"))

