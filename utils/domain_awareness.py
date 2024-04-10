import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from models import dinov2_network

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
    dim = dinov2_network.CHANNELS_NUM[args.backbone]

    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name:
            if "aggregation" not in name:
                layer_name = '.'.join(name.split('.')[4:-1])
                if "qkv" in layer_name:
                    gradients[layer_name+"_q"] = 0
                    gradients[layer_name+"_k"] = 0
                    gradients[layer_name+"_v"] = 0
                    wg[layer_name+"_q"] = 0
                    wg[layer_name+"_k"] = 0
                    wg[layer_name+"_v"] = 0
                    weight[layer_name+"_q"] = 0
                    weight[layer_name+"_k"] = 0
                    weight[layer_name+"_v"] = 0
                else:
                    gradients[layer_name] = 0
                    wg[layer_name] = 0
                    weight[layer_name] = 0
            else:
                layer_name = '.'.join(name.split('.')[:-1])
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
            if param.requires_grad and "bias" not in name and "norm" not in name and "ls" not in name:
                if "aggregation" not in name:
                    layer_name = '.'.join(name.split('.')[4:-1])
                    if "qkv" in name:
                        q_weight = param[:dim, :].abs().detach()
                        k_weight = param[dim:2*dim, :].abs().detach()
                        v_weight = param[2*dim:, :].abs().detach()

                        q_grad = param.grad[:dim, :].abs().detach()
                        k_grad = param.grad[dim:2*dim, :].abs().detach()
                        v_grad = param.grad[2*dim:, :].abs().detach()

                        weight[layer_name+"_q"] += q_weight
                        weight[layer_name+"_k"] += k_weight
                        weight[layer_name+"_v"] += v_weight

                        gradients[layer_name+"_q"] += q_grad
                        gradients[layer_name+"_k"] += k_grad
                        gradients[layer_name+"_v"] += v_grad

                        wg[layer_name+"_q"] += (q_grad / q_weight).abs().detach()
                        wg[layer_name+"_k"] += (k_grad / k_weight).abs().detach()
                        wg[layer_name+"_v"] += (v_grad / v_weight).abs().detach()

                    else:
                        weight[layer_name] += param.abs().detach()
                        gradients[layer_name] += param.grad.abs().detach()
                        wg[layer_name] += (param.grad / param).abs().detach()
                else:
                    layer_name = '.'.join(name.split('.')[:-1])
                    weight[layer_name] += param.abs().detach()
                    gradients[layer_name] += param.grad.abs().detach()
                    wg[layer_name] += (param.grad * param).abs().detach()

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

