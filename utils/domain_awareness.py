import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

def domain_awareness(args, model, train_dl, optimizer, scaler, scheduler, miner, criterion):

    gradients = {}
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

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in gradients:
                    gradients[name] = param.grad.abs().clone()
                else:
                    gradients[name] += param.grad.abs().clone()

        del loss, features, miner_outputs, images, labels
    

    for name in gradients:
        gradients[name] /= len(train_dl)

    sorted_gradients = sorted(gradients.items(), key=lambda x: x[1].mean(), reverse=True)

    param_names = [name for name, _ in sorted_gradients]
    avg_gradients = [grad.mean().item() for _, grad in sorted_gradients]

    plt.figure(figsize=(10, 8))
    plt.plot(param_names, avg_gradients, marker='o')
    plt.xticks(rotation=90)
    plt.xlabel('Parameter Name')
    plt.ylabel('Average Gradient')
    plt.title('Importance of Network Parameters based on Gradient Magnitude')
    plt.tight_layout()
    plt.show()

