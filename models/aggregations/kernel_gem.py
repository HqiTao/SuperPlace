
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def kernel_gem(x, p=torch.ones(1)*3, eps: float = 1e-6, kernel = 2):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel = 2):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.kernel = kernel
    
    def forward(self, x):
        return kernel_gem(x, p=self.p, eps=self.eps, kernel = self.kernel)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class flatten_map(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H*W)

class flatten_channel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B, C, D = x.shape
        return x.view(B, C*D)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)
    
class KernelGeM(nn.Module):
    def __init__(self, features_dim, fc_output_dim):
        super().__init__()
        self.kernel_gem = GeM(kernel=1)
        self.flatten_map = flatten_map()

    def forward(self, x):
        x, _ = x
        x = self.kernel_gem(x)
        x = self.flatten_map(x)
        return x

class MaskedChannelGeM(nn.Module):
    def __init__(self, features_dim, kernel):
        super().__init__()
        self.kernel_gem = GeM(kernel=kernel)

        self.flatten_map = flatten_map()
        self.channel_mask = torch.zeros(features_dim, dtype=torch.bool)
        self.masked_channel = self.load_masked_channel()
        self.channel_mask[self.masked_channel] = True

        self.flatten_channel = flatten_channel()
        self.flatten_features_dim = int((16 / kernel) * (16 / kernel) * (len(self.masked_channel)))

        self.fc = nn.Linear(self.flatten_features_dim, self.flatten_features_dim)
        self.norm = L2Norm()

    def load_masked_channel(self):
        masked_channel = np.load('indices_of_smallest.npy')
        return masked_channel[:2]

    def forward(self, x):
        x, _ = x
        x = self.kernel_gem(x)
        x = self.flatten_map(x[:, self.channel_mask, :, :])
        x = self.flatten_channel(x)
        x = self.fc(x)
        x = self.norm(x)
        return x