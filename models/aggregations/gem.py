
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)
    
class CosGeM(nn.Module):
    def __init__(self, features_dim, fc_output_dim):
        super().__init__()
        self.norm1 = L2Norm()
        self.gem = GeM()
        self.flatten = Flatten()
        self.fc = nn.Linear(features_dim, fc_output_dim)
        self.norm2 = L2Norm()

    def forward(self, x):
        x, _ = x
        x = self.norm1(x)
        x = self.gem(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm2(x)
        return x


class CAGeM(nn.Module):
    def __init__(self, features_dim, channels_num):
        super().__init__()

        self.channel_attention = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(features_dim, channels_num),
            nn.GELU(),
            nn.Linear(channels_num, features_dim),
            nn.Sigmoid())

        self.gem = GeM()
        self.fc = nn.Linear(features_dim, features_dim)
        self.norm = L2Norm()

    def forward(self, x):
        x_in, _ = x
        x_atte = self.channel_attention(x_in)
        x_feat = self.gem(x_in)

        x_atte = x_atte.unsqueeze(-1).unsqueeze(-1)
        x = x_atte * x_feat

        x = self.fc(x.flatten(1))
        x = self.norm(x)
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 768, 16, 16), 1
    agg = CAGeM(features_dim = 768, channels_num = 4)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == '__main__':
    main()