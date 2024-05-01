
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6, k = 1):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2)//k, x.size(-1)//k)).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, k=1):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.k = k
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, k = self.k)
    
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
        # x = self.norm1(x)
        x = self.gem(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.norm2(x)
        return x


class MixedGeM(nn.Module):
    def __init__(self, num_channels = 768, num_hiddens = None, num_clusters = None, dim_clusters = 1):
        super().__init__()

        self.num_channels = num_channels
        self.num_hiddens = num_channels // 2 if num_hiddens == None else num_hiddens
        self.num_clusters = num_clusters
        self.dim_clusters = dim_clusters
        self.dim_features = self.num_clusters * self.dim_clusters * self.dim_clusters

        # self.feat_cluster = nn.Sequential(
        #     nn.Conv2d(self.num_channels, self.num_hiddens, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_hiddens, self.num_clusters, 1)
        # )

        self.gem = GeM(k=self.dim_clusters)

        self.feat_mlp = nn.Sequential(
            nn.Linear(self.dim_features, self.dim_features),
            L2Norm())

        self.cls_mlp = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels//2),
            L2Norm())

        self.norm = L2Norm()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, x):
        x_feat, x_cls = x

        x_feat = self.gem(x_feat)
        x_feat = self.feat_mlp(x_feat.flatten(1))
        x_cls = self.cls_mlp(x_cls)
        x_feat = self.norm(torch.cat([x_cls, x_feat], dim=-1))
        return x_feat


class CAGeM(nn.Module):
    # Low-rank, GeM, and GELU
    def __init__(self, num_channels, num_hiddens):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_hiddens, bias = False),
            nn.GELU(),
            nn.Linear(num_hiddens, num_channels, bias = False),
            nn.Sigmoid())

        self.gem = GeM()
        self.fc = nn.Linear(num_channels, num_channels)
        self.norm = L2Norm()

    def forward(self, x):
        x_feat, x_cls = x
        x_feat = self.gem(x_feat).flatten(1)
        x_atte = self.channel_attention(x_feat)
        x_feat = self.fc(x_feat * x_atte)
        x_feat = self.norm(x_feat)
        return x_feat


class CLS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, x = x
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(32, 768, 16, 16), torch.randn(32, 768)
    agg = MixedGeM(num_channels = 768, num_hiddens = None, num_clusters = 192, dim_clusters = 2)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == '__main__':
    main()