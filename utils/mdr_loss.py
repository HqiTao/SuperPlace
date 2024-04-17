import torch
import torch.nn as nn

from pytorch_metric_learning import distances

class MDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = nn.Parameter(torch.tensor([-3.0, 0.0, 3.0]))
        self.momentum = 0.9

        momented_mean = torch.zeros(1)
        momented_std = torch.zeros(1)
        self.register_buffer("momented_mean", momented_mean)
        self.register_buffer("momented_std", momented_std)

        # The variable is used to check whether momented_mean and momented_std are initialized
        self.init = False
        self.distances = distances.CosineSimilarity()

    def initialize_statistics(self, mean, std):
        self.momented_mean = mean
        self.momented_std = std
        self.init = True

    def forward(self, embeddings):
        dist_mat = self.distances(embeddings)
        pdist_mat = dist_mat[~torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device,)]
        dist_mat = dist_mat.view(-1)

        mean = dist_mat.mean().detach()
        std = dist_mat.std().detach()

        if not self.init:
            self.initialize_statistics(mean, std)
        else:
            self.momented_mean = (
                1 - self.momentum
            ) * mean + self.momentum * self.momented_mean
            self.momented_std = (
                1 - self.momentum
            ) * std + self.momentum * self.momented_std

        normalized_dist = (pdist_mat - self.momented_mean) / self.momented_std
        difference = (normalized_dist[None] - self.levels[:, None]).abs().min(dim=0)[0]
        loss = difference.mean()

        return loss