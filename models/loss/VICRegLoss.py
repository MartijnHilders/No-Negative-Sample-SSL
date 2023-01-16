import torch.nn.functional as F
import torch.nn as nn
import torch


class VICRegLoss(nn.Module):
    def __init__(self, ssl_batch_size, embedding_size, sim_coeff, std_coeff, cov_coeff):
        super(VICRegLoss, self).__init__()
        self.ssl_batch_size = ssl_batch_size
        self.embedding_size = embedding_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(self.embedding_size) + self.off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        vic_loss = (
                self.sim_coeff * repr_loss +
                self.std_coeff * std_loss +
                self.cov_coeff * cov_loss
        )

        return repr_loss, std_loss, cov_loss, vic_loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()