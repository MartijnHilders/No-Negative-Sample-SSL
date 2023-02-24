import torch.nn as nn
import torch


class BarlowLoss(nn.Module):
    """
            Implementation of barlow Twins (adapted from https://github.com/facebookresearch/barlowtwins)
            """
    def __init__(self, ssl_batch_size, embedding_size, lambda_coeff):
        super(BarlowLoss, self).__init__()
        self.ssl_batch_size = ssl_batch_size
        self.embedding_size = embedding_size
        self.lambda_coeff = lambda_coeff


    def forward(self, x_1, x_2):
        # empirical cross-correlation matrix
        self.batchnorm = nn.BatchNorm1d(self.embedding_size, affine=False, device=x_1.device)
        c = torch.matmul(self.batchnorm(x_1).T, self.batchnorm(x_2))
        c.div_(self.ssl_batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag
        return loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()