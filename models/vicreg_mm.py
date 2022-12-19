import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch import nn

from models.mlp import ProjectionMLP


import matplotlib.pyplot as plt

class MultimodalVicReg(LightningModule):
    """
    Implementation of VicReg for two modalities (adapted from https://github.com/facebookresearch/vicreg/)
    """
    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, sim_coeff=10, std_coeff=10, cov_coeff=5, optimizer_name_ssl='adam', lr=0.001, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'sim_coeff', 'std_coeff', 'cov_coeff', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

        self.projections = {}
        for m in modalities:
            self.projections[m] = ProjectionMLP(in_size=encoders[m].out_size, hidden=hidden)
        self.projections = nn.ModuleDict(self.projections)

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr

        self.ssl_batch_size = batch_size
        self.embedding_size = hidden[-1]
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def _forward_one_modality(self, modality, inputs):
        x = inputs[modality]
        x = self.encoders[modality](x)
        x = nn.Flatten()(x)
        x = self.projections[modality](x)
        return x

    # todo try to add projector and decoder and see what happens.
    def _compute_vicreg_loss(self, x, y, partition):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"ssl_{partition}_loss", loss)
        return loss

    def forward(self, x):
        outs = {}
        for m in self.modalities:
            outs[m] = self._forward_one_modality(m, x)
        return outs

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        x, y = outs[self.modalities[0]], outs[self.modalities[1]]
        loss  = self._compute_vicreg_loss(x, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs = self(batch)
        x, y = outs[self.modalities[0]], outs[self.modalities[1]]
        loss = self._compute_vicreg_loss(x, y, 'val')

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_val_loss'
                }
            }


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()