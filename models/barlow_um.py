import torch
from pytorch_lightning.core.module import LightningModule
from torch import nn

from models.mlp import ProjectionMLP

class UnimodalBarlow(LightningModule):
    """
        Implementation of barlow Twins (adapted from https://github.com/facebookresearch/barlowtwins)
        """

    def __init__(self, modality, encoder, hidden=[256, 128], batch_size=64, optimizer_name_ssl='adam', lr=0.001, lambda_coeff = 5e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters('modality', 'hidden', 'batch_size', 'optimizer_name_ssl', 'lr')
        
        self.modality = modality
        self.encoder = encoder
        self.projection = ProjectionMLP(self.encoder.out_size, hidden)
        self.modality = modality
        self.projections = {}
        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr

        self.ssl_batch_size = batch_size
        self.embedding_size = hidden[-1]
        self.lambda_coeff = lambda_coeff

        self.batchnorm = nn.BatchNorm1d(hidden[-1], affine=False)


    def forward(self, x1, x2):
        x_1 = self.encoder(x1)
        x_1 = nn.Flatten()(x_1)
        x_1 = self.projection(x_1)

        x_2 = self.encoder(x2)
        x_2 = nn.Flatten()(x_2)
        x_2 = self.projection(x_2)
        return x_1, x_2

    def _compute_barlow_loss(self, x_1, x_2, partition):

        # empirical cross-correlation matrix
        c = torch.matmul(self.batchnorm(x_1).T, self.batchnorm(x_2))
        c.div_(self.ssl_batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag
        self.log(f"ssl_{partition}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x1, x2 = batch[self.modality]
        outs = self(x1.float(), x2.float())
        loss = self._compute_barlow_loss(outs[0], outs[1], 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch[self.modality]
        outs = self(x1.float(), x2.float())
        loss = self._compute_barlow_loss(outs[0], outs[1], 'val')

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