import torch
from pytorch_lightning.core.module import LightningModule
from torch import nn

from models.mlp import ProjectionMLP
from models.loss import BarlowLoss

class MultiModalBarlow(LightningModule):

    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, optimizer_name_ssl='adam', lr=0.001,
                 lambda_coeff = 5e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'optimizer_name_ssl', 'lr')
        
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
        self.lambda_coeff = lambda_coeff


    def _forward_one_modality(self, modality, inputs):
        x1, x2 = inputs[modality]
        x1 = x1.float()
        x_1 = self.encoders[modality](x1)
        x_1 = nn.Flatten()(x_1)
        x_1 = self.projections[modality](x_1)

        x2 = x2.float()
        x_2 = self.encoders[modality](x2)
        x_2 = nn.Flatten()(x_2)
        x_2 = self.projections[modality](x_2)
        return x_1, x_2

    def forward(self, x):
        outs = {}
        for m in self.modalities:
            outs[m] = self._forward_one_modality(m, x)
        return outs

    def _compute_loss(self, outs, partition):

        b_loss = BarlowLoss.BarlowLoss(ssl_batch_size=self.ssl_batch_size, embedding_size=self.embedding_size,
                                            lambda_coeff=self.lambda_coeff)

        loss = {}
        for m in self.modalities:
            x1, x2 = outs[m]
            loss[m] = b_loss(x1, x2)

        intra_loss_1 = b_loss(outs[self.modalities[0]][0], outs[self.modalities[1]][1])
        intra_loss_2 = b_loss(outs[self.modalities[0]][1], outs[self.modalities[1]][0])


        total_loss = loss[self.modalities[0]] + loss[self.modalities[1]] + intra_loss_1 + intra_loss_2

        self.log(f"ssl_{partition}_loss", total_loss)
        self.log(f"{self.modalities[0]}_loss", loss[self.modalities[0]])
        self.log(f"{self.modalities[1]}_loss", loss[self.modalities[1]])
        self.log(f"extra_1_loss", intra_loss_1)
        self.log(f"extra_2_loss", intra_loss_2)

        return total_loss



    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m]
        outs = self(batch)
        loss = self._compute_loss(outs, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m]
        outs = self(batch)
        loss = self._compute_loss(outs, 'val')

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