import torch
from pytorch_lightning.core.module import LightningModule
from torch import nn
from models.mlp import ProjectionMLP
from models.loss import WassOrderDistance, VICRegLoss

class MultimodalVicRegOrder(LightningModule):
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

        x_local = x.T
        x_global = self.projections[modality](x)
        return x_local, x_global

    def forward(self, x):
        outs_local = {}
        outs_global = {}
        for m in self.modalities:
            outs_local[m], outs_global[m] = self._forward_one_modality(m, x)

        return outs_local, outs_global

    def _compute_loss(self, x, y, partition):
        x_local, y_local = x[0], y[0]
        x_global, y_global = x[1], y[1]

        # initiate all the loss modules
        vicreg_loss = VICRegLoss.VICRegLoss(ssl_batch_size=self.ssl_batch_size, embedding_size=self.embedding_size,
                                     sim_coeff=self.sim_coeff, std_coeff=self.std_coeff, cov_coeff=self.cov_coeff)

        order_loss = WassOrderDistance.WassOrderDistance()

        #calculate losses
        repr_loss, std_loss, cov_loss, vic_loss = vicreg_loss(x_global, y_global)
        order_loss, transport = order_loss(x_local, y_local)  # todo can adjust params but enable this when we sweep also add this to model init

        #todo delete naïve method to save heatmaps locally.
        


        loss = order_loss + vic_loss
        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"vic_{partition}_loss", vic_loss)
        self.log(f"order_{partition}_loss", order_loss)
        self.log(f"ssl_{partition}_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs_local, outs_global = self(batch)
        x = [outs_local[self.modalities[0]], outs_global[self.modalities[0]]]
        y = [outs_local[self.modalities[1]], outs_global[self.modalities[1]]]
        loss = self._compute_loss(x, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m].float()
        outs_local, outs_global = self(batch)
        x = [outs_local[self.modalities[0]], outs_global[self.modalities[0]]]
        y = [outs_local[self.modalities[1]], outs_global[self.modalities[1]]]
        loss = self._compute_loss(x, y, 'val')

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