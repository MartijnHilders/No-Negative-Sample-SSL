import random
import torch.nn.functional as F
import torch
from pytorch_lightning.core.module import LightningModule
from torch import nn
from models.mlp import ProjectionMLP, LocalProjectionMLP
from models.loss import VICRegLoss
from models.loss.order_preserving import WassOrderDistance_OPW, WassOrderDistance_OPW_batch, WassOrderDistance_Gromov, WassOrderDistance_Sinkhorn
import seaborn as sns
import matplotlib.pyplot as plt

class MultimodalVicRegOrder(LightningModule):
    """
    Implementation of VicReg for two modalities (adapted from https://github.com/facebookresearch/vicreg/)
    """
    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, sim_coeff=10, std_coeff=10, cov_coeff=5, optimizer_name_ssl='adam', lr=0.001, alpha_coeff= 1,
                 beta_coeff=1, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'sim_coeff', 'std_coeff', 'cov_coeff', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

        # create local and global projections
        self.local_projections = {}
        self.global_projections = {}

        for m in modalities:
            local_projection_size = self.get_local_projection_size(encoders[m].out_sample)
            self.local_projections[m] = LocalProjectionMLP(in_size=local_projection_size, hidden=hidden)
            self.global_projections[m] = ProjectionMLP(in_size=encoders[m].out_size, hidden=hidden)
        self.local_projections = nn.ModuleDict(self.local_projections)
        self.global_projections = nn.ModuleDict(self.global_projections)

        self.optimizer_name_ssl = optimizer_name_ssl
        self.lr = lr

        self.ssl_batch_size = batch_size
        self.embedding_size = hidden[-1]
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff

    def _forward_one_modality(self, modality, inputs):
        x = inputs[modality]
        x = self.encoders[modality](x)

        # local feature projection
        x_local = x.squeeze().permute(0, 2, 1)
        x_local = self.local_projections[modality](x_local)

        # global feature projection
        x_global = nn.Flatten()(x)
        x_global = self.global_projections[modality](x_global)

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

        order_loss = WassOrderDistance_OPW_batch.WassOrderDistance()

        #calculate losses
        repr_loss, std_loss, cov_loss, vic_loss = vicreg_loss(x_global, y_global)
        order_loss, transport = order_loss(x_local, y_local)
        order_loss = torch.mean(order_loss)


        #todo delete naïve method to save heatmaps locally. + find out how to only do it or certain epochs.
        # if random.random() < 0.005:
        #
        #
        #     x_local_norm =  (x_local - x_local.mean(dim=1, keepdim=True)/torch.sqrt(x_local.var(dim=1, keepdim=True) + 0.0001))
        #     y_local_norm = (y_local - y_local.mean(dim=1, keepdim=True)/torch.sqrt(y_local.var(dim=1, keepdim=True) + 0.0001))
        #
        #
        #     for i in range(3):
        #         idx = random.randint(0, x_local.shape[0]-1)
        #         # cdist_local = self.get_cosine_sim_matrix(x_local_norm[idx], y_local_norm[idx])
        #         cdist_local = torch.cdist(x_local_norm[idx], y_local_norm[idx])
        #         transport_plot = transport[idx].cpu().detach().numpy()
        #         s = sns.heatmap(transport_plot)
        #         s.set(ylabel="Inertial Features", xlabel="Skeleton Features", title='Transport Plan')
        #         plt.show()
        #
        #
        #         cdist = cdist_local.cpu().detach().numpy()
        #         s2 = sns.heatmap(cdist)
        #         s2.set(ylabel="Inertial Features", xlabel="Skeleton Features", title='Cdist Heatmap')
        #         plt.show()

        loss =  self.alpha_coeff * vic_loss + self.beta_coeff * order_loss


        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"vic_{partition}_loss", self.alpha_coeff * vic_loss)
        self.log(f"order_{partition}_loss", self.beta_coeff * order_loss)
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

    @staticmethod
    def get_local_projection_size(input):
        sample = input
        out = torch.squeeze(sample).permute(0, 2, 1) # need to adjust for different encoder configurations
        return out.shape[-1]

    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix