import random
import torch.nn.functional as F
import torch
from pytorch_lightning.core.module import LightningModule
from torch import nn
from models.mlp import ProjectionMLP, LocalProjectionMLP
from models.loss import BarlowLoss
from models.loss.order_preserving import WassOrderDistance_OPW, WassOrderDistance_OPW_batch, WassOrderDistance_Gromov, WassOrderDistance_Sinkhorn
import seaborn as sns
import matplotlib.pyplot as plt


class MultiModalBarlowOrder(LightningModule):

    def __init__(self, modalities, encoders, hidden=[256, 128], batch_size=64, optimizer_name_ssl='adam', lr=0.001,
                 lambda_coeff = 5e-3, alpha_coeff = 1, beta_coeff = 1, **kwargs):
        super().__init__()
        self.save_hyperparameters('modalities', 'hidden', 'batch_size', 'optimizer_name_ssl', 'lr')
        
        self.modalities = modalities
        self.encoders = nn.ModuleDict(encoders)

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
        self.lambda_coeff = lambda_coeff
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff


    def _forward_one_modality(self, modality, inputs):
        x1, x2 = inputs[modality]

        # x1
        x1 = x1.float()
        x_1 = self.encoders[modality](x1)
        x_1_local = x_1.squeeze().permute(0, 2, 1)
        x_1_local = self.local_projections[modality](x_1_local)

        x_1_global = nn.Flatten()(x_1)
        x_1_global = self.global_projections[modality](x_1_global)

        #x2
        x2 = x2.float()
        x_2 = self.encoders[modality](x2)
        x_2_local = x_2.squeeze().permute(0, 2, 1)
        x_2_local = self.local_projections[modality](x_2_local)

        x_2_global = nn.Flatten()(x_2)
        x_2_global = self.global_projections[modality](x_2_global)
        return [x_1_local, x_2_local], [x_1_global, x_2_global]

    def forward(self, x):
        outs_local = {}
        outs_global = {}
        for m in self.modalities:
            outs_local[m], outs_global[m] = self._forward_one_modality(m, x)
        return outs_local, outs_global

    def _compute_loss(self, x, y, partition):
        x_local, y_local = x[0], y[0]
        x_global, y_global = x[1], y[1]

        b_loss = BarlowLoss.BarlowLoss(ssl_batch_size=self.ssl_batch_size, embedding_size=self.embedding_size,
                                            lambda_coeff=self.lambda_coeff)

        o_loss = WassOrderDistance_OPW_batch.WassOrderDistance()

        # barlow twins losses
        inter_loss_1 = b_loss(x_global[0], x_global[1])
        inter_loss_2 = b_loss(y_global[0], y_global[1])
        intra_loss_1 = b_loss(x_global[0], y_global[1])
        intra_loss_2 = b_loss(x_global[1], y_global[0])
        bt_loss = inter_loss_1 + inter_loss_2 + intra_loss_1 + intra_loss_2

        # order loss
        order_loss, transport = o_loss(x_local[0], y_local[1])
        order_loss = torch.mean(order_loss)

        total_loss = self.alpha_coeff * bt_loss + self.beta_coeff * order_loss

        # plotting the heatmaps
        # todo delete
        if random.random() < 0.005:


            x_local_norm =  (x_local[0] - x_local[0].mean(dim=1, keepdim=True)/torch.sqrt(x_local[0].var(dim=1, keepdim=True) + 0.0001))
            y_local_norm = (y_local[1] - y_local[1].mean(dim=1, keepdim=True)/torch.sqrt(y_local[1].var(dim=1, keepdim=True) + 0.0001))


            for i in range(3):
                idx = random.randint(0, x_local[0].shape[0]-1)
                # cdist_local = self.get_cosine_sim_matrix(x_local_norm[idx], y_local_norm[idx])
                cdist_local = torch.cdist(x_local_norm[idx], y_local_norm[idx])
                transport_plot = transport[idx].cpu().detach().numpy()
                s = sns.heatmap(transport_plot)
                s.set(ylabel="Inertial Features", xlabel="Skeleton Features", title='Transport Plan')
                plt.show()


                cdist = cdist_local.cpu().detach().numpy()
                s2 = sns.heatmap(cdist)
                s2.set(ylabel="Inertial Features", xlabel="Skeleton Features", title='Cdist Heatmap')
                plt.show()



        self.log(f"bt_{partition}_loss",self.alpha_coeff * bt_loss)
        self.log(f"order_{partition}_loss", self.beta_coeff * order_loss)
        self.log(f"ssl_{partition}_loss", total_loss)
        self.log(f"{self.modalities[0]}_loss", inter_loss_1)
        self.log(f"{self.modalities[1]}_loss", inter_loss_2)
        self.log(f"extra_1_loss", intra_loss_1)
        self.log(f"extra_2_loss", intra_loss_2)

        return total_loss



    def training_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m]
        outs_local, outs_global = self(batch)
        x = [outs_local[self.modalities[0]], outs_global[self.modalities[0]]]
        y = [outs_local[self.modalities[1]], outs_global[self.modalities[1]]]
        loss = self._compute_loss(x, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        for m in self.modalities:
            batch[m] = batch[m]
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
        out = torch.squeeze(sample).permute(0, 2, 1)  # need to adjust for different encoder configurations
        return out.shape[-1]

    @staticmethod
    def get_cosine_sim_matrix(features_1, features_2):
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)
        similarity_matrix = torch.matmul(features_1, features_2.T)
        return similarity_matrix