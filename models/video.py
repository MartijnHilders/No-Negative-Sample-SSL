import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule
import torchvision.models as models

class ResNet(LightningModule):
    def __init__(self,
                 out_size,
                 sample_length,
                 resnet_type,
                 lr=1e-3,
                 optimizer_name='adam',
                 metric_scheduler='accuracy',
                 **kwargs):

        super().__init__()

        resnet = {34: models.resnet34, 50: models.resnet50, 101: models.resnet101}
        self.res_net = resnet[resnet_type](weights=None)  # initialize randomly
        in_size = self.res_net.fc.in_features
        self.res_net.fc = Identity(in_size=in_size)

        self.classifier = nn.Linear(in_size, out_size)
        self.loss = nn.CrossEntropyLoss()
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.out_size = out_size

    def forward(self, x):
        x = torch.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))  # unpack by reshaping
        x = torch.permute(x, (0, -1, 1, 2)).float()  # put into correct shape for the model.
        x = self.model(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        try:
            x = batch['depth']
        except:
            x = batch['rgb']

        y = batch['label'] - 1
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        try:
            x = batch['depth']
        except:
            x = batch['rgb']

        y = batch['label'] - 1
        out = self(x)
        preds = torch.argmax(out, dim=1)

        loss = self.loss(out, y)
        self.log(f"{prefix}_loss", loss)
        return {f"{prefix}_loss": loss, "preds": preds}


    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        ### Add LR Schedulers
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['val', self.metric_scheduler])
            }
        }


class VideoNet(LightningModule):
    def __init__(self,
                 out_size,
                 sample_length,
                 model_type,
                 lr=1e-3,
                 optimizer_name='adam',
                 metric_scheduler='accuracy',
                 **kwargs):

        super().__init__()
        # self.save_hyperparameters('out_size', 'sample_length', 'model_type')
        self.save_hyperparameters()

        # todo need to adapt for the latter 2, these are sequential type
        video = {"res": models.video.r3d_18, "mvit": models.video.mvit_v1_b , "s3d": models.video.s3d}
        self.video_net = video[model_type](weights=None)
        in_size = self.video_net.fc.in_features
        self.video_net.fc = Identity(in_size=in_size)  # add identity layer to keep flexibility in last layer

        self.classifier = nn.Linear(in_size, out_size)
        self.loss = nn.CrossEntropyLoss()
        self.metric_scheduler = metric_scheduler
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.out_size = out_size
        self.sample_length = sample_length


    def forward(self, x):
        x = self.video_net(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        try:
            x = batch['depth']
        except:
            x = batch['rgb']

        y = batch['label'] - 1
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        try:
            x = batch['depth']
        except:
            x = batch['rgb']

        y = batch['label'] - 1
        out = self(x)
        preds = torch.argmax(out, dim=1)

        loss = self.loss(out, y)
        self.log(f"{prefix}_loss", loss)
        return {f"{prefix}_loss": loss, "preds": preds}

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        ### Add LR Schedulers
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": '_'.join(['val', self.metric_scheduler])
            }
        }

# create custom identity layer to safe the sizes
class Identity(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.out_size = in_size

    def forward(self, x):
        return x