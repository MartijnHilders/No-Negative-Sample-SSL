import random

import pytorch_lightning.core.module as pl
import pytorch_lightning.loggers as loggers
import wandb



class TransHeatmap(pl.Callback):
    """
    A callback which caches all labels and predictions encountered during a testing epoch,
    then logs a confusion matrix to WandB at the end of the test.
    """

    def __init__(self, class_names):
        self.class_names = class_names
        self._reset_state()

    def _reset_state(self):
        self.labels = []
        self.preds = []
        self.transport = []

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_state()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx,
                          dataloader_idx) -> None:

        #todo check how to get transportation plan here

        self.labels += (batch['label'] - 1).tolist()
        self.preds += outputs['preds'].tolist()
        self.transport += outputs['transport'].tolist()


    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Retrieve the WandB logger, if it exists.
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, loggers.WandbLogger):
                wandb_logger = logger
        if wandb_logger == None:
            return

        # Log the heatmap on 5 instances.
        for i in range(5):

            idx = random.randint(0, len(self.transport))

            heatmap = wandb.plots.HeatMap(
                x_labels=self.class_names,
                y_labels=self.class_names,
                matrix_values=self.transport[idx]

            )

            wandb_logger.experiment.log({
                "heatmap_" + str(i): heatmap,
                "global_step": trainer.global_step
            })

