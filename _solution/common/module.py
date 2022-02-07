import pytorch_lightning as pl
import math
import torch
import torch.utils.data


class BaseModule(pl.LightningModule):
    """
    Base training configuration.
    Suitable for most cases.
    """
    def __init__(self, config=None):
        super(BaseModule, self).__init__()
        self.config = config

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)
        return {"y_hat": y_hat, "y": y}

    def test_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)
        return {"y_hat": y_hat, "y": y}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.config.trainer.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.config.trainer.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        if self.config.main.is_tested:
            return torch.utils.data.DataLoader(
                self.ds_test,
                batch_size=self.config.trainer.batch_size,
                num_workers=4
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        if self.config.optimizer.use_scheduler == True:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.optimizer.lr,
                epochs=self.config.trainer.max_epochs + 1,
                steps_per_epoch=math.ceil(self.ds_train.__len__() / self.config.trainer.batch_size),
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _scheduler_step(self):
        self.lr_schedulers().step()

    def _scheduler_log(self):
        sch = self.lr_schedulers()

        lr = sch.get_last_lr()[0]
        progress = sch._step_count / sch.total_steps

        self.log("trainer/lr", lr)
        self.log("trainer/progress", progress)
