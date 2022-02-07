import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchmetrics.functional

from src.common.module import BaseModule
from src.tasks.qed.task_clf.arch import QedNet
from src.tasks.qed.task_adapt.dataset import PairDataset


class QedAdaptModule(BaseModule):
    def __init__(self, config=None):
        super(QedAdaptModule, self).__init__(config)

        # TODO i tak musi byc inny base
        self.ds_train = PairDataset(config, role="train")
        self.ds_val = PairDataset(config, role="val")

        self.model = QedNet(config)
        self.loss_fnc = nn.BCELoss(reduction='none')

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)

        loss = self.loss_fnc(y_hat, y)
        loss = loss.mean()
        retval = {"loss": loss, "y_hat": y_hat, "y": y}

        if self.config.optimizer.use_scheduler == True:
            self._scheduler_step()

        return retval

    def training_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])
        loss = torch.Tensor([batch['loss'] for batch in outputs])

        auc = torchmetrics.functional.auroc(y_hat, y.int()).item()
        loss_avg = loss.mean()
        self.log("Train/AUC", auc)
        self.log("Train/Loss", loss_avg)


    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)
        return {"y_hat": y_hat, "y": y}


    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])

        auc = torchmetrics.functional.auroc(y_hat, y.int()).item()
        print(f"\n AUC = {auc:.2f}")
        self.log("Val/AUC", auc)

        if self.config.optimizer.use_scheduler == True:
            self._scheduler_log()
