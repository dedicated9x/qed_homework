from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchmetrics.functional
from _solution.tasks.qed.task_clf.dataset import QedDataset
from _solution.common.module import BaseModule
from _solution.tasks.qed.task_clf.arch import QedNet

def weighted_loss(loss_unweighted, y, weights: Dict[str, float]):
    w = torch.where(y > 0.5, weights["1"], weights["0"])
    w = w / w.sum()     # Sum of all weights == 1
    w = w.shape[0] * w
    loss_weighted = w * loss_unweighted
    return loss_weighted

class QedModule(BaseModule):
    def __init__(self, config=None):
        super(QedModule, self).__init__(config)

        csv_paths = {
            "train": r"C:\temp\qed\raw\cybersecurity_training\cybersecurity_training.csv",
            "test": r"C:\temp\qed\raw\cybersecurity_test\cybersecurity_test.csv"
        }
        self.ds_train = QedDataset(config, csv_path=csv_paths["train"], role="train")
        self.ds_val = QedDataset(config, csv_path=csv_paths["train"], role="val")
        self.ds_test = QedDataset(config, csv_path=csv_paths["test"], role="test")

        self.model = QedNet(config)
        self.loss_fnc = nn.BCELoss(reduction='none')

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)

        loss = self.loss_fnc(y_hat, y)
        # TODO ta proporcja jest jeszcze gorsza (w treningowym: 0.05 vs 0.95)
        if self.config.model.loss_balanced == True:
            loss = weighted_loss(loss, y, {"0": 0.15, "1": 0.85})
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

    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])

        auc = torchmetrics.functional.auroc(y_hat, y.int()).item()
        print(f"\n AUC = {auc:.2f}")
        self.log("Val/AUC", auc)

        if self.config.optimizer.use_scheduler == True:
            self._scheduler_log()
