from typing import Dict, Tuple

from pathlib import Path
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.utils.data
import torchmetrics
from _solution.common.module import BaseModule
from _solution.tasks.conditions.dataset import DenseConditionsDataset

from src.models.gat import GAT
from src.featurization.gat_featurizer import GatGraphFeaturizer


class ConditionsModule(BaseModule):
    def __init__(self, config=None):
        super(ConditionsModule, self).__init__(config)

        self.featurizer = GatGraphFeaturizer(n_jobs=1)
        self.loss_fnc = nn.BCELoss(reduction='mean')
        self.best_val_f1 = 0

        self.model = GAT()
        self.ds_train = DenseConditionsDataset(max_idx=config.dataset.split_idx)
        self.ds_val = DenseConditionsDataset(min_idx=config.dataset.split_idx)

    def training_step(self, batch: Dict, batch_idx: int):
        X, y = self._xy_split(batch)
        X = self._compress_batch(batch_X=X)
        y_hat = self(X)
        loss = self.loss_fnc(y_hat, y)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        X, y = self._xy_split(batch)
        X = self._compress_batch(batch_X=X)
        y_hat = self(X)
        return {"y_hat": y_hat, "y": y, "n_nodes": X['n_nodes']}

    def validation_epoch_end(self, outputs: Dict) -> None:
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])

        auc = torchmetrics.functional.auroc(y_hat, y.int())
        f1 = torchmetrics.functional.classification.f_beta.f1_score(y_hat, y.int())

        n_nodes = torch.Tensor(np.concatenate([batch['n_nodes'] for batch in outputs]))
        self._log_validation_epoch_end(y_hat, y, n_nodes, auc, f1)

    def _log_validation_epoch_end(self, y_hat, y, n_nodes, auc, f1) -> None:
        # Console
        print(f"\n AUC = {auc:.2f} F1 = {f1:.2f}")
        # Checkpoint-saving callback.
        self.log("val_f1", f1)
        # Tensorboard
        self.logger.experiment.add_scalar("Val/AUC", auc, self.current_epoch)
        self.logger.experiment.add_scalar("Val/F1", f1, self.current_epoch)
        # Future analysis
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1

            path_outputs = Path(__file__).parent / "analysis" / "outputs"
            torch.save(y_hat.cpu().detach(), path_outputs / "y_hat.pickle")
            torch.save(y.cpu().detach(), path_outputs / "y.pickle")
            torch.save(n_nodes.cpu().detach(), path_outputs / "n_nodes.pickle")

    def _xy_split(self, batch: Dict) -> Tuple[Dict, Dict]:
        batch_X ={
            "n_nodes": batch["n_nodes"],
            "atom": batch["atom"],
            "bond": batch["bond"],
        }
        batch_y = batch['y']
        return batch_X, batch_y

    def _compress_batch(self, batch_X: Dict) -> Dict:
        batch_X = {
            "n_nodes": batch_X['n_nodes'].cpu().numpy(),
            "atom": scipy.sparse.csr_matrix(batch_X['atom'].cpu().numpy().astype(np.float64)),
            "bond": scipy.sparse.csr_matrix(batch_X['bond'].cpu().numpy().astype(np.float64)),
        }
        batch_X = self.featurizer.unpack(batch_X)
        return batch_X

