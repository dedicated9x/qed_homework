import torch
import torch.nn as nn
import torch.nn.functional as F
from _solution.common.module import BaseModule
from _solution.tasks.flowers.arch import CctFlower17
from _solution.tasks.flowers.dataset import FlowersDataset

class FlowersModule(BaseModule):
    def __init__(self, config=None):
        super(FlowersModule, self).__init__(config)

        self.model = CctFlower17()
        self.loss_train = nn.CrossEntropyLoss()

        self.ds_train = FlowersDataset(role="train")
        self.ds_val = FlowersDataset(role="val")
        self.ds_test = FlowersDataset(role="test")

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)
        y = F.one_hot(y.to(torch.int64), num_classes=17).to(torch.float32)
        loss = self.loss_train(y_hat, y)
        return loss


    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])
        acc = (y_hat.argmax(dim=1) == y).to(torch.float32).mean()
        print(f"\n Val/Acc1 = {acc:.2f}")
        self.log("Val/Acc1", acc)


    def test_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])
        acc = (y_hat.argmax(dim=1) == y).to(torch.float32).mean()
        self.log("Test/Acc1", acc)


