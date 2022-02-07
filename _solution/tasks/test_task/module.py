import torch
import torch.nn.functional as F
import torchmetrics.functional
from _solution.tasks.test_task.dataset import MnistDataset
from _solution.common.module import BaseModule
from _solution.tasks.test_task.arch import SomeMLP

class MnistModule(BaseModule):
    def __init__(self, config=None):
        super(MnistModule, self).__init__(config)

        self.model = SomeMLP(config)
        self.ds_train = MnistDataset(max_idx=config.dataset.split_idx)
        self.ds_val = MnistDataset(min_idx=config.dataset.split_idx)

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss


    def validation_epoch_end(self, outputs):
        y_hat = torch.cat([batch['y_hat'] for batch in outputs])
        y = torch.cat([batch['y'] for batch in outputs])
        acc = torchmetrics.functional.accuracy(y_hat.argmax(dim=1), y).item()
        print(f"\n ACC = {acc:.2f}")