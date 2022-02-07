import torch
import torch.nn as nn

class SomeMLP(nn.Module):
    def __init__(self, config):
        super(SomeMLP, self).__init__()
        self.l1 = nn.Linear(28 * 28, config.model.hidden_dim)
        self.l2 = nn.Linear(config.model.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x