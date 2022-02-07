from collections import OrderedDict
import torch
import torch.nn as nn
from _solution.tasks.qed.utils.utils import ClampedRelu

class HiddenLinear(nn.Module):
    def __init__(self, hidden_dim):
        super(HiddenLinear, self).__init__()
        self.core = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.core(x)
        x = torch.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        list_dims = config.model.dims
        list_dim_pairs = [(_in, _out) for _in, _out in zip(list_dims[:-1], list_dims[1:])]

        self.bn0 = nn.BatchNorm1d(list_dims[0])
        self.first = nn.Linear(*list_dim_pairs[0])
        self.hidden_layers = nn.Sequential(OrderedDict([
            (f"layer_{idx + 1}", HiddenLinear(in_dim))
            for idx, (in_dim, out_dim)
            in enumerate(list_dim_pairs[1:-1])
        ]))

    def forward(self, x):
        # First
        x = self.bn0(x)
        x = torch.relu(self.first(x))

        # Hidden
        x =self.hidden_layers(x)

        return x

class QedNet(nn.Module):
    def __init__(self, config):
        super(QedNet, self).__init__()
        self.encoder = Encoder(config)

        self.last = nn.Linear(*tuple(config.model.dims[-2:]))

        if config.model.last_act == "relu":
            self.last_act = ClampedRelu()
        elif config.model.last_act == "sigmoid":
            self.last_act = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Last
        x = self.last(x)
        x = self.last_act(x)

        # Postprocessing
        x = x.flatten()
        return x


# TODO https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b

# TODO dodaj dropouty (bo w koncu w nie wierzysz)
# TODO i batch-normy 0 wszystkie kombinacje

# TODO trenowalne embeddingi

# TODO ten trik z IEEE_CIS Fraud Detection
"""
63 / 69 / 0.71
"""