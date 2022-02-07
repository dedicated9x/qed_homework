import torch
import torch.nn as nn

class ClampedRelu(nn.Module):
    def __init__(self):
        super(ClampedRelu, self).__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = x.clamp(0, 1)
        return x