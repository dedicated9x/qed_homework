import torch
import torch.utils.data
import torchvision
from _solution.common.utils import pprint_sample

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, min_idx=0, max_idx=60000):
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.wrapped = torchvision.datasets.mnist.MNIST(
            root=rf"C:\Datasets",
            train=True,
            download=False,
            transform=torchvision.transforms.ToTensor()
        )

    def __len__(self):
        return (self.max_idx - self.min_idx)

    def __getitem__(self, idx):
        global_idx = idx + self.min_idx
        x, y = self.wrapped[global_idx]
        return {
            "x": x.squeeze(),
            "y": torch.tensor(y),
        }


def display_sample():
    ds = MnistDataset(min_idx=0, max_idx=55000)
    sample = next(iter(ds))
    pprint_sample(sample)

if __name__ == '__main__':
    display_sample()