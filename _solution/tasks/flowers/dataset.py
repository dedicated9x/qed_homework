from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import hydra
from _solution.common.utils import pprint_sample
import scipy.io
import PIL.Image
import timm.data.transforms_factory


def get_transform(role):
    """
    Note that these transforms play 2 roles:
    - preprocessing,
    - augmentation (in the case of training)
    """
    assert role in ["train", "val", "test"]

    common_params = dict(
        input_size=(3, 384, 384),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if role == "train":
        return timm.data.transforms_factory.create_transform(
            **common_params,
            is_training=True,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.3333333333333333],
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation="random",
            re_prob=0.25,
            re_mode="pixel"
        )
    else:
        return timm.data.transforms_factory.create_transform(
            **common_params,
            is_training=False,
            interpolation="bicubic",
            crop_pct=0.9
        )


class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, role):
        super(FlowersDataset, self).__init__()
        assert role in ["train", "test", "val"]

        # TODO ta sciezka powinna byc inna
        path_root = Path(r"C:\temp\appsilon")
        self.path_images = path_root / "17flowers" / "jpg"
        list_filenames = (self.path_images / "files.txt").read_text().split("\n")
        self.df = pd.DataFrame().assign(filename=list_filenames)
        self.df["label"] = self.df["filename"].apply(lambda x: divmod(int(x[6:-4]) - 1, 80)[0])

        mask_idxs = scipy.io.loadmat(path_root / "datasplits.mat")[self.get_mat_key(role)]
        mask_idxs = mask_idxs - 1   #Indices in .mat file they start from 1, not from 0.
        mask_dense = np.zeros(self.df.shape[0])
        mask_dense[mask_idxs] = 1
        self.df = self.df[mask_dense.astype(bool)].reset_index(drop=True)
        self.transform = get_transform(role)

    def get_mat_key(self, role):
        return {
            "train": "trn1",
            "val": "val1",
            "test": "tst1",
        }[role]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = PIL.Image.open(self.path_images / row["filename"])
        x = self.transform(x)

        return {
            "x": x,
            "y": torch.tensor(row["label"]).to(torch.float32),
        }


@hydra.main(config_path="conf", config_name="base")
def display_sample(config):
    ds = FlowersDataset(role="train")
    sample = next(iter(ds))
    pprint_sample(sample)

if __name__ == '__main__':
    display_sample()


