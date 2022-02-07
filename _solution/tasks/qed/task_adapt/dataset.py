import torch
import torch.utils.data
import hydra
from _solution.common.utils import pprint_sample
from _solution.tasks.qed.utils.paths import LIST_CSV_PATHS
from _solution.tasks.qed.task_clf.dataset import QedDataset


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, config, role):
        assert role in ["train", "val"]

        self.ds_train = QedDataset(config, csv_path=LIST_CSV_PATHS["train"], role="train")
        self.ds_test = QedDataset(config, csv_path=LIST_CSV_PATHS["test"], role="test")

        # Save parameters
        self.role = role

    def __len__(self):
        return self.ds_train.__len__()

    def __getitem__(self, idx):
        idx_train = idx

        if self.role == "train":
            idx_test = torch.randint(0, self.ds_test.__len__(), (1,)).item()
        else:
            raise NotImplementedError

        sample_train = self.ds_train.__getitem__(idx_train)
        sample_test = self.ds_test.__getitem__(idx_test)

        return {
            "x": sample_train["x"],
            "z": sample_test["x"],
        }


@hydra.main(config_path="../qed/conf", config_name="005_shallow_std")
def display_sample(config):
    config.dataset.standarize = True

    ds = PairDataset(config, role="train")
    sample = next(iter(ds))

    pprint_sample(sample)

if __name__ == '__main__':
    display_sample()
