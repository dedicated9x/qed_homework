import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data
import hydra

from src.common.utils import pprint_sample
from src.tasks.qed._preprocessing.process_df import process_df

STATS = {
    'correlatedcount': {'skewed': True, 'mean': 170.31, 'std': 5209.46},
    'srcip_cd': {'skewed': True, 'mean': 18.42, 'std': 228.76},
    'dstip_cd': {'skewed': True, 'mean': 10.1, 'std': 254.94},
    'srcport_cd': {'skewed': True, 'mean': 54.62, 'std': 794.02},
    'dstport_cd': {'skewed': True, 'mean': 9.95, 'std': 206.13},
    'domain_cd': {'skewed': True, 'mean': 1.09, 'std': 76.63},
    'protocol_cd': {'skewed': True, 'mean': 0.46, 'std': 11.57},
    'thrcnt_month': {'skewed': False, 'mean': 4962.43, 'std': 6930.01},
    'thrcnt_week': {'skewed': False, 'mean': 1483.98, 'std': 1838.28},
    'thrcnt_day': {'skewed': False, 'mean': 237.18, 'std': 301.41}
}


def get_split_mask(df: pd.DataFrame, type_dest, idx_fold, n_folds=5) -> pd.Series:
    if type_dest == "test":
        mask = pd.Series(np.ones(df.shape[0])).astype(bool)
        return mask

    labels = df["notified"].values.astype(np.int32)
    idxs = np.arange(len(labels))[:, np.newaxis]

    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    list_masks_pairs = [
        {"train": t, "val": v}
        for t, v in skf.split(idxs, labels)
    ]
    mask_sparse = list_masks_pairs[idx_fold][type_dest]
    mask = np.zeros(df.shape[0], dtype=int)
    mask[mask_sparse] = 1
    mask = pd.Series(mask).astype(bool)
    return mask


class QedDataset(torch.utils.data.Dataset):
    def __init__(self, config, csv_path, role):
        assert role in ["train", "val", "test"]
        preprocessing_type = {
            "train": "train",
            "val": "train",
            "test": "test"
        }[role]

        self._df_raw = pd.read_csv(csv_path, sep="|")
        self._df_all = process_df(self._df_raw, config, preprocessing_type)

        # Train/val/test split
        split_mask = get_split_mask(self._df_all, type_dest=role, idx_fold=config.dataset.idx_fold)
        self.df = self._df_all[split_mask]

        # Normalization
        if config.dataset.standarize == True:
            self.df = self._standarize_df(df=self.df, col2stats=STATS)

    def _standarize_df(self, df, col2stats):
        for col, stats in col2stats.items():
            if stats['skewed'] == True:
                col_std = df[col].apply(np.log).replace(-np.inf, -1) + 1
            else:
                col_std = (df[col] - stats["mean"]) / stats["std"]
                col_std = (col_std - col_std.min()) / (col_std.max() - col_std.min())
            df = df.assign(**{col: col_std})
        return df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        global_idx = idx

        row = self.df.iloc[global_idx]

        if "notified" in row.index:
            y = row["notified"]
            row = row.iloc[:-1]
        else:
            y = np.nan

        return {
            "x": torch.Tensor(row.values),
            "y": torch.tensor(y).type(torch.float32),
        }


@hydra.main(config_path="conf", config_name="005_shallow_std")
def display_sample(config):
    config.dataset.standarize = True

    ds = QedDataset(config, max_idx=config.dataset.split_idx)
    sample = next(iter(ds))

    pprint_sample(sample)

if __name__ == '__main__':
    display_sample()

