import numpy as np
import hydra
from _solution.tasks.qed.dataset import QedDataset, get_split_mask


def is_valid_split(df):
    m_train = get_split_mask(df, type_dest="train", idx_fold=0)
    m_val = get_split_mask(df, type_dest="val", idx_fold=0)
    return ((m_train.astype(int) + m_val.astype(int)) == 1).all()


def is_stratified(df):
    m_train = get_split_mask(df, type_dest="train", idx_fold=0)
    m_val = get_split_mask(df, type_dest="val", idx_fold=0)

    mean_train = df[m_train]["notified"].mean()
    mean_val = df[m_val]["notified"].mean()
    return abs(mean_train - mean_val) < 0.001


def are_folds_valid(df):
    list_masks = [get_split_mask(df, type_dest="val", idx_fold=i) for i in range(5)]
    return (np.stack(list_masks).mean(axis=0) == 0.2).all()

@hydra.main(config_path="../conf", config_name="005_shallow_std")
def _test(config):
    ds = QedDataset(config, max_idx=config.dataset.split_idx)
    df = ds._df_all

    assert is_valid_split(df)
    assert is_stratified(df)
    assert are_folds_valid(df)
    print("OK!")

if __name__ == '__main__':
    _test()
