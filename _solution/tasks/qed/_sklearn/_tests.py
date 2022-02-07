import numpy as np
import pandas as pd
import omegaconf
import hydra
from _solution.tasks.qed.preprocessing.process_df import get_both_dfs
from _solution.tasks.qed._sklearn._boosting import get_mask_train_idxs


def _test_split(df_train, split_func):
    list_masks = [split_func(labels=df_train["notified"], idx_fold=i) for i in range(5)]
    train_ratios = [df_train["notified"].values[mask["train"]].mean() for mask in list_masks]
    val_ratios = [df_train["notified"].values[mask["val"]].mean() for mask in list_masks]
    print(train_ratios)
    print(val_ratios)
    assert (pd.Series(np.concatenate([mask["train"] for mask in list_masks]).tolist()).value_counts() == 4).all()




@hydra.main(config_path="conf", config_name="s_base")
def main(config: omegaconf.DictConfig) -> None:
    df_train, _ = get_both_dfs(config)
    _test_split(df_train, split_func=get_mask_train_idxs)
    a = 2

if __name__ == '__main__':
    main()

