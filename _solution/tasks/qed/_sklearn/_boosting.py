from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torchmetrics.functional
import omegaconf
import hydra
from _solution.tasks.qed.preprocessing.process_df import get_data



def get_mask_train_idxs(labels: pd.Series, idx_fold, n_folds=5):
    labels = labels.values.astype(np.int32)
    idxs = np.arange(len(labels))[:, np.newaxis]

    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    list_masks = [
        {"train": t, "val": v}
        for t, v in skf.split(idxs, labels)
    ]
    mask = list_masks[idx_fold]
    return mask

def train_single(df_train, model, feature_col_names, idx_fold):
    split_mask = get_mask_train_idxs(labels=df_train["notified"], idx_fold=idx_fold)

    X_train = df_train[feature_col_names].values[split_mask["train"]]
    X_val = df_train[feature_col_names].values[split_mask["val"]]
    y_train = df_train["notified"].values[split_mask["train"]]
    y_val = df_train["notified"].values[split_mask["val"]]

    model = model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_val)[:, 1]

    val_auc = torchmetrics.functional.auroc(torch.Tensor(y_hat), torch.Tensor(y_val).int()).item()
    print(f"VAL/auc={val_auc:.3f}")
    return model

def create_predictions(df_test, model, config) -> None:
    X_test = df_test.values
    y_hat = model.predict_proba(X_test)[:, 1]

    path_results = Path(r"C:\temp\qed\results") / f"{config.description}.txt"
    text = "\n".join(y_hat.round(4).astype(str).tolist())
    path_results.write_text(text)

# @hydra.main(config_path="conf", config_name="s_base")
@hydra.main(config_path="conf", config_name="32")
def main(config: omegaconf.DictConfig) -> None:
    # print(omegaconf.OmegaConf.to_yaml(config))

    data = get_data(config)

    model = train_single(
        df_train=data["train"],
        model=GradientBoostingClassifier(random_state=0, max_depth=7),
        feature_col_names=data["feature_col_names"],
        idx_fold=0
    )
    # create_predictions(data["test"], model, config)


if __name__ == '__main__':
    main()

"""VAL/auc=0.877"""
# TODO zrob feature seelction
