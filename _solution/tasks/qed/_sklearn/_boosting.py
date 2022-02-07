import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
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


@hydra.main(config_path="conf", config_name="32")
def main(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))

    data = get_data(config)
    model = train_single(
        df_train=data["train"],
        model=sklearn.ensemble.GradientBoostingClassifier(random_state=0),
        feature_col_names=data["feature_col_names"],
        idx_fold=0
    )

if __name__ == '__main__':
    main()

