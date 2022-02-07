from pathlib import Path
import numpy as np
import pandas as pd
import hydra
import omegaconf

import src.tasks.qed.preprocessing._dictionaries as _dictionaries


def have_mapped_columns(df_train, df_test):
    return df_train.columns.to_list() == (df_test.columns.to_list() + ["notified"])


def imput_mean_and_mask(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    mean = df[col_name].mean()
    mask = df[col_name].isna().astype(np.float32)
    col_filled = df[col_name].fillna(mean)
    df = df.assign(**{col_name: col_filled})
    df = df.assign(**{f"_mask_{col_name}": mask})
    return df


def encode(df, encoding):
    for col_name, _map in encoding.items():
        if col_name in df.columns.to_list():
            df = df.assign(**{col_name: df[col_name].replace(_map)})
    return df


def process_df(df, config, tag: str):
    if tag == "train":
        buffer_label = df["notified"]

    # Feature selection
    selection = _dictionaries.tag2selection[config.dataset.selection]
    df = df[selection]

    # Data imputation
    for col_name in _dictionaries.list_incomplete_columns:
        if col_name in df.columns.to_list():
            df = imput_mean_and_mask(df, col_name)

    # Encode cathegorical variables
    df = encode(df, _dictionaries.encoding)

    # We have to be ensured that target is at the end of dataframe.
    if tag == "train":
        df["notified"] = buffer_label

    # Cast to unified type.
    df = df.astype(np.float32)
    return df


def get_data(config):
    path_root = Path(r"C:\temp\qed")

    df_train = pd.read_csv(path_root / config.paths["train"], sep="|")
    df_test = pd.read_csv(path_root / config.paths["test"], sep="|")

    df_train = process_df(df_train, config, "train")
    df_test = process_df(df_test, config, "test")

    assert have_mapped_columns(df_train, df_test)
    feature_col_names = df_test.columns.to_list()

    return {
        "train": df_train,
        "test": df_test,
        "feature_col_names": feature_col_names
    }

@hydra.main(config_path="../_sklearn/conf", config_name="32")
def main(config: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(config))
    data = get_data(config)


if __name__ == '__main__':
    main()