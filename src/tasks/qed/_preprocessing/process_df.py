from pathlib import Path
import numpy as np
import pandas as pd
import hydra
import omegaconf

import src.tasks.qed._preprocessing._encoding as module_encoding
import src.tasks.qed._preprocessing._feature_selection as module_feature_selection
import src.tasks.qed._preprocessing._imputation as module_imputation


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
    selection = module_feature_selection.tag2selection[config.dataset.selection]
    df = df[selection]

    # Data imputation
    for col_name in module_imputation.list_incomplete_columns:
        if col_name in df.columns.to_list():
            df = imput_mean_and_mask(df, col_name)

    # Encode cathegorical variables
    df = encode(df, module_encoding.encoding)

    # We have to be ensured that target is at the end of dataframe.
    if tag == "train":
        df["notified"] = buffer_label

    # Cast to unified type.
    df = df.astype(np.float32)
    return df


def get_data(config):
    path_data = Path(__file__).parent.parent.parent.parent.parent / "data"

    df_train = pd.read_csv(path_data / "cybersecurity_training.csv", sep="|")
    df_test = pd.read_csv(path_data / "cybersecurity_test.csv", sep="|")

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