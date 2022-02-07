import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import omegaconf
import hydra
from _solution.tasks.qed.preprocessing.process_df import get_data
from _solution.tasks.qed._sklearn._boosting import train_single

def learnable_stack_25_7(data, idx_fold):
    import _solution.tasks.qed.preprocessing._dictionaries as _dictionaries


    df_train_25 = data["train"][_dictionaries.tag2selection[25] + ["notified"]]
    col_names_25 = df_train_25.columns.to_list()[:-1]
    backbone25 = train_single(
        df_train=df_train_25,
        model=GradientBoostingClassifier(random_state=0),
        feature_col_names=col_names_25,
        idx_fold=idx_fold
    )

    df_train_7 = data["train"][_dictionaries.tag2selection[7] + ["notified"]]
    col_names_7 = df_train_7.columns.to_list()[:-1]
    backbone7 = train_single(
        df_train=df_train_7,
        model=GradientBoostingClassifier(random_state=0),
        feature_col_names=col_names_7,
        idx_fold=idx_fold
    )

    y25 = backbone25.predict_proba(df_train_25[col_names_25].values)[:, 1]
    y7 = backbone7.predict_proba(df_train_7[col_names_7].values)[:, 1]
    df_train_final = pd.DataFrame().assign(
        y25=y25,
        y7=y7,
        notified=data["train"]["notified"]
    ).astype(np.float32)

    head = train_single(
        df_train=df_train_final,
        model=GradientBoostingClassifier(random_state=0),
        feature_col_names=["y25", "y7"],
        idx_fold=idx_fold
    )

    return head

# @hydra.main(config_path="conf", config_name="s_base")
@hydra.main(config_path="../../_sklearn/conf", config_name="32")
def main(config: omegaconf.DictConfig) -> None:
    # print(omegaconf.OmegaConf.to_yaml(config))

    data = get_data(config)
    model = learnable_stack_25_7(data, idx_fold=0)


if __name__ == '__main__':
    main()