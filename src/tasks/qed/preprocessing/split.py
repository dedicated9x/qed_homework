from pathlib import Path
import pandas as pd


def main(idx_max):
    path_raw = Path(r"C:\temp\qed\raw")
    path_output = path_raw.with_name("processed") / str(idx_max)


    path_tab = path_raw / "cybersecurity_training" / "cybersecurity_training.csv"
    path_ts = path_raw / "localized_alerts_data" / "localized_alerts_data.csv"
    df = pd.read_csv(path_tab, sep="|")

    df = df.iloc[:idx_max]
    set_chosen_idxs = set(df["alert_ids"].values.tolist())

    df_ts = pd.read_csv(path_ts, sep="|")
    df_ts = df_ts[df_ts["alert_ids"].isin(set_chosen_idxs)]

    path_output.mkdir()
    df.to_csv(path_output / "cybersecurity_training.csv", index=False, sep="|")
    df_ts.to_csv(path_output / "localized_alerts_data.csv", index=False, sep="|")



if __name__ == '__main__':
    main(idx_max=1000)
