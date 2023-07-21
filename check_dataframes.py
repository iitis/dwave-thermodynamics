import pandas as pd
import os

from tqdm import tqdm

cwd = os.getcwd()
data = "Q4_pausing"

if __name__ == "__main__":

    df = pd.read_csv(os.path.join(cwd, "results", "raw_data", f"{data}.csv"), sep=";")
    df = df.drop(columns=["Unnamed: 0"])
    times = df.annealing_time.unique().tolist()

    df_clean = pd.DataFrame(columns=df.columns)
    df2 = df[(df["annealing_time"] == 2.0)]

    df_clean = pd.concat([df_clean, df2], ignore_index=True)

    for i in range(len(times)):
        if i == 0:
            continue

        current_df = df[(df["annealing_time"] == times[i])]
        previous_df = df[(df["annealing_time"] == times[i-1])]
        current_unique = current_df.init_state.unique().tolist()
        previous_unique = previous_df.init_state.unique().tolist()

        to_keep = [s for s in current_unique if s not in previous_unique]
        current_df_clean = current_df[current_df["init_state"].isin(to_keep)]
        assert len(current_df_clean) == 1000

        df_clean = pd.concat([df_clean, current_df_clean], ignore_index=True)

    df_clean.to_csv(os.path.join(cwd, "results", "raw_data", f"{data}_clean.csv"), sep=";")
