import pandas as pd
import os

from tqdm import tqdm

cwd = os.getcwd()

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(cwd, "results", "raw_data", "Q1_pausing.csv"), sep=";")
    df = df.drop(columns=["Unnamed: 0"])
    c = 0
    un2 = df[(df["annealing_time"] == 2.0)].init_state.unique().tolist()
    un13 = df[(df["annealing_time"] == 12.88888888888889)].init_state.unique().tolist()
    to_keep = [i for i in un13 if i not in un2]

    df13_clean = df[(df["annealing_time"] == 12.88888888888889) & (df["init_state"].isin(to_keep))]
    print(len(df13_clean))

    #print(df.annealing_time.unique().tolist())

