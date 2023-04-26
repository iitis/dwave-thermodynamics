import pandas as pd
import os
import numpy as np
from tqdm import tqdm

cwd = os.getcwd()


def clean_pausing_with_h():
    new_df = pd.DataFrame()
    for h in np.linspace(0.1, 1.0, num=10):
        h = round(h, 1)
        df = pd.read_csv(os.path.join(cwd, "results", "raw_data", f"pausing__beta_1_h{h}_s0.41.csv"),
                         sep=";")
        df["h"] = [h for _ in df.iterrows()]
        new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, f"results", "raw_data", "pausing_with_h_beta_1_s_0.41.csv"), sep=";", index=False)


def clean_reverse_with_h():
    new_df = pd.DataFrame()
    for h in np.linspace(0.1, 1.0, num=10):
        h = round(h, 1)
        for tau in np.linspace(2, 200, num=10):
            if tau > 156:
                continue
            df = pd.read_csv(os.path.join(cwd, "results", "raw_data",
                                          f"raw_data_reverse_tau_{tau:.2f}_beta_1_h{h}_s0.41.csv"), sep=";")
            df["h"] = [h for _ in df.iterrows()]
            df["annealing_time"] = [tau for _ in df.iterrows()]
            new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, "results", "raw_data", f"reverse_with_h_beta_1_s_0.41.csv"), sep=";", index=False)


def clean_quadrant(q, annealing):
    new_df = pd.DataFrame()
    for tau in np.linspace(2, 100, num=10):
        df = pd.read_csv(os.path.join(cwd, "results", "raw_data", f"raw_data_Q{q}_{annealing}_tau_{tau:.2f}.csv"),
                         sep=";")
        df["annealing_time"] = [tau for _ in df.iterrows()]
        new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, "results", "raw_data", f"Q{q}_{annealing}.csv"),
                  sep=";", index=False)


def clean_pausing(beta):
    new_df = pd.DataFrame()
    for tau in np.linspace(1.5, 200, num=10):
        df = pd.read_csv(os.path.join(cwd, f"results", "raw_data",
                                      f"raw_data_pausing_tau_{tau:.2f}_beta_{beta}_s0.5.csv"), sep=";")
        df["annealing_time"] = [tau for _ in df.iterrows()]
        new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, "results", "raw_data", f"pausing_beta_{beta}_s_0.5.csv"), sep=";", index=False)


def clean_reverse(beta):
    new_df = pd.DataFrame()
    for tau in np.linspace(1, 200, num=10):
        df = pd.read_csv(os.path.join(cwd, f"results", "raw_data",
                                      f"raw_data_reverse_tau_{tau:.2f}_beta_{beta}_s0.5.csv"), sep=";")
        df["annealing_time"] = [tau for _ in df.iterrows()]
        new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, "results", "raw_data", f"reverse_beta_{beta}_s_0.5.csv"), sep=";", index=False)


if __name__ == "__main__":
    clean_reverse_with_h()
