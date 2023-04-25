import pandas as pd
import os
import numpy as np

cwd = os.getcwd()


def clean_pausing_with_h():
    new_df = pd.DataFrame()
    for h in np.linspace(0.1, 1.0, num=10):
        h = round(h, 1)
        df = pd.read_csv(os.path.join(cwd, f"results/raw_data/pausing__beta_1_h{h}_s0.41.csv"),
                         sep=";")
        df["h"] = [h for _ in df.iterrows()]
        new_df = pd.concat([new_df, df], ignore_index=True)
    new_df.to_csv(os.path.join(cwd, f"results/raw_data/pausing_with_h_beta_1_s_0.41.csv"), sep=";", index=False)


if __name__ == "__main__":
    clean_pausing_with_h()