import pandas as pd
import os
from tqdm import tqdm
import pickle

cwd = os.getcwd()

prob = {}
distance = {}
Q = 4
way = "reverse"
data = f"Q4_pausing_v2"

df = pd.read_csv(os.path.join(cwd, "results", "raw_data", f"{data}.csv"), sep=";")
#df.drop(["Unnamed: 0"], axis=1, inplace=True)


if __name__ == "__main__":
    h = 0.1
    for tau in df.annealing_time.unique().tolist():
        p = 0
        c = 0
        s = 0
        count = 0
          # & (df["h"] == h) _h_{h} <- stuff for computing with h
        for init_state in tqdm(df[(df["annealing_time"] == tau) ].init_state.unique().tolist(), desc=f"{tau} "):

            temp_df = df[(df["annealing_time"] == tau) & (df["init_state"] == init_state)]
            c += 1
            if -299 in temp_df["energy"].values:
                p += 1
            s += sum([(row[1]["energy"]/-299) * row[1]["num_occurrences"] for row in temp_df.iterrows()])/len(temp_df)

        print(f"probability for tau {tau}: ", p/c)
        print(f"closeness for tau {tau}: ", s/c)
        prob[tau] = p/c
        distance[tau] = s/c

    with open(os.path.join(cwd, "results", f"results_eff_{data}.pkl"), "wb") as f:
        l = [prob, distance]
        pickle.dump(l, f)
