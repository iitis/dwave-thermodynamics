import pickle
import numpy as np
import pandas as pd
import os
from scipy import optimize
from tqdm import tqdm
from utils import energy, vectorize, gibbs_sampling_ising, pseudo_likelihood

rng = np.random.default_rng()
cwd = os.getcwd()


def calculate_from_raw_data(h_vect, J_vect, chain_length, df: pd.DataFrame):
    init_states = df.init_state.unique().tolist()
    E_fin = []
    configurations = []
    Q = []
    for state in init_states:
        temp_df = df[df["init_state"] == state]
        state = eval(state)
        init_state = np.array(list(state.values()))
        E_init = energy(init_state, h_vect, J_vect) / chain_length
        samples = temp_df["sample"].tolist()
        for s in samples:
            s = eval(s)
            final_state = np.array(list(s.values()))
            E_fin.append(energy(final_state, h_vect, J_vect) / chain_length)
            configurations.append(final_state)
            Q.append(energy(final_state, h_vect, J_vect) / chain_length - E_init)
    return E_fin, configurations, Q


if __name__ == '__main__':
    with open(os.path.join(cwd, "paper2", "instance.pkl"), "rb") as f:
        h, J = pickle.load(f)
    h_vect, J_vect = vectorize(h, J)
    chain_length = 300
    mean_E = {}
    var_E = {}
    mean_Q = {}
    var_Q = {}
    beta_eff = {}
    for anneal_param in tqdm(np.linspace(0, 1, num=25), desc="Anneal parameters"):
        if anneal_param < 0.38 or anneal_param > 0.74:
            continue
        s = f"{anneal_param:.2f}"

        path = os.path.join(cwd, "results", "raw_data", "pegasus_thermo", f"raw_data_pegasus_thermo_0_{s}.csv")
        raw_data = pd.read_csv(path, sep=";", index_col=0)

        E_fin, configurations, Q = calculate_from_raw_data(h_vect, J_vect, chain_length, raw_data)

        optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
        beta_eff[(0, anneal_param)] = optim.x
        mean_E[(0, anneal_param)] = np.mean(np.array(E_fin))
        var_E[(0, anneal_param)] = np.var(np.array(E_fin))
        mean_Q[(0, anneal_param)] = np.mean(np.array(Q))
        var_Q[(0, anneal_param)] = np.var(np.array(Q))

        with open(os.path.join(cwd, "results\\checkpoints", f"checkpoints_pegasus_thermo_0_rest.pkl"), "wb") as f:
            pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)
