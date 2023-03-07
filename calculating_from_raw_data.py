import pickle
import numpy as np
import pandas as pd
import os
from scipy import optimize
from tqdm import tqdm


def pseudo_likelihood(beta_eff, samples):
    J = - 1.0
    L = 0.0
    N = samples.shape[1]
    D = samples.shape[0]
    for i in tqdm(range(D-1), desc="i"):
        for j in range(N-1):
            if j == 0:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1])*beta_eff))
            elif j == N-1:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j-1])*beta_eff))
            else:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1]+J*samples[i,j]*samples[i,j-1])*beta_eff))
    return -L/(N*D)

cwd = os.getcwd()


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\average_spin_energy_s0.5_beta0.1.pkl",
          "rb") as f:
    mean_E, var_E = pickle.load(f)
    mean_E = [e/2 for e in mean_E]

mean_Q = []
var_Q = []
beta_eff = []
chain_length = 300
h = {i: 0 for i in range(chain_length)}
J = {(i, i + 1): 1.0 for i in range(chain_length - 1)}
h_vect, J_vect = vectorize(h, J)

for t in [1.0, 23.11, 45.22, 67.33, 89.44, 111.55, 133.66, 155.77, 177.88, 200.0]:
    E_fin = []
    configurations = []
    Q = []
    df = pd.read_csv(os.path.join(cwd, f"results\\raw_data\\raw_data_reverse_tau_{t}_beta_0.1_s0.5.csv"),
                     sep=";", index_col=0)
    for init in tqdm(df.init_state.unique().tolist(), desc=f"{t}"):
        samples = df[df.init_state == init]

        initial_state = eval(init)
        init_state = np.array(list(initial_state.values()))
        E_init = (np.dot(init_state, np.dot(J_vect, init_state)))/chain_length

        for sample in samples.itertuples(index=False):
            s = eval(sample.sample)
            final_state = np.array(list(s.values()))
            assert s == {i: final_state[i] for i in range(300)}  # sanity check
            E_fin.append(sample.energy/chain_length)
            configurations.append(final_state)
            Q.append(sample.energy/chain_length - E_init)

    optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
    beta_eff.append(optim.x)
    mean_Q.append(np.mean(np.array(Q)))
    var_Q.append(np.var(np.array(Q)))
    with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\checkpoints\\"
              "checkpoint_result_reverse_s0.5_beta0.1_200.pkl", "wb") as f:
        pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)


with open("C:\\Users\\tsmierzchalski\\PycharmProjects\\dwave-thermodynamics\\results\\results_reverse_s0.5_beta0.1.pkl",
          "rb") as f:
    pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)
