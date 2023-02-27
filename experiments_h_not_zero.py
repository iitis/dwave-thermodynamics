import copy
import os
import numpy as np
from scipy import optimize
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd

import dimod
from dimod.reference import samplers
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding
from tqdm import tqdm
from collections import OrderedDict
import dwave.inspector

rng = np.random.default_rng()
cwd = os.getcwd()


def pseudo_likelihood(beta_eff, samples):
    J = - 1.0
    L = 0.0
    N = samples.shape[1]
    D = samples.shape[0]
    for i in range(D-1):
        for j in range(N-1):
            if j == 0:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1])*beta_eff))
            elif j == N-1:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j-1])*beta_eff))
            else:
                L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1]+J*samples[i,j]*samples[i,j-1])*beta_eff))
    return -L/(N*D)


def gibbs_sampling_ising(h: dict, J: dict, beta: float, num_steps: int):
    h = OrderedDict(sorted(h.items()))
    J = OrderedDict(sorted(J.items()))
    s = OrderedDict({i: rng.choice([-1, 1]) for i in h.keys()})
    h_vect, J_vect = vectorize(h, J)
    nodes = list(h.keys())

    for _ in range(num_steps):
        pos = rng.choice(nodes)  # we chose an index

        s_plus = np.array(list(s.values()))
        s_plus[pos] = 1
        s_minus = np.array(list(s.values()))
        s_minus[pos] = -1

        deltaE = energy(s_plus, h_vect, J_vect) - energy(s_minus, h_vect, J_vect)
        prob = 1/(1+np.exp(beta*deltaE))  # P(s_i = 1| s_-i)
        s[pos] = rng.choice([-1, 1], p=[1-prob, prob])
    return s


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


def energy(s: np.ndarray, h: np.ndarray, J: np.ndarray):
    return np.dot(np.dot(s, J), s) + np.dot(s, h)


if __name__ == "__main__":

    chain_length = 300
    # Hamiltonian per spin
    #H = (np.diag(np.ones(chain_length - 1), -1) + np.diag(np.ones(chain_length - 1), 1)) / chain_length

    qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
    J = {(i, i + 1): 1.0 for i in range(chain_length - 1)}
    embedding = find_embedding(J, qpu_sampler.edgelist)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

    max_anneal_length = 200
    max_h = 1
    num_samples = 100
    anneal_param = 0.41
    #beta = 1
    gibbs_num_steps = 10**4

    for beta in [1]:
        mean_E_pause = {}
        var_E_pause = {}
        mean_Q_pause = {}
        var_Q_pause = {}
        beta_eff_pause = {}

        mean_E_reverse = {}
        var_E_reverse = {}
        mean_Q_reverse = {}
        var_Q_reverse = {}
        beta_eff_reverse = {}
        tic = time.time()
        # vary in time
        for anneal_length in np.linspace(2, max_anneal_length, num=10):
            for h_val in np.linspace(0.1, max_h, num=10):
                h = {i: h_val for i in range(chain_length)}
                J = {(i, i + 1): 1.0 for i in range(chain_length - 1)}
                h_vect, J_vect = vectorize(h, J)

                E_fin_pause = []
                configurations_pause = []
                Q_pause = []

                E_fin_reverse = []
                configurations_reverse = []
                Q_reverse = []

                raw_data_pause = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
                raw_data_reverse = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
                for i in tqdm(range(num_samples), desc=f"samples for tau {anneal_length:.2f} h {h_val}"):
                    initial_state = dict(gibbs_sampling_ising(h, J, beta, gibbs_num_steps))
                    init_state = np.array(list(initial_state.values()))

                    E_init = (np.dot(init_state, np.dot(J_vect, init_state)) + np.dot(h_vect, init_state))/chain_length  # per spin

                    anneal_schedule_pausing = [[0, 1], [anneal_length * 1/3, anneal_param], [anneal_length * 2/3, anneal_param],
                                       [anneal_length, 1]]
                    anneal_schedule_reverse = [[0, 1], [anneal_length * 1/2, anneal_param], [anneal_length, 1]]

                    sampleset_pausing = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                   anneal_schedule=anneal_schedule_pausing,
                                                   num_reads=10)
                    sampleset_reverse = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                   anneal_schedule=anneal_schedule_reverse,
                                                   num_reads=10)

                    for sample in sampleset_pausing.record:
                        final_state = np.array(sample.sample)
                        E_fin_pause.append(sample.energy/chain_length)  # per spin
                        configurations_pause.append(final_state)
                        Q_pause.append(sample.energy/chain_length - E_init)

                    for sample in sampleset_reverse.record:
                        final_state = np.array(sample.sample)
                        E_fin_reverse.append(sample.energy/chain_length)  # per spin
                        configurations_reverse.append(final_state)
                        Q_reverse.append(sample.energy/chain_length - E_init)

                    df_pause = sampleset_pausing.to_pandas_dataframe(sample_column=True)
                    df_pause["init_state"] = [initial_state for _ in range(len(df_pause))]
                    raw_data_pause = pd.concat([raw_data_pause, df_pause], ignore_index=True)
                    raw_data_pause.to_csv(os.path.join(cwd, "results\\raw_data",
                                                 f"raw_data_pausing_tau_{anneal_length:.2f}_beta_{beta}_h{h_val}_s{anneal_param}.csv"),
                                    sep=";")

                    df_reverse = sampleset_reverse.to_pandas_dataframe(sample_column=True)
                    df_reverse["init_state"] = [initial_state for _ in range(len(df_reverse))]
                    raw_data_reverse = pd.concat([raw_data_reverse, df_reverse], ignore_index=True)
                    raw_data_reverse.to_csv(os.path.join(cwd, "results\\raw_data",
                                                 f"raw_data_reverse_tau_{anneal_length:.2f}_beta_{beta}_h{h_val}_s{anneal_param}.csv"),
                                    sep=";")

                optim_pause = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations_pause),))
                beta_eff_pause[(anneal_length, h_val)] = (optim_pause.x)
                mean_E_pause[(anneal_length, h_val)] = (np.mean(np.array(E_fin_pause)))
                var_E_pause[(anneal_length, h_val)] = (np.var(np.array(E_fin_pause)))
                mean_Q_pause[(anneal_length, h_val)] = (np.mean(np.array(Q_pause)))
                var_Q_pause[(anneal_length, h_val)] = (np.var(np.array(Q_pause)))

                optim_reverse = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations_reverse),))
                beta_eff_reverse[(anneal_length, h_val)] = (optim_reverse.x)
                mean_E_reverse[(anneal_length, h_val)] = (np.mean(np.array(E_fin_reverse)))
                var_E_pause[(anneal_length, h_val)] = (np.var(np.array(E_fin_pause)))
                mean_Q_pause[(anneal_length, h_val)] = (np.mean(np.array(Q_pause)))
                var_Q_reverse[(anneal_length, h_val)] = (np.var(np.array(Q_reverse)))

                with open(os.path.join(cwd, "results\\checkpoints", f"checkpoints_h{h_val}_pausing.pkl"), "wb") as f:
                    pickle.dump([beta_eff_pause, mean_E_pause, var_E_pause, mean_Q_pause, var_Q_pause], f)
                with open(os.path.join(cwd, "results\\checkpoints", f"checkpoints_h{h_val}_reverse.pkl"), "wb") as f:
                    pickle.dump([beta_eff_reverse, mean_E_reverse, var_E_reverse, mean_Q_reverse, var_Q_reverse], f)

            toc = time.time()
            with open(os.path.join("results", f"results_pausing_s{anneal_param}_h.pkl"), "wb") as f:
                pickle.dump([beta_eff_pause, mean_E_pause, var_E_pause, mean_Q_pause, var_Q_pause], f)

            with open(os.path.join("results", f"results_reverse_s{anneal_param}_h.pkl"), "wb") as f:
                pickle.dump([beta_eff_reverse, mean_E_reverse, var_E_reverse, mean_Q_reverse, var_Q_reverse], f)
            print('Thermalization at s =', anneal_param, 'completed.\n')
            print('Elapsed time:', toc - tic, 's.\n')

