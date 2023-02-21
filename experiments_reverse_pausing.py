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
    h = {i: 0 for i in range(chain_length)}
    J = {(i, i + 1): 1.0 for i in range(chain_length-1)}
    h_vect, J_vect = vectorize(h, J)

    # Hamiltonian per spin
    H = (np.diag(np.ones(chain_length - 1), -1) + np.diag(np.ones(chain_length - 1), 1)) / chain_length

    qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
    embedding = find_embedding(J, qpu_sampler.edgelist)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

    max_anneal_length = 200
    num_samples = 1000
    anneal_param = 0.5
    #beta = 1
    gibbs_num_steps = 10**4

    for beta in [1, 0.1]:

        mean_E_therm = []
        var_E_therm = []
        mean_E = []
        var_E = []
        mean_Q = []
        var_Q = []
        beta_eff = []
        tic = time.time()
        # vary in time
        for anneal_length in np.linspace(1, max_anneal_length, num=10):
            E_fin = []
            configurations = []
            Q = []
            raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
            for i in tqdm(range(num_samples), desc=f"samples for tau {anneal_length:.2f} beta {beta}"):
                initial_state = dict(gibbs_sampling_ising(h, J, beta, gibbs_num_steps))
                init_state = np.array(list(initial_state.values()))

                E_init = np.dot(init_state, np.dot(H, init_state))  # per spin

                anneal_schedule = [[0, 1], [anneal_length * 1/3, anneal_param], [anneal_length * 2/3, anneal_param],
                                   [anneal_length, 1]]

                sampleset = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                               anneal_schedule=anneal_schedule,
                                               num_reads=10)

                for s in sampleset.samples():
                    final_state = np.array(list(s.values()))
                    E_fin.append(np.dot(final_state, np.dot(H, final_state)))
                    configurations.append(final_state)
                    E_fin.append(np.dot(final_state, np.dot(H, final_state)))
                    Q.append(np.dot(final_state, np.dot(H, final_state)) - E_init)

                df = sampleset.to_pandas_dataframe(sample_column=True)
                df["init_state"] = [initial_state for _ in range(len(df))]
                raw_data = pd.concat([raw_data, df], ignore_index=True)
                raw_data.to_csv(os.path.join(cwd, "results\\raw_data",
                                             f"raw_data_pausing_tau_{anneal_length:.2f}_beta_{beta}_s{anneal_param}.csv"),
                                sep=";")

            mean_E_therm.append(np.mean(np.array(E_fin)))
            var_E_therm.append(np.var(np.array(E_fin)))
            optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
            beta_eff.append(optim.x)
            mean_E.append(np.mean(np.array(E_fin)))
            var_E.append(np.var(np.array(E_fin)))
            mean_Q.append(np.mean(np.array(Q)))
            var_Q.append(np.var(np.array(Q)))

            with open(os.path.join(cwd, "results\\checkpoints", f"checkpoints_s{anneal_param}_pausing.pkl"), "wb") as f:
                pickle.dump([mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q], f)

        toc = time.time()
        with open(os.path.join("results", f"results_pausing_s{anneal_param}_beta{beta}.pkl"), "wb") as f:
            pickle.dump([mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q], f)
        print('Thermalization at s =', anneal_param, 'completed.\n')
        print('Elapsed time:', toc - tic, 's.\n')

