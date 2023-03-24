import os
import pickle
import dwave.inspector
import networkx as nx
import numpy as np
import pandas as pd
import dwave_networkx as dnx
import matplotlib.pyplot as plt

from scipy import optimize
from tqdm import tqdm
from copy import deepcopy
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding
from collections import OrderedDict

rng = np.random.default_rng()
cwd = os.getcwd()


def odd(x): return 2 * x + 1


def test_graph(g: nx.Graph):
    plt.figure(figsize=(32, 32))
    dnx.draw_chimera(g, with_labels=True)
    plt.show()


def test_embedding(t: nx.Graph):
    embedding = find_embedding(chain, t)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
    sampleset = sampler.sample_ising(h, J)
    dwave.inspector.show(sampleset)


def pseudo_likelihood(beta_eff, samples):
    J = - 1.0
    L = 0.0
    N = samples.shape[1]
    D = samples.shape[0]
    for i in range(D-1):
        for j in range(N-1):
            if j == 0:
                L += -np.log(1+np.exp(-2*(J*samples[i, j]*samples[i, j+1])*beta_eff))
            elif j == N-1:
                L += -np.log(1+np.exp(-2*(J*samples[i, j]*samples[i, j-1])*beta_eff))
            else:
                L += -np.log(1+np.exp(-2*(J*samples[i, j]*samples[i, j+1]+J*samples[i, j]*samples[i, j-1])*beta_eff))
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


def energy(s: np.ndarray, h: np.ndarray, J: np.ndarray): return np.dot(np.dot(s, J), s) + np.dot(s, h)


if __name__ == "__main__":

    # Setup
    qpu_sampler = DWaveSampler(solver='DW_2000Q_6', token="DEV-ea8fe294d81c576d653e3e925574c4f9b9fa13dc")
    target = qpu_sampler.to_networkx_graph()
    middle_label = 1023

    # First Quadrant
    Q1 = deepcopy(target)
    for node in target.nodes:
        if node > middle_label or any([True if (node in range(64 * odd(i), 128 * (i + 1))) else False for i in range(8)]):
            Q1.remove_node(node)

    # Second Quadrant
    Q2 = deepcopy(target)
    for node in target.nodes:
        if node > middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8)]):
            Q2.remove_node(node)

    # Third Quadrant
    Q3 = deepcopy(target)
    for node in target.nodes:
        if node <= middle_label or any(
                [True if (node in range(64 * odd(i), 128 * (i+1))) else False for i in range(8, 16)]):
            Q3.remove_node(node)

    # Fourth Quadrant
    Q4 = deepcopy(target)
    for node in target.nodes:
        if node <= middle_label or any([True if (node in range(i * 128, 64 * odd(i))) else False for i in range(8, 16)]):
            Q4.remove_node(node)

    # Experiment setup
    chain_length = 300
    h = {i: 0 for i in range(chain_length)}
    J = {(i, i + 1): 1 for i in range(chain_length - 1)}
    h_vect, J_vect = vectorize(h, J)
    chain = nx.Graph(J.keys())

    max_time = 100
    num_samples = 100
    anneal_param = 0.41
    beta = 1
    gibbs_num_steps = 10 ** 4
    quadrant = Q2
    quadrant_name = "Q2"

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

    raw_data_pause = pd.DataFrame(columns=["init_state", "sample", "energy", "num_occurrences", "anneal_length"])
    raw_data_reverse = pd.DataFrame(columns=["init_state", "sample", "energy", "num_occurrences", "anneal_length"])

    for anneal_length in np.linspace(2, max_time, num=10):
        E_fin_pause = []
        configurations_pause = []
        Q_pause = []

        E_fin_reverse = []
        configurations_reverse = []
        Q_reverse = []

        for i in tqdm(range(num_samples), desc=f"samples for tau {anneal_length:.2f}"):
            initial_state = dict(gibbs_sampling_ising(h, J, beta, gibbs_num_steps))
            init_state = np.array(list(initial_state.values()))
            # Energy per spin
            E_init = (np.dot(init_state, np.dot(J_vect, init_state)) + np.dot(h_vect, init_state)) / chain_length
            # due to random nature of finding embedding, sometimes this function fails to find an embedding. Running it
            # again should work in the vast majority of cases.
            try:
                embedding = find_embedding(chain, quadrant, tries=1000)
            except ValueError:
                embedding = find_embedding(chain, quadrant, tries=100000)
            sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
            
            anneal_schedule_pausing = [[0, 1], [anneal_length * 1 / 3, anneal_param], 
                                       [anneal_length * 2 / 3, anneal_param], [anneal_length, 1]]
            anneal_schedule_reverse = [[0, 1], [anneal_length * 1 / 2, anneal_param], [anneal_length, 1]]

            sampleset_pausing = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                     anneal_schedule=anneal_schedule_pausing,
                                                     num_reads=10)
            sampleset_reverse = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                     anneal_schedule=anneal_schedule_reverse,
                                                     num_reads=10)

            for sample in sampleset_pausing.record:
                final_state = np.array(sample.sample)
                E_fin_pause.append(sample.energy / chain_length)  # per spin
                configurations_pause.append(final_state)
                Q_pause.append(sample.energy / chain_length - E_init)

            for sample in sampleset_reverse.record:
                final_state = np.array(sample.sample)
                E_fin_reverse.append(sample.energy / chain_length)  # per spin
                configurations_reverse.append(final_state)
                Q_reverse.append(sample.energy / chain_length - E_init)

            df_pause = sampleset_pausing.to_pandas_dataframe(sample_column=True)
            df_pause["init_state"] = [initial_state for _ in range(len(df_pause))]
            df_pause["anneal_length"] = [anneal_length for _ in range(len(df_pause))]
            raw_data_pause = pd.concat([raw_data_pause, df_pause], ignore_index=True)
            raw_data_pause.to_csv(os.path.join(cwd, "results", "raw_data",
                                  f"raw_data_{quadrant_name}_pausing_tau_{anneal_length:.2f}.csv"), sep=";")

            df_reverse = sampleset_reverse.to_pandas_dataframe(sample_column=True)
            df_reverse["init_state"] = [initial_state for _ in range(len(df_reverse))]
            df_reverse["anneal_length"] = [anneal_length for _ in range(len(df_reverse))]
            raw_data_reverse = pd.concat([raw_data_reverse, df_reverse], ignore_index=True)
            raw_data_reverse.to_csv(os.path.join(cwd, "results", "raw_data",
                                    f"raw_data_{quadrant_name}_reverse_tau_{anneal_length:.2f}.csv"), sep=";")

        optim_pause = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations_pause),))
        beta_eff_pause[anneal_length] = (optim_pause.x)
        mean_E_pause[anneal_length] = (np.mean(np.array(E_fin_pause)))
        var_E_pause[anneal_length] = (np.var(np.array(E_fin_pause)))
        mean_Q_pause[anneal_length] = (np.mean(np.array(Q_pause)))
        var_Q_pause[anneal_length] = (np.var(np.array(Q_pause)))
        
        optim_reverse = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations_reverse),))
        beta_eff_reverse[anneal_length] = (optim_reverse.x)
        mean_E_reverse[anneal_length] = (np.mean(np.array(E_fin_reverse)))
        var_E_reverse[anneal_length] = (np.var(np.array(E_fin_reverse)))
        mean_Q_reverse[anneal_length] = (np.mean(np.array(Q_reverse)))
        var_Q_reverse[anneal_length] = (np.var(np.array(Q_reverse)))

        with open(os.path.join(cwd, "results", "checkpoints", f"checkpoint_{quadrant_name}_pausing.pkl"), "wb") as f:
            pickle.dump([beta_eff_pause, mean_E_pause, var_E_pause, mean_Q_pause, var_Q_pause], f)
        with open(os.path.join(cwd, "results", "checkpoints", f"checkpoint_{quadrant_name}_reverse.pkl"), "wb") as f:
            pickle.dump([beta_eff_reverse, mean_E_reverse, var_E_reverse, mean_Q_reverse, var_Q_reverse], f)


    with open(os.path.join("results", f"results_pausing_{quadrant_name}.pkl"), "wb") as f:
        pickle.dump([beta_eff_pause, mean_E_pause, var_E_pause, mean_Q_pause, var_Q_pause], f)

    with open(os.path.join("results", f"results_reverse_{quadrant_name}.pkl"), "wb") as f:
        pickle.dump([beta_eff_reverse, mean_E_reverse, var_E_reverse, mean_Q_reverse, var_Q_reverse], f)
