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
import dwave.inspector


if __name__ == "__main__":

    chain_length = 300
    h = {i: 0 for i in range(chain_length - 1)}
    J = {(i, i + 1): 1.0 for i in range(chain_length - 1)}

    H = -(-np.diag(np.ones(chain_length - 1), -1) - np.diag(np.ones(chain_length - 1), 1)) / chain_length

    qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
    embedding = find_embedding(J, qpu_sampler.edgelist)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

    mean_E_therm = []
    var_E_therm = []

    anneal_length = 200
    num_samples = 1000
    # anneal_param = 1

    for anneal_param in [1, 0.99]:
        tic = time.time()
        for anneal_length in np.linspace(1, 200, num=10):
            E_fin = []
            for i in tqdm(range(num_samples)):
                init_state = np.random.choice([-1, 1], size=(chain_length,), p=[1 / 2, 1 / 2])
                E_init = np.dot(init_state, np.dot(H, init_state))
                initial_state = dict(enumerate(init_state.tolist()))

                # init_state = sample_boltzmann(beta)
                # initial_state = init_state.first.sample
                # # init_state = np.ones(chain_lenght)
                # E_init = init_state.first.energy
                anneal_schedule = [[0, 1], [anneal_length * anneal_param, anneal_param], [anneal_length, 1]] \
                    if anneal_param != 1 else [[0, 1], [200, 1]]

                samples = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                               anneal_schedule=anneal_schedule,
                                               num_reads=10)

                for s in samples.samples():
                    final_state = np.array(list(s.values()))
                    E_fin.append(np.dot(final_state, np.dot(H, final_state)))
            mean_E_therm.append(np.mean(np.array(E_fin)))
            var_E_therm.append(np.var(np.array(E_fin)))
            with open(f"checkpoints_s{anneal_param}.pkl", "wb") as f:
                pickle.dump([mean_E_therm, var_E_therm], f)
        toc = time.time()
        with open(f"average_spin_energy_s{anneal_param}.pkl", "wb") as f:
            pickle.dump([mean_E_therm, var_E_therm], f)
        print('Thermalization at s =', anneal_param, 'completed.\n')
        print('Elapsed time:', toc - tic, 's.\n')
    #
    # mean_E = []
    # var_E = []
    # mean_Q = []
    # var_Q = []
    # beta_eff = []
    #
    # tic = time.time()
    # # to save QPU time you can just do one value of anneal_param instead af sweeping the full range
    # anneal_length = 100 # redefinition
    # for anneal_param in np.linspace(0.1, 0.9, num=5):
    #     configurations = []
    #     E_fin = []
    #     Q = []
    #     for i in tqdm(range(num_samples)):
    #         # Initial state at infinite temperature
    #         init_state = sample_boltzmann(beta)
    #         initial_state = init_state.first.sample
    #         # init_state = np.ones(chain_lenght)
    #         E_init = init_state.first.energy
    #         # Reverse annealing
    #         samples = sampler.sample_ising(h=h, J=J,  initial_state=initial_state,
    #                                  anneal_schedule=[[0, 1], [anneal_length / 2, anneal_param], [anneal_length, 1]],
    #                                  num_reads=10)
    #         for s in samples.samples():
    #             final_state = np.array(list(s.values()))
    #             configurations.append(final_state)
    #             E_fin.append(np.dot(final_state, np.dot(H, final_state)))
    #             Q.append(np.dot(final_state, np.dot(H, final_state)) - E_init)
    #     optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
    #     beta_eff.append(optim.x)
    #     mean_E.append(np.mean(np.array(E_fin)))
    #     var_E.append(np.var(np.array(E_fin)))
    #     mean_Q.append(np.mean(np.array(Q)))
    #     var_Q.append(np.var(np.array(Q)))
    #     with open("checkpoints_beta1_s.pkl", "wb") as f:
    #         pickle.dump([mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q], f)
    # toc = time.time()
    #
    # print('Collected proper work statistics.')
    # print('Elapsed time:', toc - tic, 's.\n')
    #
    # # if you want to save the experimental data
    # with open('experimental_results_betabig.pkl', 'wb') as f:
    #     pickle.dump([mean_E_therm, var_E_therm, beta_eff, mean_E, var_E, mean_Q, var_Q], f)
    #
    # print('Saved data into ->', f.name)
