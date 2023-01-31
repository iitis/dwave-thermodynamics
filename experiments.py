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

chain_length = 300
h = {i: 0 for i in range(chain_length - 1)}
J = {(i, i+1): 1.0 for i in range(chain_length-1)}

H = -(-np.diag(np.ones(chain_length - 1), -1) - np.diag(np.ones(chain_length - 1), 1)) / chain_length

qpu_sampler = DWaveSampler(solver='DW_2000Q_6')
embedding = find_embedding(J,qpu_sampler.edgelist)
sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

num_samples = 10
anneal_length = 2000
anneal_param = 0.5


def reverse_anneal(num_samples, anneal_length, anneal_param):
    for i in tqdm(range(num_samples)):
        init_state = np.random.choice([-1, 1], size=(chain_length,), p=[1 / 2, 1 / 2])
        initial_state = dict(enumerate(init_state.tolist()))
        # init_state = np.ones(chain_lenght)
        E_init = np.dot(init_state, np.dot(H, init_state))

        samples = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                 anneal_schedule=[[0, 1], [anneal_length * anneal_param, anneal_param], [anneal_length, 1]],
                                 num_reads=300)

        df = samples.to_pandas_dataframe()
        row = {i: [init_state[i]] for i in range(chain_length-1)} | {"chain_break_fraction": [0], "energy": [E_init],
                                                                     "num_occurrences": [0]}
        row = pd.DataFrame.from_dict(row)
        df = df.sort_values("energy")
        df = pd.concat([df, row], ignore_index=True)

        df.to_csv(f"results/reverse/samples_{anneal_param}_{i}.csv")


def forward_anneal(num_samples):

    for i in tqdm(range(num_samples)):
        init_state = np.random.choice([-1, 1], size=(chain_length,), p=[1 / 2, 1 / 2])
        initial_state = dict(enumerate(init_state.tolist()))
        # init_state = np.ones(chain_lenght)
        E_init = np.dot(init_state, np.dot(H, init_state))

        samples = sampler.sample_ising(h=h, J=J, annealing_time=anneal_length, num_reads=300)

        df = samples.to_pandas_dataframe()
        row = {i: [init_state[i]] for i in range(chain_length - 1)} | {"chain_break_fraction": [0], "energy": [E_init],
                                                                       "num_occurrences": [0]}
        row = pd.DataFrame.from_dict(row)
        df = df.sort_values("energy")
        df = pd.concat([df, row], ignore_index=True)

        df.to_csv(f"results/forward/samples_{anneal_param}_{i}.csv")


if __name__ == "__main__":
    forward_anneal(num_samples)
