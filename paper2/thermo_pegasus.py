
import os
import numpy as np
from scipy import optimize
import pickle
import time
import pandas as pd

import dimod

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding
from tqdm import tqdm

# import dwave.inspector

from utils import energy, vectorize, gibbs_sampling_ising, pseudo_likelihood

rng = np.random.default_rng()
cwd = os.getcwd()

ANNEAL_LENGTH = 200
NUM_SAMPLES = 1000
GIBBS_NUM_STEPS = 10 ** 4

if __name__ == "__main__":
    chain_length = 300
    h = {i: 0 for i in range(chain_length)}
    J = {(i, i + 1): rng.uniform(-1, 1) for i in range(chain_length - 1)}
    h_vect, J_vect = vectorize(h, J)
    with open("instance.pkl", "wb") as f:
        l = [h, J]
        pickle.dump(l, f)

    qpu_sampler = DWaveSampler(token="OBi2-bf11ab4b1a5f98d4d14ea244a5f25e048d6f764c")
    embedding = find_embedding(J, qpu_sampler.edgelist)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

    mean_E = {}
    var_E = {}
    mean_Q = {}
    var_Q = {}
    beta_eff = {}

    for pause_duration in [0, 20, 40, 60, 80, 100]:
        for anneal_param in np.linspace(0, 1, num=25):

            E_fin = []
            configurations = []
            Q = []
            raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
            for i in tqdm(range(NUM_SAMPLES), desc=f"samples for pause duration {pause_duration:.2f} s {anneal_param:.2f}"):
                initial_state = dict(gibbs_sampling_ising(h, J, 1, GIBBS_NUM_STEPS))
                init_state = np.array(list(initial_state.values()))

                E_init = energy(init_state, h_vect, J_vect)/chain_length   # per spin

                anneal_schedule = [[0, 1], [ANNEAL_LENGTH * 1 / 2 - pause_duration/2, anneal_param],
                                   [ANNEAL_LENGTH * 1 / 2 + pause_duration/2, anneal_param],
                                   [ANNEAL_LENGTH, 1]] if pause_duration != 0 else [[0, 1],
                                                                                    [ANNEAL_LENGTH/2, anneal_param],
                                                                                    [ANNEAL_LENGTH, 1]]

                sampleset = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                 anneal_schedule=anneal_schedule,
                                                 num_reads=10, auto_scale=False, reinitialize_state=True)

                for s in sampleset.samples():
                    final_state = np.array(list(s.values()))
                    E_fin.append(energy(final_state, h_vect, J_vect)/chain_length)
                    configurations.append(final_state)
                    Q.append(energy(final_state, h_vect, J_vect)/chain_length - E_init)

                df = sampleset.to_pandas_dataframe(sample_column=True)
                df["init_state"] = [initial_state for _ in range(len(df))]
                raw_data = pd.concat([raw_data, df], ignore_index=True)
                raw_data.to_csv(os.path.join(cwd, "..\\results\\raw_data\\pegasus_thermo",
                                             f"raw_data_pegasus_thermo_{pause_duration}_{anneal_param:.2f}.csv"),
                                sep=";")

            optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
            beta_eff[(pause_duration, anneal_param)] = optim.x
            mean_E[(pause_duration, anneal_param)] = np.mean(np.array(E_fin))
            var_E[(pause_duration, anneal_param)] = np.var(np.array(E_fin))
            mean_Q[(pause_duration, anneal_param)] = np.mean(np.array(Q))
            var_Q[(pause_duration, anneal_param)] = np.var(np.array(Q))

            with open(os.path.join(cwd, "..\\results\\checkpoints", f"checkpoints_pegasus_thermo.pkl"), "wb") as f:
                pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)

    with open(os.path.join(cwd, "..", "results", f"results_pegasus_thermo.pkl"), "wb") as f:
        pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)

    print('Thermalization at completed')


