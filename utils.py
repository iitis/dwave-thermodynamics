import numpy as np
from collections import OrderedDict

rng = np.random.default_rng()

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

