import numpy as np
from numba import jit
from scipy.special import gamma

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@jit(nopython=True)
def rd_argmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)


def Chernoff_Interval(mu, n, alpha):
    if n == 0:
        return np.inf
    return mu + (alpha + np.sqrt(2*n*mu*alpha+alpha**2))/n


def second_order_Pareto_UCB(tr, b, D, E, delta, r):
    B1 = D * np.sqrt(np.log(1 / delta)) * tr.Na ** (-b / (2 * b + 1))
    B2 = E * np.sqrt(np.log(tr.Na / delta)) * np.log(tr.Na) * tr.Na ** (-b / (2 * b + 1))
    hk = np.zeros(tr.nb_arms)
    Ck = np.zeros(tr.nb_arms)
    Bk = np.zeros(tr.nb_arms)
    for k in range(tr.nb_arms):
        s = int(r * tr.Na[k])
        rwd = np.array(np.maximum(tr.sorted_rewards_arm[k], 1))
        hk[k] = np.log(rwd[-s:] / rwd[-s]).mean()
        Ck[k] = tr.Na[k] ** (1 / (2 * b + 1)) * (rwd >= tr.Na[k] ** (hk[k] / (2 * b + 1))).mean()
        if hk[k] + B1[k] >= 1:
            Bk[k] = np.inf
        else:
            Bk[k] = ((Ck[k] + B2[k]) * tr.T) ** (hk[k] + B1[k]) * gamma(1 - hk[k] - B1[k])
    return int(rd_argmax(Bk))