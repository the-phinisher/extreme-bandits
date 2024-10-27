import numpy as np
from tqdm import tqdm
from utils import rd_argmax, Chernoff_Interval, second_order_Pareto_UCB
from bisect import insort


class TrackerMax:
    def __init__(
        self,
        nb_arms,
        T,
        store_rewards_arm=False,
        store_max_rewards_arm=False,
        store_sorted_rewards_arm=False,
        store_all_sorted_rewards=False,
        store_maxima=False,
    ):
        self.nb_arms = nb_arms
        self.T = T
        self.store_rewards_arm = store_rewards_arm
        self.store_max_rewards_arm = store_max_rewards_arm
        self.store_sorted_rewards_arm = store_sorted_rewards_arm
        self.store_all_sorted_rewards = store_all_sorted_rewards
        self.store_maxima = store_maxima
        self.current_max = -np.inf
        self.max_arms = np.zeros(self.nb_arms)
        self.reset()

    def reset(self):
        self.Na = np.zeros(self.nb_arms, dtype="int")
        self.t = 0
        if self.store_rewards_arm:
            self.rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_max_rewards_arm:
            self.max_rewards_arm = [-np.inf for _ in range(self.nb_arms)]
        if self.store_sorted_rewards_arm:
            self.sorted_rewards_arm = [[] for _ in range(self.nb_arms)]
        if self.store_all_sorted_rewards:
            self.all_sorted_rewards = []
        if self.store_maxima:
            self.maxima = dict(
                zip(np.arange(self.nb_arms), [{} for _ in range(self.nb_arms)])
            )
            self.n = np.zeros(self.nb_arms, dtype=np.int32)
            self.nb_batch = np.zeros(self.nb_arms, dtype=np.int32)
            self.qomax = np.inf * np.ones(self.nb_arms)

    def update(self, t, arm, reward):
        self.Na[arm] += 1
        self.t = t
        if self.store_rewards_arm:
            self.rewards_arm[arm].append(reward)
        if self.store_max_rewards_arm:
            if reward > self.max_rewards_arm[arm]:
                self.max_rewards_arm[arm] = reward
        if self.store_sorted_rewards_arm:
            insort(self.sorted_rewards_arm[arm], reward)
        if self.store_all_sorted_rewards:
            insort(self.all_sorted_rewards, reward)
        if self.current_max < reward:
            self.current_max = reward
        if self.max_arms[arm] < reward:
            self.max_arms[arm] = reward


class ArmPareto:
    def __init__(self, alpha, C, random_state=0):
        self.alpha = alpha
        self.scale = C ** (1 / alpha)
        self.local_random = np.random.RandomState(random_state)

    def sample(self, size=1):
        return (self.local_random.pareto(self.alpha, size) + 1) * self.scale


class GenericMAB:
    """
    Generic class to simulate an Extreme Bandit problem
    """

    def __init__(self, p):
        self.params = p
        self.MAB = self.generate_arms(p)
        self.nb_arms = len(self.MAB)
        self.mc_regret = None

    @staticmethod
    def generate_arms(p):
        arms_list = list()
        for i in range(len(p)):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            alg = ArmPareto
            arms_list.append(alg(*args))
        return arms_list

    def MC(self, method, N, T, param_dic):
        mc_count = np.zeros(self.nb_arms)
        all_counts = np.zeros((N, self.nb_arms))
        all_maxima = np.zeros(N)
        alg = self.__getattribute__(method)
        for i in tqdm(range(N), desc="Computing " + str(N) + " simulations"):
            tr = alg(T, **param_dic)
            mc_count += tr.Na / N
            all_counts[i] = tr.Na
            all_maxima[i] = tr.current_max
        return mc_count, all_counts, all_maxima

    def Threshold_Ascent(self, T, s, delta):
        tr = TrackerMax(
            self.nb_arms, T, store_all_sorted_rewards=True, store_rewards_arm=True
        )
        alpha = np.log(2 * self.nb_arms * T / delta)
        threshold = -np.inf
        for t in range(T):
            if t > s:
                former_thresh = threshold
                threshold = tr.all_sorted_rewards[-int(s)]
                if threshold != former_thresh:
                    S = [
                        np.array(tr.rewards_arm[k])[
                            np.array(tr.rewards_arm[k]) >= threshold
                        ].shape[0]
                        for k in range(self.nb_arms)
                    ]
            else:
                S = tr.Na
            Idx = np.array(
                [
                    Chernoff_Interval(S[k] / (tr.Na[k] + 1e-9), tr.Na[k], alpha)
                    for k in range(self.nb_arms)
                ]
            )
            arm = rd_argmax(Idx)
            reward = self.MAB[arm].sample()[0]
            tr.update(t, arm, reward)
        return tr

    def ExtremeHunter(
        self, T, b=100, D=1e-4, E=1e-4, N=None, r=None, steps=1, delta=0.1
    ):
        tr = TrackerMax(self.nb_arms, T, store_sorted_rewards_arm=True)
        if r is None:
            r = T ** (-1 / (2 * b + 1))
        if N is None:
            N = np.log(T) ** ((2 * b + 1) / b)
        if delta == "theoretic":
            delta = np.exp(-np.log(T) ** 2) / (2 * T * self.nb_arms)
        t = 0
        while t < T:
            if t < self.nb_arms * N:
                arm = t % self.nb_arms
                tr.update(t, arm, self.MAB[arm].sample()[0])
                t += 1
            else:
                arm = second_order_Pareto_UCB(tr, b, D, E, delta, r)
                nb_pulls = min(T - t, steps)
                for _ in range(nb_pulls):
                    tr.update(t, arm, self.MAB[arm].sample()[0])
                    t += 1
        return tr
