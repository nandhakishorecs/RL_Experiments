import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Arm: 
    __slots__ = '_mean', '_std_dev', '_type', '_n', '_p'
    def __init__(self, type: str = 'normal', p1: float = None, p2: float = None):
        if(type == 'gaussian' or type == 'normal'): 
            self._type = type
            self._mean = p1 
            self._std_dev = p2 
            self._p = self._n = None
        # Yet to test
        # elif(type == 'binomial'): 
        #     self._type = type
        #     self._n = int(p1) 
        #     if(p2 <= 1 and p2 >= 0): 
        #         self._p = p2
        #     else: 
        #         raise ValueError('\nProbability parameter for Bionomial Distribution cannot be more than 1 or less than 0\n')
        #     self._mean = self._std_dev = None
        elif(type == 'exponential'): 
            self._type = type
            self._mean = self._std_dev = p1
            self._p = self._n = None

class Environment: 
    __slots__ = '_n_arms', '_reward_range', '_std_dev', '_arms', '_arm_type', '_n_samples'
    def __init__(self, arm_type:str = 'gaussian', n_arms: int = 10, mean_reward_range: tuple = (-10, 10), std_dev: float = 0.25, n_samples:int = 1_000):
        self._arm_type = arm_type
        self._n_arms = n_arms
        self._reward_range = mean_reward_range
        self._std_dev = std_dev 
        self._n_samples = n_samples
        self._arms = self._create_arms()

    def _create_arms(self) -> dict: 
        low_reward, high_reward = self._reward_range
        # sampling the initial rewards from a uniform distribution and then simulating uniform as a gaussian 
        # this can be directly done from gaussian as well - # yet to implement 
        means = np.random.uniform(
            low = low_reward, 
            high = high_reward, 
            size = (self._n_arms,)
        )
        arms = {id: Arm(self._arm_type, mu, self._std_dev) for id, mu in enumerate(means)}
        return arms

    def step(self, arm_ID:int): 
        choosen_arm = self._arms[arm_ID]
        if(self._arm_type == 'gaussian' or self._arm_type == 'normal'): 
            # print('ARM',choosen_arm._mean, choosen_arm._std_dev) - test code
            return np.random.normal(choosen_arm._mean, choosen_arm._std_dev)
        # Yet to test
        # elif(self._arm_type == 'binomial'): 
        #     return np.random.binomial(choosen_arm._n, choosen_arm._p, size = 100)
        elif(self._arm_type == 'exponential'): 
            # print('ARM',choosen_arm._mean, choosen_arm._std_dev) - test code
            # scale parameter is 1/mean for exponential distribution 
            return np.random.exponential(scale = 1 / choosen_arm._mean)
    
    def best_arm(self) -> tuple:
        # best arm has high expected reward over a given number of time steps 
        arm_ID = max(self._arms, key = lambda i: self._arms[i]._mean)
        return arm_ID, float(self._arms[arm_ID]._mean)

    def average_reward(self) -> float : 
        # average reward over all the arms over all time steps 
        return np.mean(np.array([i._mean for i in self._arms.values()]))

    @property 
    def Arms(self):
        return list(self._arms.keys()) 
    
    def reward_distribution(self):
        rewards = [] 
        _, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (9, 5))
        colors = sns.color_palette("hls", self._n_arms)
        for i, arm_ID in enumerate(self.Arms):
            reward_samples = [self.step(arm_ID) for _ in range(self._n_samples)]
            rewards.append(reward_samples)
            # plt.hist(reward_samples, bins = 100, label = f'arm_{arm_ID + 1}')
            sns.histplot(reward_samples, ax = ax, stat = 'density', color=colors[i], kde = True, bins = 100, label = f'arm_{arm_ID + 1}')
            plt.title(f'{self._n_arms} Arm Testbed: Rewards vs Desnity Plot')
            plt.ylabel('Reward Distribution')
            plt.xlabel('Desnity')
        ax.legend(loc = 'lower right')
        plt.grid()
        plt.savefig('Arm_reward_distribution.png', dpi = 100) 
        plt.show()
        return rewards