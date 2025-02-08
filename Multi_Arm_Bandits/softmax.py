import numpy as np
from typing import List

from .base_policy import BasePolicy

class SoftmaxPolicy(BasePolicy): 
    __slots__ = '_tau', '_arm_IDs', '_Q', '_n_triggers_arm'
    def __init__(self, tau: float, arm_IDs: List) -> None:
        self._tau = tau 
        self._arm_IDs = arm_IDs
        self._Q = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}

    @property
    def policyName_(self) -> str: 
        return f'Softmax Policy: (Tau = {self._tau})'
    
    def reset_arms(self):
        self._Q = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}

    def update_arm(self, arm_ID: int, arm_reward: float) -> None:
        step_size = 1/(self._n_triggers_arm[arm_ID] + 1)
        self._Q[arm_ID] += step_size * (arm_reward - self._Q[arm_ID])
        self._n_triggers_arm[arm_ID] += 1

    def select_arm(self) -> int: 
        pi = [] 
        Q_tau = np.exp(np.array(list(self._Q.values()))/self._tau)
        Q_tau[Q_tau == np.inf] = 1 

        for i in range(len(self._arm_IDs)): 
            pi.append(Q_tau[i]/np.sum(Q_tau))

        return int(np.random.choice(self._arm_IDs, 1, p = pi))