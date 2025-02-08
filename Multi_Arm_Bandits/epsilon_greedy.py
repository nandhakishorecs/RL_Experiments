import numpy as np
from typing import List

from .base_policy import BasePolicy

class EpsilonGreedyPolicy(BasePolicy): 
    __slots__ = '_epsilon', '_arm_IDs', '_Q', '_n_triggers_arm'
    def __init__(self,  epsilon: float, arm_IDs: List[int]) -> None:
        self._epsilon = epsilon 
        self._arm_IDs = arm_IDs
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}
        self._Q = {id: 0 for id in self._arm_IDs}

    @property
    def policyName_(self) -> str: 
        return f'Epsilon Greedy (Epsilon:{self._epsilon})'
    
    @property
    def __name__(self) -> str: 
        return 'Epsilon Greedy Policy'
    
    def reset_arms(self) -> None:
        self._Q = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}

    def update_arm(self, arm_ID: int, arm_reward: float) -> None: 
        step_size = 1/(self._n_triggers_arm[arm_ID] + 1)
        self._Q[arm_ID] += step_size * (arm_reward - self._Q[arm_ID])
        self._n_triggers_arm[arm_ID] += 1

    def select_arm(self) -> int:
        selected_arm = np.random.rand() 
        if(selected_arm < self._epsilon): 
            return np.random.choice(self._arm_IDs)
        else: 
            return int(max(self._Q, key = self._Q.get))