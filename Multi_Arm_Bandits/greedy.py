from typing import List

from .base_policy import BasePolicy

class GreedyPolicy(BasePolicy): 
    __slots__ = '_arm_IDs', '_Q', '_n_triggers_arm'
    def __init__(self, arm_IDs: List[int]) -> None:
        self._arm_IDs = arm_IDs
        self._Q = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}
    
    @property
    def policyName_(self) -> str: 
        return f'Greedy Policy'
    
    @property
    def __name__(self) -> str: 
        return 'Greedy Policy'

    def reset_arms(self):
        self._Q = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}

    def update_arm(self, arm_ID: int, arm_reward: float) -> None:
        step_size = 1/(self._n_triggers_arm[arm_ID] + 1)
        self._Q[arm_ID] += step_size * (arm_reward - self._Q[arm_ID])
        self._n_triggers_arm[arm_ID] += 1

    def select_arm(self) -> int:
        return int(max(self._Q, key = self._Q.get))