import numpy as np
from typing import List

from .base_policy import BasePolicy

class UpperConfidenceBond(BasePolicy): 
    __slots__ = '_arm_IDs', '_c', '_r', '_total_reward', '_n_triggers_arm', '_upper_bound'
    def __init__(self, arm_IDs: List[int], c:float = 0.1):
        self._arm_IDs = arm_IDs
        self._total_reward = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}
        self._c = c 
        self._r = 0  
        self._upper_bound = {} 

    @property
    def policyName_(self) -> str: 
        return f'Upper Confidence Bound (UCB): c = {self._c}'

    def reset_arms(self) -> None:
        self._total_reward = {id: 0 for id in self._arm_IDs}
        self._n_triggers_arm = {id: 0 for id in self._arm_IDs}
        self._upper_bound = {id: 0 for id in self._arm_IDs}

    def update_arm(self, arm_ID:int, arm_reward: float) -> None:
        self._total_reward[arm_ID] += arm_reward
        self._n_triggers_arm[arm_ID] += 1 
        average_reward_per_arm = self._total_reward[arm_ID]
        
        # Implementing UCB equation form TextBook 
        N = sum(self._n_triggers_arm.values())
        self._upper_bound[arm_ID] = average_reward_per_arm + (self._c * np.sqrt(np.log(N)/self._n_triggers_arm[arm_ID]))
        self._r = average_reward_per_arm
    
    def select_arm(self):
        selected_arm = False 

        for arm_ID in self._arm_IDs: 
            if(self._upper_bound[arm_ID] > self._r): 
                selected_arm = True 
                return int(arm_ID)
            elif(self._upper_bound[arm_ID] == 0): 
                selected_arm = True
                return int(arm_ID)

        if(not selected_arm): 
            return int(np.random.choice(self._arm_IDs)) 