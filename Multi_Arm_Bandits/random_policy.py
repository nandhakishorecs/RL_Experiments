import numpy as np
from typing import List

from .base_policy import BasePolicy

class RandomPolicy(BasePolicy): 
    __slots__ = '_arm_IDs'
    def __init__(self, arm_IDs: List[int]) -> None:
        self._arm_IDs = arm_IDs

    @property 
    def policyName_(self) -> str: 
        return f'Random Policy'
    
    @property
    def __name__(self) -> str: 
        return 'Random Policy'
    
    def reset_arms(self):
        ''' no reseting of arms in random policy '''
        pass

    def update_arm(self, *args):
        ''' no updation in random policy '''
        pass

    def select_arm(self) -> int:
        return np.random.choice(self._arm_IDs)