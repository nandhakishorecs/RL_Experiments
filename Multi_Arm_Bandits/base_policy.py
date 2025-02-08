class BasePolicy:

    def reset_arms(self): 
        # reset all the arms 
        pass

    def update_arm(self, *args) -> None: 
        # keep track of the estimates for the given policy 
        pass

    def select_arm(self) -> int: 
        # return id of the selected arm 
        raise Exception('Not Implemted')
    
    @property 
    def policyName_(self) -> str: 
        return 'Base Policy'