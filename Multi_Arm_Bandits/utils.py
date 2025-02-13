import numpy as np
import matplotlib.pyplot as plt

from .environment import Environment
from .base_policy import BasePolicy

def train(env: Environment, policy: BasePolicy, time_steps: int) -> np.array: 
    policy_reward = np.zeros((time_steps, ))
    for t in range(time_steps): 
        arm_ID = policy.select_arm() 
        reward = env.step(arm_ID)
        policy.update_arm(arm_ID, reward)
        policy_reward[t] = reward
    return policy_reward 

def get_average_over_runs(env: Environment, policy: BasePolicy, time_steps: int, n_runs: int) -> np.array:
    _, expected_max_reward = env.best_arm() 
    policy_reward_per_run = np.zeros((n_runs, time_steps)) 
    for run in range(n_runs): 
        policy.reset_arms() 
        policy_reward = train(env, policy, time_steps)
        policy_reward_per_run[run, :] = policy_reward
    
    average_policy_reward = np.mean(policy_reward_per_run, axis = 0)
    total_policy_regret = np.sum(expected_max_reward - average_policy_reward)

    return average_policy_reward, total_policy_regret

def plot_reward_regret(env: Environment, policies: BasePolicy, time_steps = 200, n_runs = 500):
    _, ax = plt.subplots(1, 1, sharex = False, sharey = False, figsize = (10, 6))
    for policy in policies: 
        filename = policy.__name__
        average_policy_reward, total_policy_regret  = get_average_over_runs(env, policy, time_steps, n_runs)
        print('Regret for {}:\t {:.3f}'.format(policy.policyName_, total_policy_regret))
        ax.plot(np.arange(time_steps), average_policy_reward, '-', label = policy.policyName_)
    print('\n')

    _, expected_max_reward = env.best_arm()
    ax.plot(np.arange(time_steps), [expected_max_reward] * time_steps, 'g-')

    average_arm_reward = env.average_reward() 
    ax.plot(np.arange(time_steps), [average_arm_reward] * time_steps, 'r-')
    plt.xlabel('Time Steps')
    plt.ylabel('Rewards')   
    plt.title(f'{env._n_arms} - Arm Test Bed using {filename}')
    plt.legend(loc='lower right')
    filename += '.png'
    plt.savefig(filename, dpi = 100)
    plt.show() 