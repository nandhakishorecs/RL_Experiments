'''
Author: Nandhakishore C S
'''
import numpy as np 
import pandas as pd
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")

from .base_policy import BasePolicy
from .random_policy import RandomPolicy
from .greedy import GreedyPolicy
from .epsilon_greedy import EpsilonGreedyPolicy
from .softmax import SoftmaxPolicy
from .UCB import UpperConfidenceBond

from .environment import Environment
from .utils import * 

if __name__ == '__main__': 
    env = Environment(
        arm_type = 'gaussian', 
        n_arms = 10, 
        mean_reward_range = (-3, 3), 
        std_dev = 1.0, 
        n_samples = 2_000
    )
    
    print('\nMulti Arm Test bed to test Random, Greedy, Epsilon Greedy, Softmax and UCB policies\n')

    seed = 42 
    np.random.seed(seed)

    # Reward distribution for Multi-Arm Testbed
    rewawrd_distribution = env.reward_distribution() 
    rewawrd_distribution = np.array(rewawrd_distribution)

    # Getting violin plot to see the reward distribution for actions
    actions_rewards = env._create_arms() 
    actions = np.array([*actions_rewards.keys()])

    data = dict(zip(actions, rewawrd_distribution))
    df = pd.DataFrame(data)
    df_melted = df.melt(var_name = 'Action', value_name = 'Reward Distribution') 

    plt.figure(figsize=(8, 6))
    sns.violinplot(x = 'Action', y = 'Reward Distribution', data = df_melted, inner = 'box', palette = 'grey')
    plt.axhline(y = (env._reward_range[1] + env._reward_range[0])/2, color = "black", linestyle = "dashed", linewidth = 2) 
    plt.title(f'{env._n_arms} - Arm Test Bed')
    plt.show()

    # Random Policy 
    random_policy = RandomPolicy(env.Arms)
    plot_reward_regret(env, [random_policy], time_steps = 200, n_runs = 500)
    
    # Greedy Policy
    greedy_policy = GreedyPolicy(env.Arms)
    plot_reward_regret(env, [greedy_policy], time_steps = 200, n_runs = 500)

    # Epsilon Greedy Policy 
    explore_epgreedy_epsilons =  [0.001, 0.01, 0.5, 0.9]
    e_greedy_policies = [EpsilonGreedyPolicy(ep, env.Arms) for ep in explore_epgreedy_epsilons]
    plot_reward_regret(env, e_greedy_policies, time_steps = 200, n_runs = 500)

    # Softmax Policy 
    explore_softmax_taus =  [0.001, 1.0, 5.0, 50.0]
    softmax_polices = [SoftmaxPolicy(tau, env.Arms) for tau in explore_softmax_taus]
    plot_reward_regret(env, softmax_polices, time_steps=200, n_runs=500)

    # UCB 
    explore_c_UCBs =  [0.0001, 0.001, 0.01, 1]
    UCB_polices = [UpperConfidenceBond(env.Arms, c) for c in explore_c_UCBs]
    plot_reward_regret(env, UCB_polices , time_steps=200, n_runs=500)