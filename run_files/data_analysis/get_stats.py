import pandas as pd
import numpy as np
import math
from scipy.stats import norm


def get_distilled_pp_episode_rewards(round):
    file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{round}/distilled_plastic_stats/episode_rewards.csv"
    return pd.read_csv(file)


def get_pp_episode_rewards(round):
    file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{round}/plastic_stats/episode_rewards.csv"
    return pd.read_csv(file)


def get_sa_episode_rewards(round):
    file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{round}/specialized_agents_stats/episode_rewards.csv"
    return pd.read_csv(file)


def get_distillation_test_results(round):
    file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{round}/specialized_agents_vs_distilled_agent_stats/distilled_agent_episode_rewards.csv"
    return pd.read_csv(file)


def get_dqn_test_results(round):
    file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{round}/specialized_agents_vs_distilled_agent_stats/specialized_agent_episode_rewards.csv"
    return pd.read_csv(file)


def get_avg_distilled_pp_prob_correct_team():
    round_vals =[]
    for r in range(1, 11):
        file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{r}/distilled_plastic_stats/prob_of_correct_teammate.csv"
        round_vals.append(pd.read_csv(file, header=None))
    values = pd.concat(round_vals, ignore_index=True).values
    return np.average(values, axis=0)


def get_avg_distilled_pp_prob_correct_team_is_max():
    round_vals =[]
    for r in range(1, 11):
        file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{r}/distilled_plastic_stats/prob_correct_teammate_is_max.csv"
        round_vals.append(pd.read_csv(file, header=None))
    values = pd.concat(round_vals, ignore_index=True).values
    return np.average(values, axis=0)


def get_avg_pp_prob_correct_team():
    round_vals =[]
    for r in range(1, 11):
        file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{r}/plastic_stats/prob_of_correct_teammate.csv"
        round_vals.append(pd.read_csv(file, header=None))
    values = pd.concat(round_vals, ignore_index=True).values
    return np.average(values, axis=0)


def get_avg_pp_prob_correct_team_is_max():
    round_vals =[]
    for r in range(1, 11):
        file = f"C:/Users/inesl/Documents/Thesis/Code/Overcooked/overcooked-gym/data/round{r}/plastic_stats/prob_correct_teammate_is_max.csv"
        round_vals.append(pd.read_csv(file, header=None))
    values = pd.concat(round_vals, ignore_index=True).values
    return np.average(values, axis=0)


def p_value_mean_difference(sample1, sample2):
    """Computes the p-value for testing:
    H0: The mean of the distribution from sample1 is the same as the mean of the distribution from sample2
    H1: the two means are different

    This is the test used for when the standards deviations are known. We can use it with an empyrical estimator of the
    variance with very large samples, as it is done in this function.

    THEREFORE IT IS ASSUMED THAT THE SAMPLE SIZE IS REASONABLY LARGE when this function is used
    (around 40 samples or more for each distribution)
    """
    mean1 = np.average(sample1)
    std1 = np.std(sample1)
    n1 = math.prod(sample1.shape)
    mean2 = np.average(sample2)
    std2 = np.std(sample2)
    n2 = math.prod(sample2.shape)

    z0 = (mean1 - mean2) / math.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    p_val = 2 * (1 - norm.cdf(abs(z0)))
    return p_val


def cohens_d_effect_size(sample1, sample2):
    mean1 = np.average(sample1)
    std1 = np.std(sample1)
    mean2 = np.average(sample2)
    std2 = np.std(sample2)

    z0 = (mean1 - mean2) / math.sqrt((std1 ** 2 + std2 ** 2) / 2)
    return abs(z0)