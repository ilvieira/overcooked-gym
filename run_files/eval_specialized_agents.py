from agents.distilled_plastic_policy.teammate_model import TeammateModel
from environment.overcooked import LAYOUTS, Overcooked, StatesAsFramesWrapper
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack
from agents.networks.overcooked_d_q_network import OvercookedDQNetwork
from dqn.policies.e_greedy import EGreedy
from agents.dqn_agents.overcooked_dqn_agent import OvercookedDQNAgent
import os
import time
import random as rnd
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates the specialized agents to obtain an upper bound for the "
                                                 "performance of the ad hoc agent.")
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    # begin training -------------------------------------------------------------------------------------------------
    save_dir = os.getcwd() + f"/data/round{args.round}/specialized_agents_stats/" if args.data_dir is None else args.data_dir
    seed = args.seed
    if seed is None:
        seed = int(time.time() * 100000) % 100000
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/random_seed.txt", "w") as seed_value:
        seed_value.write(str(seed))

    rnd.seed(seed)
    np.random.seed(seed)

    layout = "simple_kitchen"
    teammates = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                 ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)]

    eta = 0.25
    teammates_names = ["fetcher", "jack_of_all_trades", "cook_n_server", "clockwise_jack", "counterclockwise_jack"]

    # load the models learned for each teammate and the respective specialized agent
    teammates_models = []
    teammates_specialized_agents_dirs = [os.getcwd() + f"/data/round{args.round}/teammate{teammate_id}/best"
                                         for teammate_id in range(5)]

    for teammate_id in range(5):
        model_path = os.getcwd() + f"/data/round{args.round}/teammates_models/model{teammate_id}.pickle"
        teammates_models.append(TeammateModel.load(model_path))

    teammates_envs = []
    teammates_specialized_agents = []
    for idx, teammate in enumerate(teammates):
        # load the single agent perspective of the environment with the teammate
        env = Overcooked(layout=layout, max_timesteps=500, rewards=(0, 0, 0, 1))
        teammate_env = StatesAsFramesWrapper(env, teammate)
        teammates_envs.append(teammate_env)

        # load the agent that was trained to play with this teammate
        teammate_specialized_agent = OvercookedDQNAgent.load(teammate_env, teammates_specialized_agents_dirs[idx],
                                                             OvercookedDQNetwork, import_replay=False, policy=EGreedy(epsilon=0))
        teammates_specialized_agents.append(teammate_specialized_agent)

    n_trials = 100
    episodes_per_trial = 1
    episode_rewards_file_name = save_dir + "/episode_rewards.csv"

    # create the stats files
    with open(episode_rewards_file_name, "w") as episode_rewards_file:
        label_line = "trial,teammate_id" + "".join([f",reward_ep{str(i)}" for i in range(0, episodes_per_trial)])
        episode_rewards_file.write(label_line)

    # begin the trials
    st = time.time()
    for i in range(n_trials):
        print(f"Trial {i}")
        teammate_index = rnd.randrange(len(teammates_names))
        print(f"current teammate: {teammates_names[teammate_index]}")
        teammate_env = teammates_envs[teammate_index]
        agent = teammates_specialized_agents[teammate_index]
        r = [agent.play(render=False) for _ in range(episodes_per_trial)]

        # store the statistics needed for the evaluation plots
        with open(episode_rewards_file_name, "a") as episode_rewards_file:
            line = f"\n{i},{teammate_index}" + "".join([f",{str(i)}" for i in r])
            episode_rewards_file.write(line)

        print(f"...................................{time.time()-st}s")