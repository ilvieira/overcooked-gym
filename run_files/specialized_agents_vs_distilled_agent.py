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
from dqn.memory.distillation_replay_memory import DistillationReplayMemory
from dqn.agents.multi_distilled_agent import MultiDistilledAgent
from agents.networks.overcooked_multi_distillation_net import OvercookedMultiDistillationNetwork
import os
import time
import random as rnd
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a DQN to obtain a policy for acting with the specified "
                                                 "teammate in the overcooked environment")
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    # begin training -------------------------------------------------------------------------------------------------
    save_dir = os.getcwd() + f"/data/round{args.round}/specialized_agents_vs_distilled_agent_stats/" if args.data_dir is None else args.data_dir
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

    # load and the specialized agent learned for each teammate
    teammates_specialized_agents_dirs = [os.getcwd() + f"/data/round{args.round}/teammate{teammate_id}/best"
                                         for teammate_id in range(5)]

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

    distilled_agent = MultiDistilledAgent.load(teammates_envs,
                                               [None for _ in range(5)],
                                               os.getcwd() + f"/data/round{args.round}/distilled_agent/",
                                               OvercookedMultiDistillationNetwork,
                                               import_replay=False,
                                               replay_type=DistillationReplayMemory)
    distilled_agent.seed = seed

    n_trials = 1
    episodes_per_trial = 100
    episode_rewards_specialized_file = save_dir + "/specialized_agent_episode_rewards.csv"
    episode_rewards_distilled_file = save_dir + "/distilled_agent_episode_rewards.csv"

    # create the files to store the stats
    with open(episode_rewards_specialized_file, "w") as episode_rewards_file:
        label_line = "trial,teammate_id" + "".join([f",reward_ep{str(i)}" for i in range(0, episodes_per_trial)])
        episode_rewards_file.write(label_line)
    with open(episode_rewards_distilled_file, "w") as episode_rewards_file:
        label_line = "trial,teammate_id" + "".join([f",reward_ep{str(i)}" for i in range(0, episodes_per_trial)])
        episode_rewards_file.write(label_line)

    # test each agent with one of the teammates
    for teammate_id in range(5):
        print(f"current teammate: {teammates_names[teammate_id]}")
        specialized_agent = teammates_specialized_agents[teammate_id]
        distilled_agent.select_task(teammate_id)
        teammate_env = teammates_envs[teammate_id]

        # begin the trials
        st = time.time()
        for i in range(n_trials):
            print(f"trial {i} began")
            r_specialized_agent = [specialized_agent.play(render=False) for _ in range(episodes_per_trial)]
            r_distilled_agent = [distilled_agent.play(render=False) for _ in range(episodes_per_trial)]

            # store the statistics needed for the evaluation plots
            with open(episode_rewards_specialized_file, "a") as episode_rewards_file:
                line = f"\n{i},{teammate_id}" + "".join([f",{str(i)}" for i in r_specialized_agent])
                episode_rewards_file.write(line)
            with open(episode_rewards_distilled_file, "a") as episode_rewards_file:
                line = f"\n{i},{teammate_id}" + "".join([f",{str(i)}" for i in r_distilled_agent])
                episode_rewards_file.write(line)

        print(f"...................................{time.time()-st}s")