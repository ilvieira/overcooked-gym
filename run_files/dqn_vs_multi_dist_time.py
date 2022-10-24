from environment.overcooked import LAYOUTS, Overcooked, StatesAsFramesWrapper
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack
from agents.networks.overcooked_d_q_network import OvercookedDQNetwork
from dqn.policies.e_greedy import EGreedy
from agents.dqn_agents.overcooked_dqn_agent import OvercookedDQNAgent
from dqn.agents.multi_distilled_agent import MultiDistilledAgent
from agents.networks.overcooked_multi_distillation_net import OvercookedMultiDistillationNetwork
from dqn.memory.distillation_replay_memory import DistillationReplayMemory
import os
import time
import random as rnd
import argparse
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obtains the forward time of running one DQN at time to obtain the "
                                                 "estimates for each specialized agent with the time of doing that with "
                                                 "the Multi-Task Distilled agent")
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_datapoints", type=int, default=100_000)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = os.getcwd() + f"/data/forward_time_round{args.round}_{args.n_datapoints}_{args.device}/" if args.data_dir is None else args.data_dir
    seed = args.seed
    if seed is None:
        seed = int(time.time() * 100000) % 100000
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/random_seed.txt", "w") as seed_value:
        seed_value.write(str(seed))

    rnd.seed(seed)
    np.random.seed(seed)
    device = args.device
    layout = "simple_kitchen"
    teammates = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                 ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)]

    # import the environments and the DQN agents
    teammates_dqn_agents_dirs = [os.getcwd() + f"/data/round{args.round}/teammate{teammate_id}/best"
                                         for teammate_id in range(5)]
    teammates_envs = []
    teammates_dqns = []
    for idx, teammate in enumerate(teammates):
        # load the single agent perspective of the environment with the teammate
        env = Overcooked(layout=layout, max_timesteps=500, rewards=(0, 0, 0, 1))
        teammate_env = StatesAsFramesWrapper(env, teammate)
        teammates_envs.append(teammate_env)

        # load the agent that was trained to play with this teammate
        teammate_specialized_agent = OvercookedDQNAgent.load(teammate_env, teammates_dqn_agents_dirs[idx],
                                                             OvercookedDQNetwork, import_replay=False,
                                                             policy=EGreedy(epsilon=0), device=device)

        teammates_dqns.append(teammate_specialized_agent.Q)

    # import the distilled agent
    distilled_agent = MultiDistilledAgent.load(teammates_envs,
                                               [None for _ in range(5)],
                                               os.getcwd() + f"/data/round{args.round}/distilled_agent/",
                                               OvercookedMultiDistillationNetwork,
                                               import_replay=False,
                                               replay_type=DistillationReplayMemory, 
                                               device=device)
    distilled_agent_network = distilled_agent.net

    data_st = time.time()
    # collect some states
    states = []
    env = teammates_envs[0]  # the teammate at this stage is irrelevant since the states are sampled randomly
    for _ in range(args.n_datapoints):
        state = env.random_state()
        rgb_frame = env.env.render_state(state)
        preprocessed_frame = torch.tensor(np.expand_dims(env.preprocess(rgb_frame), axis=0)).float()
        states.append(preprocessed_frame.to(device))
    print(f"time to store the data: {time.time()-data_st}")

    dqn_st = time.time()
    for state in states:
        for dqn in teammates_dqns:
            dqn.forward(state)

    dqn_time = time.time() - dqn_st

    dist_st = time.time()
    for state in states:
        distilled_agent_network.forward_for_all_tasks(state)
    dist_time = time.time() - dist_st

    with open(save_dir + "/times_comparison.txt", "w") as times_file:
        times_file.write(f"DQNs -> {dqn_time}s")
        times_file.write(f"\nDistilled Network -> {dist_time}s")
