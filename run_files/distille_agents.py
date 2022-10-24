from environment.overcooked import LAYOUTS, Overcooked, StatesAsFramesWrapper
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack
from agents.networks.overcooked_d_q_network import OvercookedDQNetwork
from agents.networks.overcooked_multi_distillation_net import OvercookedMultiDistillationNetwork
from dqn.policies.e_greedy import EGreedy
from agents.dqn_agents.overcooked_dqn_agent import OvercookedDQNAgent
import os
from dqn.agents.multi_distilled_agent import MultiDistilledAgent
from dqn.memory.distillation_replay_memory import DistillationReplayMemory
import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs multi-task distillation in order to compress the specialized"
                                                 " agents trained with DQNs into a single distilled agent for the "
                                                 "current round. The DQN agents for the current round need to already "
                                                 "be trained for this script to work properly.")
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # begin training -------------------------------------------------------------------------------------------------
    save_dir = os.getcwd() + f"/data/round{args.round}/distilled_agent"
    seed = args.seed
    if seed is None:
        seed = int(time.time() * 100000) % 100000
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/random_seed.txt", "w") as seed_value:
        seed_value.write(str(seed))

    layout = "simple_kitchen"
    teammates_names = ["fetcher", "jack_of_all_trades", "cook_n_server", "clockwise_jack", "counterclockwise_jack"]
    teammates = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                 ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)]
    teachers_dirs = [os.getcwd() + f"/data/round{args.round}/teammate{teammate_id}/best"
                                         for teammate_id in range(5)]
    envs = []
    teachers = []
    for idx, teammate in enumerate(teammates):
        # load the single agent perspective of the environment with the teammate
        env = Overcooked(layout=layout, max_timesteps=500, rewards=(0, 0, 0, 1))
        teammate_env = StatesAsFramesWrapper(env, teammate)
        envs.append(teammate_env)

        # load the agent that was trained to play with this teammate
        teammate_specialized_agent = OvercookedDQNAgent.load(teammate_env, teachers_dirs[idx],
                                                             OvercookedDQNetwork, import_replay=False, policy=EGreedy(epsilon=0.05))
        teachers.append(teammate_specialized_agent)

    student = MultiDistilledAgent(envs, teachers, OvercookedMultiDistillationNetwork,
                                  {"number_of_actions": [5,5,5,5,5]}, policy=EGreedy(0),
                                  memory_type=DistillationReplayMemory, memory_size=1_000_000, temperature=0.01,
                                  seed=seed, optimizer_parameters={"lr": 0.001, "alpha": 0.95, "eps": 0.01})

    st = time.time()
    student.learn(save_dir=save_dir, save_replay=True, verbose=True,
                  max_epochs=100, updates_per_episode=500, frames_per_episode=10_000)


    with open(save_dir + "/training_time.txt", "w") as time_value:
        time_value.write(str(time.time()-st))