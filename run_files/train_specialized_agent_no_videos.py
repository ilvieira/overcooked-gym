from agents.dqn_agents.overcooked_dqn_agent import OvercookedDQNAgent
from dqn.memory.dqn_replay_memory import DQNReplayMemory
from agents.networks.overcooked_d_q_network import OvercookedDQNetwork
from dqn.policies.train_eval_policy import TrainEvalPolicy, EGreedy, EGreedyLinearDecay
import torch
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack

from torch.nn.functional import mse_loss
import os

from environment.overcooked import Overcooked, LAYOUTS, StatesAsFramesWrapper
import argparse
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a DQN to obtain a policy for acting with the specified "
                                                 "teammate in the overcooked environment but does not store videos of "
                                                 "the training process.")
    parser.add_argument("--teammate_id", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # begin training -------------------------------------------------------------------------------------------------
    save_dir = os.getcwd() + f"/data/round{args.round}/teammate{args.teammate_id}"
    seed = args.seed
    if seed is None:
        seed = int(time.time() * 100000) % 100000
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + "/random_seed.txt", "w") as seed_value:
        seed_value.write(str(seed))

    layout = "simple_kitchen"
    teammates = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                 ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)]
    C = 10_000 if args.teammate_id != 2 else 500

    rmsprop = True
    decay_steps = 1_000_000
    optimizer = torch.optim.RMSprop if rmsprop else torch.optim.Adam
    lr = 0.00025
    optimizer_parameters = {"lr": lr, "alpha": 0.95, "eps": 0.01} if rmsprop else {"lr": lr}
    minibatch = 32
    gamma = 0.99
    teammate = teammates[args.teammate_id]
    train_rewards = (0, 1, 1, 1)
    eval_rewards = (0, 0, 0, 1)

    train_env = Overcooked(layout=layout, max_timesteps=500, rewards=train_rewards)
    train_env = StatesAsFramesWrapper(train_env, teammate)

    eval_env = Overcooked(layout=layout, max_timesteps=500, rewards=eval_rewards)
    eval_env = StatesAsFramesWrapper(eval_env, teammate)

    replay = DQNReplayMemory(1_000_000)

    policy = TrainEvalPolicy(eval_policy=EGreedy(epsilon=0),
                             train_policy=EGreedyLinearDecay(epsilon=1, min_epsilon=0.1,
                                                             steps_of_decay=decay_steps))

    agent = OvercookedDQNAgent(train_env, replay, train_env.num_actions, OvercookedDQNetwork,
                               {"number_of_actions": train_env.num_actions}, policy=policy, loss=mse_loss,
                               minibatch_size=minibatch, C=C, update_frequency=1, gamma=gamma,
                               optimizer_parameters={"lr": lr, "alpha": 0.95, "eps": 0.01},
                               avg_loss_per_steps=10_000, seed=seed, eval_env=eval_env,
                               save_eval_videos=False)

    # Create agent and populate the replay memory
    agent.populate_replay_memory(50_000)

    st = time.time()

    # Learning stage
    agent.learn(save_dir, save_after_steps=50_000, eval_after_steps=20_000, max_steps=2_000_000, max_time=3*24*3600,
                feedback_after_episodes=100)

    with open(save_dir + "/training_time.txt", "w") as time_value:
        time_value.write(str(time.time()-st))