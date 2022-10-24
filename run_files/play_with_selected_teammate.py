from environment.overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from agents.teammates.counterclockwise_jack import CounterclockwiseJack
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.fetcher import Fetcher
from agents.teammates.clockwise_jack import ClockwiseJack
from environment.overcooked import ACTION_MEANINGS
from yaaf.agents import HumanAgent
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Allows a human agent to play with one of the teammates.")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument("--teammate_id", type=int)
    args = parser.parse_args()

    layout = "simple_kitchen"

    env = Overcooked(layout=layout, max_timesteps=args.max_timesteps, rewards=(0, 0, 0, 1))
    agent = HumanAgent(action_meanings=ACTION_MEANINGS, num_actions=5)
    teammate = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)][args.teammate_id]
    env = SingleAgentWrapper(env, teammate)

    rewards = []
    for _ in range(args.n_episodes):
        ep_reward = 0
        state = env.reset()
        terminal = False
        env.render(mode="plt")
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            state = next_state
            env.render(mode="plt")
            ep_reward += reward
        rewards.append(ep_reward)

    rewards = np.asarray(rewards)
    print(f"avg: {np.average(rewards)}")
    print(f"std: {np.std(rewards)}")
    print(f"max: {np.max(rewards)}")
    print(f"min: {np.min(rewards)}")

