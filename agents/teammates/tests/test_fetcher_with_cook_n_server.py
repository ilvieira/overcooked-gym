from environment.overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from agents.teammates.fetcher import Fetcher
from agents.teammates.cook_n_server import CookNServer
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Makes the CookNServer and the Fetcher agent work together.")
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument("--render", type=bool, default=False)
    args = parser.parse_args()

    env = Overcooked(layout="simple_kitchen", max_timesteps=args.max_timesteps, rewards=(0, 0, 0, 1))
    agent = CookNServer(LAYOUTS["simple_kitchen"], index=0)
    teammate = Fetcher(LAYOUTS["simple_kitchen"], index=1)
    env = SingleAgentWrapper(env, teammate)

    rewards = []
    for _ in range(args.n_episodes):
        ep_reward = 0
        state = env.reset()
        terminal = False
        if args.render:
            env.render(mode="plt")
        while not terminal:
            action = agent.action(state)
            next_state, reward, terminal, info = env.step(action)
            state = next_state
            if args.render:
                env.render(mode="plt")
            ep_reward += reward
        rewards.append(ep_reward)

    rewards = np.asarray(rewards)
    print(f"avg: {np.average(rewards)}")
    print(f"std: {np.std(rewards)}")
    print(f"max: {np.max(rewards)}")
    print(f"min: {np.min(rewards)}")

