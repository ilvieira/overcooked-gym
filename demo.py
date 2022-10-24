from environment.overcooked import Overcooked, LAYOUTS, SingleAgentWrapper
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack
from yaaf.agents import HumanAgent
import argparse

"""By running this demo, the user is able to play with one of the handcoded teammates. To execute this file, go to the 
main folder in the terminal and run the following command:

'''python -m demo --teammate=<teammate_name>'''

where <teammate_name> must be replaced by one of the following (without the quotes): "fetcher", "jack_of_all_trades", 
"cook_n_server", "clockwise_jack", "counterclockwise_jack"

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Allows the user to play with one of the handcoded teammates.")
    parser.add_argument("--teammate", type=str)
    parser.add_argument("--max_timesteps", type=int, default=50)
    args = parser.parse_args()

    layout = "simple_kitchen"
    teammates = [Fetcher(LAYOUTS[layout], 1), JackOfAllTrades(LAYOUTS[layout], 1), CookNServer(LAYOUTS[layout], 1),
                 ClockwiseJack(LAYOUTS[layout], 1), CounterclockwiseJack(LAYOUTS[layout], 1)]
    teammates_names = ["fetcher", "jack_of_all_trades", "cook_n_server", "clockwise_jack", "counterclockwise_jack"]
    teammate_id = teammates_names.index(args.teammate)
    teammate = teammates[teammate_id]

    env = Overcooked(layout="simple_kitchen", max_timesteps=args.max_timesteps, rewards=(0, 0, 0, 1))
    agent = HumanAgent(action_meanings=env.action_meanings)
    env = SingleAgentWrapper(env, teammate)

    state = env.reset()
    terminal = False
    env.render(mode="plt")
    tot_reward = 0
    while not terminal:
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        state = next_state
        env.render(mode="plt")
        tot_reward += reward

    print(f"Final reward: {tot_reward}")

