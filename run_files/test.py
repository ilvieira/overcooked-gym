from environment.overcooked import LAYOUTS, Overcooked, StatesAsFramesWrapper
from agents.teammates.fetcher import Fetcher
import matplotlib.pyplot as plt


layout = "simple_kitchen"
env = Overcooked(layout=LAYOUTS["simple_kitchen"], max_timesteps=500, rewards=(0, 0, 0, 1))
env = StatesAsFramesWrapper(env, Fetcher(LAYOUTS[layout], 1))

# collect some states
states = []
for _ in range(10):
    state = env.random_state()
    rgb_frame = env.env.render_state(state)
    preprocessed_frame = env.preprocess(rgb_frame)
    states.append(preprocessed_frame)
    plt.imshow(rgb_frame)
    plt.show()
    plt.imshow(preprocessed_frame.reshape((84,84)))
    plt.show()
