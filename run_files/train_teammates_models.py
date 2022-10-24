import numpy as np
import os
import pickle
import time
import random as rnd
from sklearn import neighbors
import argparse
from environment.overcooked import Overcooked, LAYOUTS
from agents.teammates.fetcher import Fetcher
from agents.teammates.jack_of_all_trades import JackOfAllTrades
from agents.teammates.cook_n_server import CookNServer
from agents.teammates.clockwise_jack import ClockwiseJack
from agents.teammates.counterclockwise_jack import CounterclockwiseJack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a model of each teammate using n-nearest neighbors.")
    parser.add_argument("--round", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    # begin training -------------------------------------------------------------------------------------------------
    save_dir = os.getcwd() + f"/data/round{args.round}/teammates_models/" if args.data_dir is None else args.data_dir
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
    st = time.time()

    # collect data if its directory is not passed as an input
    if args.data_dir is not None:
        X_ohe_path = args.data_dir + "X_ohe.npy"
        with open(X_ohe_path, 'rb') as x_ohe_file:
            X_ohe = np.load(x_ohe_file)
    else:
        data_dir = save_dir
        X_path = data_dir + "X.npy"
        X_ohe_path = data_dir + "X_ohe.npy"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        env = Overcooked(layout="simple_kitchen")
        X = []
        X_ohe = []

        for _ in range(200_000):
            state = env.random_state()
            state_ohe = env.one_hot_encoding(state)
            X.append(state)
            X_ohe.append(state_ohe)

        X = np.asarray(X)
        X_ohe = np.asarray(X_ohe)

        with open(X_path, 'wb') as x_file:
            np.save(x_file, X)

        with open(X_ohe_path, 'wb') as x_ohe_file:
            np.save(x_ohe_file, X_ohe)

        print(f"data collected - {time.time() - st}")

    for t_id, t_name in enumerate(["fetcher", "jack_of_all_trades", "cook_n_server", "clockwise_jack",
                                   "counterclockwise_jack"]):
        # get the labels for the data
        if args.data_dir is not None:
            y_path = args.data_dir + f"y_{t_id}.npy"
            with open(y_path, 'rb') as y_file:
                y = np.load(y_file)
        else:
            y_path = save_dir + f"y_{t_id}.npy"
            agent = teammates[t_id]
            y = []

            for i, state in enumerate(X):
                action = agent.action(state)
                y.append(action)

            y = np.asarray(y)

            with open(y_path, 'wb') as y_file:
                np.save(y_file, y)

            print(f"data labelled - {t_name} - {time.time() - st}")

        n_neighbors = 7
        weights = "distance"
        model_path = save_dir + f"model{t_id}.pickle"
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X_ohe, y)
        print(f"{n_neighbors}nn classifier trained for teammate {t_name} with weights {weights}: {time.time()-st}s")

        with open(model_path, 'wb') as model_file:
            pickle.dump(clf, model_file)
