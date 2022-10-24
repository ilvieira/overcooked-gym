from dqn.agents.agent import Agent
import numpy as np
import time


class DistilledPLASTICPolicyAgent(Agent):
    def __init__(self, env, n_actions, distilled_agent, eta, teammates_models,
                 teammates_names=None, initial_distribution=None):
        super().__init__(env, n_actions)
        self.eta = eta
        self.distilled_agent = distilled_agent
        self.n_teammates = distilled_agent.n_tasks
        self.distilled_agent.envs = [env for _ in range(self.n_teammates)]

        self.teammates_models = teammates_models
        self.teammates_names = [f"t{i}" for i in range(self.n_teammates)] if teammates_names is None else teammates_names

        # setup initial distribution
        self.initial_distribution = np.ones(self.n_teammates) / self.n_teammates if initial_distribution is None else initial_distribution
        self.reset_beliefs()

    def reset_beliefs(self):
        self.behaviour_distribution = self.initial_distribution

    def update_beliefs(self, state, action):
        for i in range(self.n_teammates):
            model = self.teammates_models[i]
            loss = 1 - model.action_probability(state, action)
            self.behaviour_distribution[i] *= (1 - self.eta*loss)
        self.behaviour_distribution /= self.behaviour_distribution.sum()

    def action(self, observation):
        teammate_index = np.argmax(self.behaviour_distribution)
        self.distilled_agent.select_task(teammate_index)
        return self.distilled_agent.action(observation)

    def act_in_domain(self, n_episodes):
        rewards = []
        behaviour_distributions = []
        self.reset_beliefs()
        behaviour_distributions.append(np.copy(self.behaviour_distribution))
        st = time.time()
        for i in range(n_episodes):
            #print(f"Epsisode {i}")
            #print(f"{self.behaviour_distribution}")
            ep_reward = 0
            state = self.env.reset()
            ohe_state = self.env.current_state_ohe()
            terminal = False
            timestep = 0
            while not terminal:
                action = self.action(state)
                next_state, reward, terminal, info = self.env.step(action)
                teammate_action = info['teammate_action']
                self.update_beliefs(ohe_state, teammate_action)
                behaviour_distributions.append(np.copy(self.behaviour_distribution))
                state = next_state
                ohe_state = self.env.current_state_ohe()
                ep_reward += reward
                timestep += 1
            rewards.append(ep_reward)
            #print(f"......................................................... {time.time()-st}s")
        return rewards, behaviour_distributions


