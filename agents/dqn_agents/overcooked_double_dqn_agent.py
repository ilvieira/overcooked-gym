from dqn.policies.train_eval_policy import TrainEvalPolicy
from agents.dqn_agents.overcooked_dqn_agent import OvercookedDQNAgent
import torch
import torch.nn.functional as F


class OvercookedDoubleDQNAgent(OvercookedDQNAgent):
    def __init__(self, env, replay, n_actions, net_type, net_parameters, minibatch_size=32,
                 optimizer=torch.optim.RMSprop, C=10_000, update_frequency=1, gamma=0.99, loss=F.mse_loss,
                 policy=TrainEvalPolicy(), populate_policy=None, seed=0,
                 device="cuda" if torch.cuda.is_available() else "cpu", optimizer_parameters=None,
                 avg_loss_per_steps=10_000, eval_env=None, save_eval_videos=True):
        super().__init__(env, replay, n_actions, net_type, net_parameters, minibatch_size=minibatch_size,
                         optimizer=optimizer, C=C, update_frequency=update_frequency, gamma=gamma, loss=loss,
                         policy=policy, populate_policy=populate_policy, seed=seed, device=device,
                         optimizer_parameters=optimizer_parameters, avg_loss_per_steps=avg_loss_per_steps,
                         save_eval_videos=save_eval_videos)

        self.eval_env = eval_env if eval_env is not None else self.env
        self.train_env = self.env

    def update_net(self, r, not_done, next_phi):
        with torch.no_grad():
            actions = self.Q_target(next_phi).max(axis=1, keepdim=True).indices.view(self.minibatch_size)
            q_estimate = torch.zeros(self.minibatch_size).to(self.device)
            q_phi = self.Q(next_phi)
            for i in range(self.minibatch_size):
                q_estimate[i] = q_phi[i, actions[i]]
            y = (r + self.gamma * not_done * q_estimate)
        return y

    # ================================================================================================================
    # Agent Methods
    # ================================================================================================================

    def eval(self):
        super().eval()
        self.env = self.eval_env

    def train(self):
        super().train()
        self.env = self.train_env

