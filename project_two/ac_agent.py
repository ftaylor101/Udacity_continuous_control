import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from .ac_model import PolicyNetwork


class A2CAgent:
    """
    Interacts with and learns from the environment.
    Learns using Advantage Actor-Critic algorithm.
    One model is used but with multiple heads. One head to provide the distribution from which to sample for actions,
    the second from which to get the Value function to use as the critic.

    Agent training based on https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/A2C_agent.py
    """

    def __init__(self, num_of_agents: int, gamma: float = 0.99, lr: float = 1e-4,
                 entropy_wt: float = 0.01, value_loss_wt: float = 1.0, gradient_clip: float = 5.0,
                 gae_tau: float = 0.99):
        """
        Constructor for A2C agent.

        :param gamma: reward discount
        :param lr: optimiser learning rate
        :param num_of_agents: number of agents in the environment
        :param entropy_wt: weight used against the entropy in the update step
        :param value_loss_wt: weight used against the critic in the update step
        :param gradient_clip: a maximum value for the gradient to prevent it from varying too much
        :param gae_tau: Generalised Advantage Estimator hyperparameter
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = 33
        self.action_dim = 4

        self.num_of_agents = num_of_agents
        self.policy = PolicyNetwork()

        self.gamma = gamma
        self.lr = lr
        self.gae_tau = gae_tau

        self.entropy_wt = entropy_wt
        self.value_loss_wt = value_loss_wt
        self.gradient_clip = gradient_clip
        self.optimiser = optim.Adam(self.policy.parameters(), lr=self.lr)

    def get_action(self, state: np.array) -> np.array:
        """
        Given a state return an action sampled from a Normal distribution

        :param state: environment state
        :return np_actions: array of actions
        :return log_probs: log probability of each action
        :return vals: Value function for each action
        :return entropy: entropy of each action
        """
        t_state = torch.from_numpy(state).float()
        action, log_prob, entropy, value = self.policy(t_state)
        actions = action.detach().cpu().numpy()

        return actions, log_prob, entropy, value

    def update_policy(self, experience):
        """
        Update step to update the Actor-Critic network using the log probabilities and advantages of each n-bootstrap
        step.

        :param experience:
            log_prob,
            entropy,
            value,
            rewards,
            done,
            next_value_from_next_state
        """
        advantages = torch.from_numpy(np.zeros((self.num_of_agents, 1))).float()
        adv = [0] * (len(experience))
        ret = [0] * (len(experience))
        returns = experience[-2][-1].detach()
        for i in reversed(range(len(experience))):
            reward = torch.from_numpy(experience[i][3]).float().unsqueeze(1)
            not_done = torch.from_numpy(1 - np.array(experience[i][4])).float().unsqueeze(1)
            value = experience[i][2].detach()
            returns = reward + self.gamma * not_done * returns  # all rewards in rollout + final expected return
            next_value = experience[i][5].detach()
            td_error = reward + self.gamma * not_done * next_value - value  # the smaller the error the better the value estimate is
            advantages = advantages * self.gae_tau * self.gamma * not_done + td_error
            adv[i] = advantages
            ret[i] = returns

        all_log_probs = torch.cat([experience[i][0] for i in range(len(experience))])
        all_values = torch.cat([experience[i][2] for i in range(len(experience))])
        all_returns = torch.cat(ret)
        all_advantages = torch.cat(adv)
        all_entropies = torch.cat([experience[i][1] for i in range(len(experience))])

        policy_loss = -all_log_probs * all_advantages
        value_loss = 0.5 * (all_returns - all_values).pow(2)
        entropy_loss = all_entropies

        self.optimiser.zero_grad()
        loss = (policy_loss - self.entropy_wt * entropy_loss + self.value_loss_wt * value_loss).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
        self.optimiser.step()
