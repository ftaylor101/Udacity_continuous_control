import torch
import numpy as np
import torch.optim as optim
from typing import List, Tuple


class A2CAgent:
    """
    Interacts with and learns from the environment.
    Learns using Advantage Actor-Critic algorithm.
    One model is used but with multiple heads. One head to provide the distribution from which to sample for actions,
    the second from which to get the Value function to use as the critic.
    """

    def __init__(self, policy: torch.nn.Module, gamma: float, lr: float, agents: int, entropy_wt: float = 0.01,
                 critic_wt: float = 1.0):
        """
        Constructor for A2C agent.

        :param policy: Neural network of an A2C agent
        :param gamma: reward discount
        :param lr: optimiser learning rate
        :param agents: number of agents in the environment
        :param entropy_wt: weight used against the entropy in the update step
        :param critic_wt: weight used against the critic in the update step
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = 33
        self.action_dim = 4

        self.GAMMA = gamma
        self.lr = lr
        self.entropy_weight = entropy_wt
        self.critic_loss_wt = critic_wt

        self.num_of_agents = agents

        self.model = policy
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_action(self, state: np.array) -> Tuple[np.array, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a state return an action sampled from a Normal distribution

        :param state: environment state
        :return np_actions: array of actions
        :return log_probs: log probability of each action
        :return vals: Value function for each action
        :return entropy: entropy of each action
        """
        tensor_state = torch.from_numpy(state).float().to(self.device)  # convert state
        mus, stds, vals, entropy = self.model(tensor_state)  # get distribution details
        dist = torch.distributions.Normal(mus, stds)
        acts = dist.sample()
        actions = np.clip(acts, -1, 1)  # sample for actions and limit to [-1, 1] for environment

        log_probs = dist.log_prob(actions)
        np_actions = actions.detach().cpu().numpy()

        return np_actions, log_probs, vals, entropy

    def discount_rewards(self, ep_rewards: List[np.array], ep_values: List[np.array]) -> np.array:
        """
        Method to discount the rewards by GAMMA for each step in the n-step bootstrapping and to return the advantage
        by using the value function. The advantage returned is normalised.

        :param ep_rewards: rewards received for the current bootstrap
        :param ep_values: values for the current bootstrap
        :return: normalised advantage
        """
        gammas = self.GAMMA ** np.arange(len(ep_rewards))
        last_values = ep_values[-1].detach().cpu().numpy().squeeze()
        ep_rewards[-1] = last_values
        dis_return = np.array([e*gamma for e, gamma in zip(ep_rewards, gammas)])

        mean = np.mean(dis_return)
        std = np.std(dis_return) + 1e-10
        norm_dis_return = (dis_return - mean) / std

        return norm_dis_return

    def update_policy(self, rewards, log_probs, values, entropy):
        """
        Update step to update the Actor-Critic network using the log probabilities and advantages of each n-bootstrap
        step.

        :param rewards: the rewards received for the n-steps in the rollout
        :param log_probs: the log probabilities of each action for each agent
        :param values: the value function for each agent for each state in the rollout
        :param entropy: the entropy of the action for each agent
        """
        discounted_rewards = self.discount_rewards(ep_rewards=rewards, ep_values=values)
        update = 0
        ads = []
        for t in range(len(log_probs) - 1):
            dis_reward = torch.from_numpy((np.sum(discounted_rewards[t:], axis=0))).float()
            value = values[t].view(self.num_of_agents)
            log_prob = log_probs[t].sum(dim=1)
            update += -log_prob * (dis_reward - value)
            ads.append(dis_reward - value)

        actor_loss = update.mean()
        advantages = torch.cat(ads)

        ent = entropy[0].view(self.num_of_agents)
        entropy_sum = torch.sum(ent)

        critic_loss = 0.5 * (advantages**2).mean()
        ac_loss = (actor_loss - self.entropy_weight * entropy_sum.mean() + self.critic_loss_wt * critic_loss)\
            .to(self.device)

        self.optimiser.zero_grad()
        ac_loss.backward()
        self.optimiser.step()
