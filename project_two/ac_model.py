import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class PolicyNetwork(nn.Module):
    """
    Gaussian Actor-Critic Model based on
    https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_heads.py
    and from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
    """

    def __init__(self, state_size: int = 33, action_size: int = 4, layer_size: int = 128):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param layer_size: Dimension of the hidden layers
        """
        super(PolicyNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # single layer shared body of network
        self.base = nn.Linear(state_size, layer_size)

        # action body and head of network. Provides the mean for the distribution from which to sample
        self.actor_body = nn.Linear(layer_size, layer_size)
        self.actor_out = nn.Linear(layer_size, action_size)

        # Provides the standard deviation for the distribution from which to sample
        self.std = nn.Parameter(torch.ones(1, action_size))

        # critic body and head of network. Provides the value function for the input state
        self.critic_body = nn.Linear(layer_size, layer_size)
        self.critic_out = nn.Linear(layer_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialise the network parameters.
        """
        self.base.weight.data.uniform_(*hidden_init(self.base))
        self.actor_body.weight.data.uniform_(*hidden_init(self.actor_body))
        self.critic_body.weight.data.uniform_(*hidden_init(self.critic_body))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps state -> action values. Builds a Normal distribution based on mean and variance and samples from it to get
        the actions. Also returns the entropy of the distribution.

        :param state: environment state
        :return: mean, standard dev, value function and entropy for the given state
        """
        base_out = F.relu(self.base(state))
        mean = self.actor_out(F.relu(self.actor_body(base_out)))
        dist = torch.distributions.Normal(mean, self.std.data)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        clamped_action = torch.clamp(action, -1, 1)

        value = self.critic_out(F.relu(self.critic_body(base_out)))
        entropy = dist.entropy()

        return clamped_action, log_prob, entropy, value

