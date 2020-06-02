import torch
import torch.nn as nn
from typing import Tuple

import numpy as np


class PolicyNetwork(nn.Module):
    """
    Actor-Critic (Policy) Model.
    """

    def __init__(self, state_size: int = 33, action_size: int = 4, seed: int = 46, layer_size: int = 128):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param layer_size: Dimension of the hidden layers
        """
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # main body of network
        self.base = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU()
        )

        # action head of network. Provides the mean for the distribution from which to sample
        self.mu = nn.Sequential(
            nn.Linear(layer_size, action_size),
            nn.Tanh()
        )

        # action head of network. Provides the variance for the distribution from which to sample
        self.var = nn.Sequential(
            nn.Linear(layer_size, action_size),
            nn.Softplus()
        )

        # value head of network. Provides the value function for the input state.
        self.value = nn.Linear(layer_size, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps state -> action values. Builds a Normal distribution based on mean and variance and samples from it to get
        the actions. Also returns the entropy of the distribution.

        :param state: environment state
        :return: mean, standard dev, value function and entropy for the given state
        """
        base_out = self.base(state)
        mu = self.mu(base_out)
        std = torch.sqrt(self.var(base_out))
        value = self.value(base_out)

        dist = torch.distributions.Normal(mu, std)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return mu, std, value, entropy
