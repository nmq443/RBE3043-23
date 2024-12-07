import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class HybridActorCritic(nn.Module):
    def __init__(
            self,
            obs_dim,
            discrete_action_dim=4,
            continuous_params_dim=[1, 1, 1, 1]
    ):
        """

        :param obs_dim: [obj_pos(3), target_pos(3), ee_pos(3), ] (20)
        :param discrete_action_dim: MoveToObj(velocity), OpenGripper(
        velocity),
        CloseGripper(velocity),
        MoveObjToTarget(velocity)
        :param continuous_action_dim: dimension of parameters of discrete
        actions
        """
        super(HybridActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_params_dim = continuous_params_dim

        # Share based layers for features extraction
        # (256, 256, 128, 64)
        self.shared_fc = nn.Sequential(
            nn.Linear(
                in_features=obs_dim,
                out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),
        )

        # Discrete actor network
        # Output of the discrete actor network is f1, f2, ..., fk
        # With 1-k are the discrete actions
        # We will then use softmax distribution to choose the discrete action
        self.discrete_actor = nn.Linear(
            in_features=64,
            out_features=self.discrete_action_dim
        )

        # Continuous actor network
        # Output of continuous actor network are mean and standard deviation
        # Of a Gaussian distribution for the continuous params
        self.continuous_actor = nn.ModuleList(
            nn.ModuleDict({
                "mean": nn.Sequential(
                    nn.Linear(
                        in_features=64,
                        out_features=64
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        in_features=64,
                        out_features=param_dim
                    )
                ),
                "std": nn.Sequential(
                    nn.Linear(
                        in_features=64,
                        out_features=64
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        in_features=64,
                        out_features=param_dim
                    ),
                    nn.Softplus()  # Ensures positive standard deviations
                )
            })
            for param_dim in self.continuous_params_dim
        )

        # Critic network
        # Output of critic network is V(s), aka state-value function
        # To estimate how good the current policy is
        self.critic_fc = nn.Linear(
            in_features=64,
            out_features=1
        )

    def forward(self, obs):
        # Shared features extraction
        features = self.shared_fc(obs)

        # Discrete actions (logits)
        discrete_logits = self.discrete_actor(features)

        continuous_params = [
            {
                "mean": head["mean"](features),
                "std": head["std"](features)
            }
            for head in self.continuous_actor
        ]

        # Value estimation
        value = self.critic_fc(features)

        return discrete_logits, continuous_params, value
