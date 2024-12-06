import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as distribution


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
            )
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
                        out_features=param_dim
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
            for head in self.continuous_heads
        ]

        # Value estimation
        value = self.critic_fc(features)

        return discrete_logits, continuous_params, value


def select_action(network, state):
    state = torch.FloatTensor(state).unsqueeze(0)

    # Forward pass
    discrete_logits, continuous_params, value = network(state)

    # Discrete action
    discrete_distribution = distribution.Categorical(logits=discrete_logits)
    discrete_action = discrete_distribution.sample()

    # Continuous params
    mean = continuous_params[discrete_action]["mean"]
    std = continuous_params[discrete_action]["std"]
    continuous_action = torch.normal(mean, std)

    return discrete_action, continuous_params, value


def compute_loss(
        log_probs,
        values,
        rewards,
        dones,
        next_values,
        gamma=0.99,
        entropy_coeff=0.01
):
    # Compute returns
    returns = []
    G = 0

    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + (1 - done) * gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    # Compute advantage
    values = torch.cat(values)
    advantage = returns - values

    # Critic loss (MSE)
    critic_loss = advantage.pow(2).mean()

    # Actor loss (policy gradient)
    log_probs = torch.cat(log_probs)
    actor_loss = -(log_probs * advantage.detach()).mean()

    # Total loss
    total_loss = critic_loss + actor_loss

    return total_loss


if __name__ == "__main__":
    hybrid_actor_critic = HybridActorCritic()
