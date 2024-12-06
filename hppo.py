import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as distribution


class HybridActorCritic(nn.Module):
    def __init__(
            self,
            obs_dim,
            discrete_actions,
            continuous_actions
    ):
        self.obs_dim = obs_dim
        self.discrete_actions = discrete_actions
        self.continuous_actions = continuous_actions

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
            out_features=discrete_actions
        )

        # Continuous actor network
        # Output of continuous actor network are mean and standard deviation
        # Of a Gaussian distribution for the continuous actions
        self.continuous_actor_mean = nn.Linear(
            in_features=64,
            out_features=continuous_actions
        )
        self.continuous_actor_std = nn.Linear(
            in_features=64,
            out_features=continuous_actions
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

        # Continuous actions (mean and std)
        mean = self.continuous_actor_mean(features)
        # Use softplus to ensure std is positive
        std = F.softplus(self.continuous_actor_std(features))

        # Value estimation
        value = self.critic_fc(features)

        return discrete_logits, mean, std, value


def select_action(network, state):
    state = torch.FloatTensor(state).unsqueeze(0)

    # Forward pass
    discrete_logits, mean, std, value = network(state)

    # Discrete action
    discrete_distribution = distribution.Categorical(logits=discrete_logits)
    discrete_action = discrete_distribution.sample()

    # Continuous action
    continuous_distribution = distribution.Normal(mean, std)
    continuous_action = continuous_distribution.sample()
    # Normalized the parameters
    continuous_action = torch.clamp(
        input=continuous_action,
        min=0.0,
        max=1.0
    )

    # Log probabilities for gradient updates
    log_prob_discrete = discrete_distribution.log_prob(discrete_action)
    log_prob_continuous = continuous_distribution.log_prob(continuous_action).sum(dim=-1)

    log_prob = log_prob_discrete + log_prob_continuous

    return discrete_action.item(), continuous_action.squeeze().detach().numpy(), log_prob, value


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
