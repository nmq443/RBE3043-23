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

def sample_action(discrete_logits, continuous_params):
    """Sample a discrete action and its parameters
    Args:
        discrete_logits (torch.Tensor): discrete actor output logits
        continuous_params (torch.Tensor): continuous actor output logits

    Returns:
        a_d (int): discrete action
        x_a (torch.Tensor): continuous parameters for a_d
    """
    # Sample discrete action
    discrete_action_probabilities = F.softmax(input=discrete_logits, dim=-1)
    a_d = torch.multinomial(
        discrete_action_probabilities,
        num_samples=1
    ).item()

    log_probability = torch.multinomial(
        discrete_action_probabilities,
        num_samples=1
    )

    # Sample continuous parameters for the chosen discrete action
    mean = continuous_params[a_d]["mean"]
    std = continuous_params[a_d]["std"]
    x_a = torch.normal(mean, std)

    return a_d, x_a


def compute_advantages(
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
):
    """Computes advantages using Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor of rewards for each timestep.
        values: Tensor of value predictions for each timestep.
        next_values: Tensor of value predictions for the next timestep.
        dones: Tensor indicating episode termination (1 if done, 0 otherwise).
        gamma: Discount factor for rewards.
        lambda_: GAE parameter for bias-variance tradeoff.

    Returns:
        advantages: Tensor of computed advantages.
        returns: Tensor of computed discounted returns.
    """

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        # Terminal state advantage = 0
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = 0
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lambda_ * gae

        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def compute_loss(
        model: HybridActorCritic,
        observations: torch.Tensor,
        actions: Dict,
        advantages: float,
        returns: any,
        old_log_probs: any,
        clip_range: float,
        entropy_coeff: float = 0.01
):
    """
        Computes the PPO loss for H-PPO.

        Args:
            model: The H-PPO network model.
            observations: array like
            Batch of observations.
            actions: Dictionary with "discrete" and "continuous" actions.
            advantages: Advantage estimates for each observation.
            returns: Discounted returns for each observation.
            old_log_probs: Log probabilities of actions from the previous policy.
            clip_range: PPO clipping range.
            entropy_coeff: Coefficient for the entropy bonus.

        Returns:
            total_loss: Combined actor and critic loss.
            loss_dict: Dictionary containing actor, critic, and entropy loss.
    """
    # Forward pass
    discrete_logits, continuous_params, state_value = model(observations)

    # ---- Critic loss ----
    critic_loss = nn.functional.mse_loss(state_value.squeeze(-1), returns)

    # --- Discrete Actor loss ----
    discrete_action_logits = discrete_logits.gather(1, actions[
        "discrete"].unsqueeze(-1)).squeeze(-1)
    discrete_log_probs = nn.functional.log_softmax(discrete_action_logits,
                                                   dim=-1)
    discrete_log_probs = discrete_log_probs.gather(1, actions[
        "discrete"].unsqueeze(-1)).squeeze(-1)
    discrete_ratios = torch.exp(discrete_log_probs - old_log_probs["discrete"])

    # PPO Clipping for Discrete Actions
    clipped_discrete_ratios = torch.clamp(discrete_ratios, 1 - clip_range,
                                          1 + clip_range)
    discrete_loss = -torch.min(discrete_ratios * advantages,
                               clipped_discrete_ratios * advantages).mean()
    # ---- Continuous Actor Loss ----
    continuous_log_probs = []

    for i, action in enumerate(actions["continuous"]):
        print(action.item())
        mean = continuous_params[action.item()]["mean"]
        std = continuous_params[action.item()]["std"]
        dist = torch.distributions.Normal(mean, std)
        continuous_log_prob = dist.log_prob(action).sum(
            dim=-1)  # Sum over action dimensions
        continuous_log_probs.append(continuous_log_prob)

    continuous_log_probs = torch.stack(continuous_log_probs, dim=1)
    continuous_ratios = torch.exp(
        continuous_log_probs - old_log_probs["continuous"])

    # PPO Clipping for Continuous Parameters
    clipped_continuous_ratios = torch.clamp(continuous_ratios, 1 - clip_range,
                                            1 + clip_range)
    continuous_loss = -torch.min(continuous_ratios * advantages,
                                 clipped_continuous_ratios * advantages).mean()

    # ---- Entropy Regularization ----
    discrete_entropy = -(nn.functional.softmax(discrete_logits, dim=-1) *
                         nn.functional.log_softmax(
                             discrete_logits, dim=-1)).sum(dim=-1).mean()
    continuous_entropy = sum(
        [dist.entropy().mean() for dist in
         [torch.distributions.Normal(head["mean"], head["std"]) for head in
          continuous_params]]
    )
    entropy_loss = -(discrete_entropy + continuous_entropy)

    # ---- Total Loss ----
    actor_loss = discrete_loss + continuous_loss
    total_loss = actor_loss + 0.5 * critic_loss + entropy_coeff * entropy_loss

    # ---- Return Losses ----
    loss_dict = {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy_loss": entropy_loss.item(),
    }
    return total_loss, loss_dict

def loss_fn(
    model, batch_obs, batch_actions, batch_advantages, batch_returns, batch_log_probs, clip_range, entropy_coeff
):
    # Forward pass
    model_outputs = model(batch_obs)
    new_log_probs = model_outputs["log_probs"]  # New probabilities for discrete & continuous
    values = model_outputs["values"]           # Value predictions
    entropy = model_outputs["entropy"]         # Entropy of the distributions

    # Compute policy ratio
    ratios = {
        "discrete": torch.exp(new_log_probs["discrete"] - batch_log_probs["discrete"]),
        "continuous": torch.exp(new_log_probs["continuous"] - batch_log_probs["continuous"]),
    }

    # Compute clipped policy loss for discrete and continuous
    clipped_loss = {
        "discrete": torch.min(
            ratios["discrete"] * batch_advantages,
            torch.clamp(ratios["discrete"], 1 - clip_range, 1 + clip_range) * batch_advantages,
        ).mean(),
        "continuous": torch.min(
            ratios["continuous"] * batch_advantages,
            torch.clamp(ratios["continuous"], 1 - clip_range, 1 + clip_range) * batch_advantages,
        ).mean(),
    }

    # Total policy loss
    policy_loss = -1 * (clipped_loss["discrete"] + clipped_loss["continuous"])

    # Value loss
    value_loss = ((values - batch_returns) ** 2).mean()

    # Entropy regularization
    entropy_loss = -1 * entropy_coeff * entropy.mean()

    # Total loss
    total_loss = policy_loss + value_loss + entropy_loss

    return total_loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy_loss": entropy_loss}
