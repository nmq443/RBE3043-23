from hppo import *
import torch
from torch.optim import Adam
from torch.distributions import Categorical, Normal


def train_hppo(
        model, optimizer, env, num_episodes=10, num_steps=2048, batch_size=64,
        gamma=0.99, lambda_=0.95, clip_range=0.2, entropy_coeff=0.01,
        vf_coeff=0.5
):
    """
    Implements the H-PPO training loop.

    Args:
        model: The H-PPO network.
        optimizer: Optimizer for the model.
        env: The simulation environment.
        num_episodes: Number of training episodes.
        num_steps: Number of steps per environment rollout.
        batch_size: Batch size for training.
        gamma: Discount factor for rewards.
        lambda_: GAE parameter.
        clip_range: Clipping range for PPO.
        entropy_coeff: Entropy regularization coefficient.
        vf_coeff: Value function loss coefficient.

    Returns:
        None
    """
    for episode in range(num_episodes):
        # Rollout phase
        observations, discrete_actions, continuous_actions, rewards, dones, values, log_probs = [], [], [], [], [], [], []

        obs = env.reset()
        for step in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            discrete_logits, continuous_params, value = model(obs_tensor)

            # Sample discrete action
            discrete_dist = Categorical(logits=discrete_logits)
            discrete_action = discrete_dist.sample()

            # Sample continuous actions for the chosen discrete action
            continuous_mean = continuous_params[discrete_action.item()]["mean"]
            continuous_std = continuous_params[discrete_action.item()]["std"]
            continuous_dist = Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()

            # Interact with the environment
            next_obs, reward, done, _ = env.step(
                [discrete_action.item(), continuous_action.tolist()])

            # Store rollout data
            observations.append(obs)
            discrete_actions.append(discrete_action.item())
            continuous_actions.append(continuous_action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze().item())
            log_probs.append({
                "discrete": discrete_dist.log_prob(discrete_action).item(),
                "continuous": continuous_dist.log_prob(
                    continuous_action).sum().item()
            })

            obs = next_obs
            if done:
                obs = env.reset()

        # Prepare rollout data as tensors
        observations = torch.tensor(observations, dtype=torch.float32)
        discrete_actions = torch.tensor(discrete_actions, dtype=torch.long)
        continuous_actions = torch.stack(continuous_actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        next_values = torch.cat([values[1:], torch.tensor(
            [0.0])])  # Bootstrap next values for non-terminal steps

        # Compute advantages and returns
        advantages, returns = compute_advantages(rewards, values, next_values,
                                                 dones, gamma, lambda_)

        # Training phase
        for _ in range(num_episodes):
            indices = torch.randperm(num_steps)
            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Extract batches
                batch_obs = observations[batch_indices]
                batch_discrete = discrete_actions[batch_indices]
                batch_continuous = continuous_actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_log_probs = {
                    "discrete": torch.tensor(
                        [log_probs[i]["discrete"] for i in batch_indices]),
                    "continuous": torch.stack(
                        [log_probs[i]["continuous"] for i in batch_indices]),
                }

                # Compute loss
                loss, loss_dict = compute_loss(
                    model, batch_obs, {"discrete": batch_discrete,
                                       "continuous": batch_continuous},
                    batch_advantages, batch_returns, batch_log_probs,
                    clip_range, entropy_coeff
                )

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Logging (optional)
        print(
            f"Epoch {episode + 1}/{num_episodes}: Actor Loss = {loss_dict['actor_loss']:.4f}, "
            f"Critic Loss = {loss_dict['critic_loss']:.4f}, Entropy Loss = {loss_dict['entropy_loss']:.4f}")

