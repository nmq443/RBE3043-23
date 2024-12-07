import torch
import torch.nn as nn
from panda_gym.envs.core import RobotTaskEnv
from model import DiscreteActor, ContinuousActor, Critic

class Trainer:
    def __init__(
            self,
            env: RobotTaskEnv,
            discrete_actor: DiscreteActor,
            continuous_actor: ContinuousActor,
            critic: Critic,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            alpha: float = 3e-4,
    ):
        # Environment
        self.env = env

        # Neural networks
        self.discrete_actor = discrete_actor
        self.continuous_actor = continuous_actor
        self.critic = critic

        # Hypeparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        # Optimizers
        self.discrete_optimizer = torch.optim.Adam(
            params=discrete_actor.parameters(), lr=self.alpha)
        self.continuous_optimizer = torch.optim.Adam(
            params=continuous_actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(params=critic.parameters(), lr=self.alpha)

    def run_episode(self):
        """Run a single episode."""
        observation, _ = self.env.reset()

        timesteps = 0
        observations = []
        discrete_actions = []
        continuous_params = []
        discrete_log_probs = []
        continuous_log_probs = []
        rewards = []

        while True:
            timesteps += 1

            observations.append(observation)
            if isinstance(observation, dict):
                observation = observation['observation']

            current_discrete_dist = self.discrete_actor(observation)
            current_discrete_action = current_discrete_dist.sample()
            current_discrete_log_prob = current_discrete_dist.log_prob(current_discrete_action)
            current_discrete_action =  current_discrete_action.detach().numpy()

            current_continuous_params = self.continuous_actor(observation)
            mean = current_continuous_params[current_discrete_action.item()][
                'mean']
            std = current_continuous_params[current_discrete_action.item()][
                'std']
            current_continuous_dist = torch.distributions.Normal(mean, std)
            current_continuous_action = current_continuous_dist.sample()
            continuous_log_prob = current_continuous_dist.log_prob(
                current_continuous_action)
            current_continuous_action = current_continuous_action.detach(

            ).numpy()

            action = {
                'discrete': current_discrete_action,
                'continuous': current_continuous_action
            }
            obs, reward, terminated, _, _ = self.env.step(action)

            discrete_actions.append(current_discrete_action)
            discrete_log_probs.append(current_discrete_log_prob)

            continuous_params.append(current_continuous_action)
            continuous_log_probs.append(continuous_log_prob)

            rewards.append(reward)

            if terminated:
                break

        # Calculate the discounted rewards for this episode
        discounted_rewards = self.compute_discounted_rewards(rewards)

        # Get the terminal reward and record for status tracking
        # self.total_rewards.append(sum(rewards))

        return (observations, discrete_actions, continuous_params,
                discrete_log_probs, continuous_log_probs, discounted_rewards)

    def rollout(self):
        """Perform a rollout of the environment and return the memory of the
        episode with the current actor models
        """
