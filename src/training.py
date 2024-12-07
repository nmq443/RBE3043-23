import torch
import numpy as np
from torch import Tensor
from panda_gym.envs.core import RobotTaskEnv
from model import DiscreteActor, ContinuousActor, Critic


class Trainer:
    def __init__(
            self,
            env: RobotTaskEnv,
            discrete_actor: DiscreteActor,
            continuous_actor: ContinuousActor,
            critic: Critic,
            timesteps: int,
            timesteps_per_batch: int,
            max_timesteps_per_episode: int,
            training_cycles_per_batch: int = 5,
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

        # Iteration parameters
        self.timesteps = timesteps
        self.current_timestep = 0
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.timesteps_per_batch = timesteps_per_batch

        # Optimizers
        self.discrete_optimizer = torch.optim.Adam(
            params=discrete_actor.parameters(), lr=self.alpha)
        self.continuous_optimizer = torch.optim.Adam(
            params=continuous_actor.parameters(), lr=self.alpha)
        self.critic_optimizer = torch.optim.Adam(params=critic.parameters(),
                                                 lr=self.alpha)

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
            current_discrete_log_prob = current_discrete_dist.log_prob(
                current_discrete_action)
            current_discrete_action = current_discrete_action.detach().numpy()

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
        observations = []
        discrete_log_probabilitiess = []
        continuous_log_probabilities = []
        discrete_actions = []
        continuous_actions = []
        rewards = []

        while len(observations) < self.timesteps_per_batch:
            # self.current_action = "Rollout"
            (
                obs,
                chosen_discrete_actions,
                chosen_continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                rwds
            ) = self.run_episode()

            # Combine these arrays into overall batch
            observations += obs
            discrete_actions += chosen_discrete_actions
            continuous_actions += chosen_continuous_actions
            discrete_log_probabilitiess += discrete_log_probs
            continuous_log_probabilities += continuous_log_probs
            rewards += rwds

            # Increment count of timesteps
            self.current_timestep += len(obs)

            # self.print_status()

        # Trim the batch memory to the batch size
        observations = observations[:self.timesteps_per_batch]
        discrete_actions = discrete_actions[:self.timesteps_per_batch]
        continuous_actions = continuous_actions[:self.timesteps_per_batch]
        discrete_log_probabilitiess = discrete_log_probabilitiess[
                                      :self.timesteps_per_batch]
        continuous_log_probabilities = continuous_log_probabilities[
                                       :self.timesteps_per_batch]
        rewards = rewards[:self.timesteps_per_batch]

        return (observations, discrete_actions, continuous_actions,
                discrete_log_probabilitiess, continuous_log_probabilities,
                rewards)

    def calculate_discounted_reward(self, rewards):
        """Calculate the discounted reward of each timestep of an episode
        given its initial rewards and episode length"""
        discounted_rewards = []
        discounted_reward = 0.0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def calculate_normalized_advantage(self, observations, rewards):
        """Calculate the normalized advantage of each timestep of a given
        batch of episode """
        V = self.critic(observations).detach().squeeze()

        advantage = Tensor(np.array(rewards, dtype="float32")) - V
        normalized_advantage = (advantage - advantage.mean()) / (
            advantage.std() + 1e-8)

        return normalized_advantage

    def training_step(
            self,
            observations,
            discrete_actions,
            continuous_actions,
            discrete_log_probabilities,
            continuous_log_probabilities,
            rewards,
            normalized_advantage
    ):
        """Peform a single epoch of training for the actors and critic model. Return the loss for each model at the end of the step"""
        current_discrete_dist = self.discrete_actor(observations)
        current_discrete_log_probs = current_discrete_dist.log_prob(
            discrete_actions)
        discrete_ratio =
