from panda_gym.envs.core import RobotTaskEnv
from hppo import *
from typing import List
from torch.distributions import Categorical, Normal

class Trainer:
    def __init__(
            self,
            env: RobotTaskEnv,
            model: HybridActorCritic,
            timesteps: int,
            timesteps_per_batch: int,
            max_timesteps_per_episode: int,
            gamma: float = 0.99,
            epsilon: float = 0.2,
            alpha: float = 3e-4,
            training_cycle_per_batch: int = 5,
    ):
        self.env = env
        self.model = model
        self.timesteps = timesteps
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.training_cycle_per_batch = training_cycle_per_batch

    def run_episode(self):
        """
        run_episode runs a singular episode and returns the results
        """
        observation, _ = self.env.reset()
        if isinstance(observation, dict):
            observation = observation["observation"]

        timesteps = 0
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probabilities: List[float] = []
        rewards: List[float] = []

        while True:
            timesteps += 1

            # Feed forward
            observations.append(observation)
            discrete_logits, continuous_params, value = self.model(observation)

            # Sample discrete action
            discrete_dist = Categorical(
                logits=discrete_logits)
            discrete_action = discrete_dist.sample()

            # Sample continuous params
            continuous_params_mean = continuous_params[discrete_action.item()][
                "mean"]
            continuous_params_std = continuous_params[discrete_action.item()][
                "std"]
            continuous_params_dist = Normal(
                continuous_params_mean,
                continuous_params_std
            )
            params = continuous_params_dist.sample()

            # Interact with the environment
            action = {
                "discrete_action": discrete_action,
                "continuous_action": params,
            }
            next_obs, reward, terminated, _, _ = self.env.step(action)