import gymnasium as gym
from hppo import *
from training import train_hppo

observation_dim = 20
discrete_action_dim = 4
continuous_params_dims = [1, 1, 1, 1]

# Define a dummy environment
class DummyHybridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # self.observation_space = gym.spaces.Box(-1, 1, (20,))
        # self.action_space = gym.spaces.Dict({
        #     "discrete": gym.spaces.Discrete(4),
        #     "continuous": gym.spaces.Box(-1, 1, (3,)),
        # })
        self.observation_dim = observation_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_params_dims = continuous_params_dims

    def reset(self, seed=None, **kwargs):
        # return self.observation_space.sample()
        return torch.randn(self.observation_dim)

    def step(self, action):
        reward = 1.0  # Dummy reward
        done = torch.rand(1).item() < 0.1  # Random termination
        return torch.randn(self.observation_dim), reward, done, {}

# Instantiate the environment and model
env = DummyHybridEnv()

# Create the model and optimizer
model = HybridActorCritic(observation_dim, discrete_action_dim, continuous_params_dims)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Train the agent
train_hppo(
    model,
    optimizer,
    env
)
