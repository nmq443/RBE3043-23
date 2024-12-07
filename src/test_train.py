from experiment_env import *
from model import DiscreteActor, ContinuousActor, Critic
import torch
from training import Trainer

MOVE = 0
PICK = 1
PLACE = 2

action_space = {
    'discrete': [MOVE, PICK, PLACE],
    'continuous': [3, 3, 3]
}

discrete_dim = len(action_space['discrete'])
continuous_dim = action_space['continuous']

env = SorterEnv(
        observation_type=0,
        render_mode='rgb_array',
    )
obs, _ = env.reset()
obs_dim = len(obs['observation'])
d_actor = DiscreteActor(obs_dim=obs_dim, output_dim=discrete_dim)
c_actor = ContinuousActor(obs_dim=obs_dim,
                          continuous_param_dim=continuous_dim)
critic = Critic(obs_dim=obs_dim)

trainer = Trainer(
    env=env,
    discrete_actor=d_actor,
    continuous_actor=c_actor,
    critic=critic,
    timesteps=2000,
    timesteps_per_batch=200,
    max_timesteps_per_episode=750,
)

trainer.train()