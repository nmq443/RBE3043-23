from environment import *
from model import DiscreteActor, ContinuousActor, Critic
import torch
from trainer import Trainer
from os import path
from pathlib import Path

action_space = {
    'discrete': {'Move': 0, 'Pick': 1, 'Place': 2},
    'continuous': [4, 4, 4]
}

discrete_dim = len(action_space['discrete'])
continuous_dim = action_space['continuous']

env = My_Arm_RobotEnv(
    observation_type=0,
    render_mode='human',
    blocker_bar=True,
    objects_count=2,
    sorting_count=2
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
    timesteps=200,
    timesteps_per_batch=50,
    max_timesteps_per_episode=10,
    device='cpu'
)

Path("./training").mkdir(parents=True, exist_ok=True)
if path.isfile("./training/state.data"):
    trainer.load("./training")

trainer.train()
