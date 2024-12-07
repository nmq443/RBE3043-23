from env import MyRobotTaskEnv
from training import Trainer
from model import StateEncoder, DiscreteActor, ContinuousActor, Critic

from os import path
from pathlib import Path

env = MyRobotTaskEnv(render_mode="human")

state_encoder = StateEncoder(5)
discrete_actor = DiscreteActor(5)
continuous_actor = ContinuousActor(5, [1, 1, 1, 1])
critic = Critic(5)

trainer = Trainer(
    env=env,
    state_encoder=state_encoder,
    discrete_actor=discrete_actor,
    continuous_actor=continuous_actor,
    critic=critic,
    timesteps=200_000_000,
    timesteps_per_batch=5_000,
    max_timesteps_per_episode=750,
)

# Determine if there is a training state at training/state.data -
# if so, load the TrainingState for it. Otherwise, create a new one.

trainer.train()