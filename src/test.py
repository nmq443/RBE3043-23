
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

def test_model():
    obj_pos_dim = 3
    target_dim = 3
    ee_ori_dim = 3
    ee_pos_dim = 3
    joints_dim = 7
    gripper_dim = 2
    obs_dim = (obj_pos_dim + target_dim + ee_ori_dim + ee_pos_dim + joints_dim + gripper_dim)

    dummy_input = torch.randn(obs_dim, dtype=torch.float32)
    discrete = DiscreteActor(obs_dim, discrete_dim)
    continuous = ContinuousActor(obs_dim, continuous_dim)
    critic = Critic(obs_dim)

    discrete_dist = discrete(dummy_input)
    continuous_params = continuous(dummy_input)
    V = critic(dummy_input)

    print(f"Discrete distribution: {discrete_dist}")
    print(f"Continuous param: {continuous_params}")
    print(f"V: {V}")
def init():
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

    return trainer

if __name__ == '__main__':
    trainer = init()
    # trainer.run_episode()

    rewards = [1, 2, 3, 4, 5]
    print(trainer.calculate_discounted_reward(rewards))