from panda_gym.envs.core import RobotTaskEnv
from experiment_env import *
from model import DiscreteActor, ContinuousActor, Critic
import torch

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

def run_episode():
    env = SorterEnv(render_mode='human', observation_type=0)

    """Run a single episode."""
    obs, _ = env.reset()
    print(f"Observation type: {type(obs)}")
    obs_dim = obs['observation'].shape[0]

    discrete = DiscreteActor(obs_dim, discrete_dim)
    continuous = ContinuousActor(obs_dim, continuous_dim)
    critic = Critic(obs_dim)

    timesteps = 0
    observations = []
    discrete_actions = []
    continuous_params = []
    discrete_log_probs = []
    continuous_log_probs = []
    rewards = []

    while timesteps < 100000:
        timesteps += 1

        observations.append(obs)
        if isinstance(obs, dict):
            obs = obs['observation']
        current_discrete_dist = discrete(obs)
        current_discrete_action = current_discrete_dist.sample()
        current_discrete_log_prob = current_discrete_dist.log_prob(
            current_discrete_action)
        current_discrete_action = current_discrete_action.detach().numpy()

        current_continuous_params = continuous(obs)
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
        obs, reward, terminated, _, _ = env.step(action)

        discrete_actions.append(current_discrete_action)
        discrete_log_probs.append(current_discrete_log_prob)

        continuous_params.append(current_continuous_action)
        continuous_log_probs.append(continuous_log_prob)

        rewards.append(reward)

        print(f"Timestep: {timesteps}")
        if terminated:
            break

run_episode()