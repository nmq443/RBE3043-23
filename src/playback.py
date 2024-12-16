from environment import *
from model import *
from PIL import Image
from typing import Dict, List
from time import sleep

FPS = 60
FRAME_DELAY = 1.0/FPS
MAX_EPISODE_LENGTH_SECONDS = 5
RECORDING_EPISODE_COUNT = 10
DISCRETE_MODEL_PATH = "./training/discrete_actor.pth"
CONTINUOUS_MODEL_PATH = "./training/continuous_actor.pth"
OUTPUT_FILE = "./playback.gif"

action_space = {
    'discrete': {'Move': 0, 'Pick': 1, 'Place': 2},
    'continuous': [4, 4, 4]
}

env = My_Arm_RobotEnv(
    observation_type=0,
    render_mode="rgb_array",
    renderer="OpenGL",
    blocker_bar=True,
    objects_count=1,
    sorting_count=1,
    actions=action_space,
)

action_space = {
    'discrete': {'Move': 0, 'Pick': 1, 'Place': 2},
    'continuous': [4, 4, 4]
}

discrete_dim = len(action_space['discrete'])
continuous_dim = action_space['continuous']
obs, _ = env.reset()
obs_dim = len(obs['observation'])

discrete_actor = DiscreteActor(obs_dim=obs_dim, output_dim=discrete_dim)
discrete_actor.load(DISCRETE_MODEL_PATH)
continuous_actor = ContinuousActor(obs_dim=obs_dim, continuous_param_dim=continuous_dim)
continuous_actor.load(CONTINUOUS_MODEL_PATH)

episode = 0
episode_length = 0
frames = []
observation, info = env.reset()

while episode < RECORDING_EPISODE_COUNT:
    print(episode)
    episode_length += 1
    if isinstance(observation, Dict):
        observation = observation["observation"]
    current_discrete_action = discrete_actor(observation).sample().detach().numpy()
    current_continuous_params = continuous_actor(observation)
    mean = current_continuous_params[current_discrete_action][
        'mean']
    std = current_continuous_params[current_discrete_action][
        'std']
    current_continuous_dist = torch.distributions.Normal(mean, std)
    current_continuous_action = current_continuous_dist.sample()
    current_continuous_action = current_continuous_action.detach().cpu().numpy()

    action = {
        'discrete': current_discrete_action,
        'continuous': current_continuous_action
    }

    observation, reward, terminated, truncated, info = env.step(action)

    '''
    img = Image.fromarray(env.render())
    frames.append(img)
    '''

    if episode_length >= FPS * MAX_EPISODE_LENGTH_SECONDS:
        terminated = True

    if terminated or truncated:
        episode += 1
        episode_length = 0
        observation, info = env.reset()

env.close()

'''
Image.new('RGB', frames[0].size).save(
    fp=OUTPUT_FILE,
    format='GIF',
    append_images=frames,
    save_all=True,
    duration=FRAME_DELAY*len(frames),
    loop=0,
)
'''
