import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time

class UR10Env(gym.Env):
    def __init__(self, urdf_path, render=False):
        super().__init__()
        self.render = render
        self.time_step = 1 / 10  # Simulation timestep
        self.max_steps = 1000    # Maximum steps per episode
        self.current_step = 0

        # Connect to PyBullet
        if self.render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")

        # Load UR10 robot
        self.ur10 = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        # Get joint info
        self.num_joints = p.getNumJoints(self.ur10)
        self.joint_indices = [i for i in range(self.num_joints)]
        self.joint_limits = [p.getJointInfo(self.ur10, i)[8:10] for i in self.joint_indices]

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)  # Control 6 joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_joints * 2,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        for i, (lower, upper) in enumerate(self.joint_limits):
            joint_angle = np.random.uniform(lower, upper)
            p.resetJointState(self.ur10, i, joint_angle)

        return self._get_observation()

    def step(self, action):
        self.current_step += 1

        # Scale actions to joint limits
        scaled_actions = []
        for i, act in enumerate(action):
            lower, upper = self.joint_limits[i]
            scaled_actions.append(lower + (act + 1) * 0.5 * (upper - lower))
            p.setJointMotorControl2(self.ur10, i, p.POSITION_CONTROL, scaled_actions[i])

        p.stepSimulation()
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps
        info = {}

        if self.render:
            time.sleep(self.time_step)

        return obs, reward, done, info

    def _get_observation(self):
        obs = []
        for i in self.joint_indices:
            joint_state = p.getJointState(self.ur10, i)
            obs.append(joint_state[0])  # Joint position
            obs.append(joint_state[1])  # Joint velocity
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # Example reward: Minimize distance to a target position
        target_position = [0.5, 0.5, 0.5]  # Example target
        ee_state = p.getLinkState(self.ur10, self.num_joints - 1)  # End-effector
        ee_position = ee_state[0]
        dist = np.linalg.norm(np.array(ee_position) - np.array(target_position))
        return -dist  # Negative distance as reward

    def close(self):
        p.disconnect(self.client)

if __name__ == "__main__":
    urdf_path = "/home/hoang/rl/RBE3043-23/urdf/ur10_robot.urdf"  # Update with your URDF path
    
    env = UR10Env(urdf_path, render=True)
    obs = env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break

    env.close()
