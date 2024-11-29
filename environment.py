import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
        self.table = p.loadURDF(
            fileName="table/table.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )
        self.table_height = 0.62 # Approximate height of the table

        # Load UR10 robot
        self.ur10_base_position = [0, 0, self.table_height]
        self.ur10 = p.loadURDF(
            fileName=urdf_path,
            basePosition=self.ur10_base_position,
            useFixedBase=True
        )

        # Get joint info
        self.num_joints = p.getNumJoints(self.ur10)
        self.joint_indices = [i for i in range(self.num_joints)]
        self.joint_limits = [p.getJointInfo(self.ur10, i)[8:10] for i in self.joint_indices]

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )  # Action space: 6 continuous actions for Δx, Δy, Δz, Δroll, Δpitch, Δyaw

        obs_dim = 6 + 7 + 3 # 6 joint positions, 7 object pose, 3 target position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim, ),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        for i, (lower, upper) in enumerate(self.joint_limits):
            joint_angle = np.random.uniform(lower, upper)
            p.resetJointState(
                self.ur10,
                i,
                joint_angle
            )

        # self.object_id = p.loadURDF(
        #     fileName="cube.urdf",
        #     basePosition=[0.5, 1, 0],
        #     globalScaling=0.1
        # )
        colors = [
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1],  # Blue
            [1, 1, 0, 1],  # Yellow
        ]
        self.cube_ids = []
        cube_positions = self._generate_cube_positions()
        for i, position in enumerate(cube_positions):
            color = colors[i // 3]  # Assign color based on index
            cube_id = self._create_colored_cube(position, color)
            self.cube_ids.append(cube_id)
        # p.changeDynamics(self.object_id, -1, mass=1.0, lateralFriction=0.5, restitution=0.1)

        return self._get_observation(), {}
    def _generate_cube_positions(self):
        """Generate positions for 12 cubes randomly placed on the table."""
        positions = []
        for _ in range(12):
            x = np.random.uniform(-0.2, 0.2)  # Table surface x range
            y = np.random.uniform(-0.3, 0.3)  # Table surface y range
            z = self.table_height + 0.05  # Slightly above table
            positions.append([x, y, z])
        return positions

    def _create_colored_cube(self, position, color):
        """Create a colored cube at the specified position."""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.025, 0.025, 0.025],  # Small cube size
            rgbaColor=color
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.025, 0.025, 0.025]
        )
        cube_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        return cube_id

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
    urdf_path = "urdf/ur10_robot.urdf"  # Update with your URDF path
    
    env = UR10Env(urdf_path, render=True)
    obs = env.reset()
    rewards = []
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    env.close()