from typing import Any, Dict, Tuple
import numpy as np
from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
import time


class MyPickAndPlace(Task):
    def __init__(
            self,
            sim: PyBullet,
            distance_threshold: float = 0.05,
            use_blocking_bar=True
    ):
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.use_blocking_bar = use_blocking_bar
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self):
        """
        Create a scene that will have a plane, a table
        :return: None
        """
        # Create a plane and a table
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(
            length=2,
            width=2,
            height=0.4,
            x_offset=-0.3
        )

        # Create object and target to put object to
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0, 0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1])
        )
        self.target_position = np.array([-0.25, 0.00, 0.01])
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([0.05, 0.1, 0.01]),
            mass=0.0,
            ghost=False,
            position=np.array([-0.5, 0.00, 0.01]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.4])
        )

        if self.use_blocking_bar:
            # Create a blocking bar from the object to the target
            self.sim.create_box(
                body_name="blocker",
                half_extents=np.array([0.01, 0.3, 0.02]),
                mass=0,
                ghost=False,
                position=np.array([0.1, 0.0, 0.01]),
                rgba_color=np.array([0.0, 0.0, 0.0, 0.8])
            )

        # Limit object to be created to specific areas
        self.object_position_limits: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.15, 0.25),
            (0.15, 0.25)
        )

    def reset(self):
        """TODO
        Make more complicated objects (multiple shapes for example ...)
        """

        # Reset the object position randomly
        object_position = self._sample_object()
        self.sim.set_base_pose(
            body="object",
            position=object_position,
            orientation=np.array([0, 0, 0, 1])
        )
        self.goal = np.array([0, 0, 0.05])
        self.sim.set_base_pose(
            body="target",
            position=self.goal,
            orientation=np.array([0, 0, 0, 1])
        )

    def _sample_object(self) -> np.ndarray:
        """
        Randomize start position of the object
        :return: object position
        """
        x = np.random.uniform(
            low=self.object_position_limits[0][0],
            high=self.object_position_limits[0][1],
        )
        y = np.random.uniform(
            low=self.object_position_limits[1][0],
            high=self.object_position_limits[1][1],
        )
        return np.array([x, y, 0.01])

    def get_obs(self) -> np.ndarray:
        """
        Observation consists of the transform and velocities of the object. The transform contains position and rotation/orientation, and the velocities are velocity and angular velocity
        :return: The observation in a `numpy.ndarray` type
        """

        # TODO: add transform for target, end-effector/gripper

        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_orientation("object")
        object_velocity = self.sim.get_base_velocity("object")
        # object_angular_velocity = self.sim.get_base_angular_velocity("object")
        # observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        observation = np.concatenate([object_position, object_rotation])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        """
        :return: The achieved goal
        """
        # TODO: make a better achieved goal
        object_position = self.sim.get_base_position("object")
        return np.array(object_position)

    def is_terminated(self) -> bool:
        """
        is_terminated returns whether or not the episode is
        in a terminal state; this can be due to:
        1. All objects have been removed somehow from the env -> object
        exists in environment or not?
        2. The timer has hit 0

        It is not an indication of success
        """

        # TODO: re-implement this method later

        # return all(obj.removed for obj in self.goal.values())
        return False

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        # d = distance(achieved_goal, desired_goal)
        # return np.array(d < self.distance_threshold, dtype=bool)
        # return self.is_terminated()

        return np.array([self.is_terminated()], dtype="bool")

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        '''
        TODO: Design another reward function for H-PPO
        '''
        return -np.array(d > self.distance_threshold, dtype=np.float32)


class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(
            self,
            render_mode,
            use_blocking_bar=True
    ):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.4, 0, 0]))
        task = MyPickAndPlace(sim, use_blocking_bar=use_blocking_bar)
        super().__init__(robot, task)

    def step(self):
        # TODO: step() return obs, reward, terminated, truncated, info
        return None

from utils import add_world_frame

def test_env():

    env = MyRobotTaskEnv(render_mode="human", use_blocking_bar=False)
    add_world_frame()
    observation, info = env.reset()

    for _ in range(10000):
        time.sleep(1/24)
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()


if __name__ == '__main__':
    test_env()
