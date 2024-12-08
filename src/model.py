import torch
from torch import Tensor
from torch.nn import (Sequential, Module, Linear, ModuleList, Softplus,
                      ModuleDict, ModuleList)
from torch.nn import LeakyReLU
import numpy as np
from typing import Union, List

class DiscreteActor(Module):
    def __init__(
            self,
            obs_dim: int = 20,
            output_dim: int = 3,
            control_type=None
    ):
        """Init the discrete actor. This network estimate a distribution of
        discrete actions.
        Args:
            obs_dim (int, optional): Dimension of observation space. Defaults to 20.
            output_size (int, optional): Output size or number of discrete
            actions. Defaults to 3 (Move, Pick, Place)
        """
        super(DiscreteActor, self).__init__()

        if control_type is not None and control_type == 'pendulum':
            obs_dim = 3
            output_dim = 1

        self.model = Sequential(
            Linear(obs_dim, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, output_dim),
        )

    def forward(self, input: Union[np.ndarray, Tensor, List]) -> Union[
        np.ndarray, Tensor, List]:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(
                np.array(input).astype("float32"))
        else:
            input_tensor = input

        # return distribution
        output = self.model(input_tensor)
        output_dist = torch.distributions.Categorical(logits=output)
        return output_dist

    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])


class ContinuousActor(Module):
    def __init__(
            self,
            obs_dim: int = 20,
            continuous_param_dim: List = [3, 1, 3, 1],
            control_type=None
    ):
        """Init the continuous actor. This network predicts mean and std for
        the continuous parameters.
        Args:
            obs_dim (int, optional): Dimension of observation space. Defaults to 20.
            continuous_param_dim (int, optional): Dimension of continuous
            parameter. Defaults to [1, 1, 1, 1], meaning each discrete action only has 1 parameter
        """
        super(ContinuousActor, self).__init__()

        if control_type is not None and control_type == 'pendulum':
            obs_dim = 3
            continuous_param_dim = [1]

        self.model = ModuleList(
            ModuleDict({
                "mean": Sequential(
                    Linear(obs_dim, 256),
                    LeakyReLU(),
                    Linear(256, 128),
                    LeakyReLU(),
                    Linear(128, 64),
                    LeakyReLU(),
                    Linear(64, param_dim)
                ),
                "std": Sequential(
                    Linear(obs_dim, 256),
                    LeakyReLU(),
                    Linear(256, 128),
                    LeakyReLU(),
                    Linear(128, 64),
                    LeakyReLU(),
                    Linear(64, param_dim),
                    Softplus()  # Ensures positive standard deviations
                )
            })
            for param_dim in continuous_param_dim
        )

    def forward(self, input: Union[np.ndarray, Tensor, List]):
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(
                np.array(input).astype("float32"))
        else:
            input_tensor = input

        continuous_params = [
            {
                "mean": head["mean"](input_tensor),
                "std": head["std"](input_tensor)
            }
            for head in self.model
        ]

        return continuous_params


    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])


class Critic(Module):
    def __init__(
            self,
            obs_dim: int,
            control_type=None
    ):
        """Init the critic network. This network estimate V(s)"""
        super(Critic, self).__init__()

        if control_type is not None and control_type == 'pendulum':
            obs_dim = 3

        self.model = Sequential(
            Linear(obs_dim, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 1),
        )

    def forward(self, input: np.ndarray) -> Tensor:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(
                np.array(input).astype("float32"))
        else:
            input_tensor = input

        return self.model(input_tensor)

    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])

