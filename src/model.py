import torch
from torch import Tensor
from torch.nn import Sequential, Module, Linear, ModuleList
from torch.nn import LeakyReLU
import numpy as np

from typing import Union, List


class StateEncoder(Module):
    """Shared state encoder network for both discrete and continuous actor"""

    def __init__(
            self,
            input_size
    ):
        super(StateEncoder, self).__init__()

        self.fc = Sequential(
            Linear(input_size, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class DiscreteActor(Module):
    def __init__(
            self,
            input_size: int = 20,
            output_size: int = 4
    ):
        super(DiscreteActor, self).__init__()

        # Determine our input size
        self.input_size = input_size
        # Determine our output size
        self.output_size = output_size

        # Create our model head - input -> some output layer before splitting to
        # different outputs for mu and sigma
        self.model = Sequential(
            Linear(256, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, self.output_size),
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
        return output

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
            input_size: int = 20,
            continuous_action_dim: int = [3, 1, 3, 1]
    ):
        super(ContinuousActor, self).__init__()

        # Determine our input size
        self.input_size = input_size

        # Create our model head - input -> some output layer before splitting to
        # different outputs for mu and sigma
        self.mean_model = ModuleList(
            Sequential(
                Linear(256, 256),
                LeakyReLU(),
                Linear(256, 128),
                LeakyReLU(),
                Linear(128, 64),
                LeakyReLU(),
                Linear(64, param_dim),
            )
            for param_dim in continuous_action_dim
        )

        self.std_model = ModuleList(
            Sequential(
                Linear(256, 256),
                LeakyReLU(),
                Linear(256, 128),
                LeakyReLU(),
                Linear(128, 64),
                LeakyReLU(),
                Linear(64, param_dim),
            )
            for param_dim in continuous_action_dim
        )

    def forward(self, input: Union[np.ndarray, Tensor, List]):
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(
                np.array(input).astype("float32"))
        else:
            input_tensor = input

        # return distribution
        means = [head(input_tensor) for head in self.mean_model]
        stds = [head(input_tensor) for head in self.std_model]
        return torch.distributions.Normal(means, stds)


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
            input_size: int
    ):
        super(Critic, self).__init__()

        # Our score can be unbounded as a value from
        # some -XXX, +XXX, so we don't scale it w/ an activation
        # function

        self.model = Sequential(
            Linear(input_size, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 1),
        )

    def forward(self, input: np.ndarray) -> Tensor:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(
                np.array(input).astype("float32"))
        else:
            input_tensor = input

        # activation1 = F.relu(self.layer1(input_tensor))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2)

        # return output

        return self.model(input_tensor)

    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])

