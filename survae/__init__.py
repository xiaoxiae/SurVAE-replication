import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from survae.data import Dataset
from survae.layer import Layer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.double)


@dataclass
class TrainingSnapshot:
    model_state: dict
    training_loss: float
    testing_loss: float


class SurVAE(Layer):
    @staticmethod
    def _flatten_list(nested_list: list[Layer | list]) -> list[Layer]:
        """Flatten a nested list of layers."""
        flattened_list = []
        for item in nested_list:
            if isinstance(item, Layer):
                flattened_list.append(item)
            else:
                flattened_list.extend(SurVAE._flatten_list(item))

        return flattened_list

    def __init__(self, layers: list[Layer | list], name: str | None = None):
        """
        General framework for the SurVAE-Flow architecture.
        """
        super().__init__()

        layers = SurVAE._flatten_list(layers)

        self.name = name

        self.out_s = None
        for l in reversed(layers):
            if l.out_size() is not None:
                self.out_s = l.out_size()
                break
        else:
            raise ValueError(f"SurVAE doesn't have any fixed-out-size layers!")

        self.in_s = None
        for l in layers:
            if l.in_size() is not None:
                self.in_s = l.in_size()
                break
        else:
            raise ValueError(f"SurVAE doesn't have any fixed-in-size layers!")

        self.layers = nn.ModuleList(layers)

    def get_name(self) -> str | None:
        """Get the name of the network."""
        return self.name

    def set_name(self, name: str):
        """Set the name of the network."""
        self.name = name

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        ll_total = 0
        for layer in self.layers:
            if return_log_likelihood:
                X, ll = layer.forward(X, return_log_likelihood=True)
                ll_total += ll
            else:
                X = layer.forward(X, return_log_likelihood=False)

        if return_log_likelihood:
            return X, ll_total
        else:
            return X

    def backward(self, Z: torch.Tensor):
        for layer in reversed(self.layers):
            Z = layer.backward(Z)

        return Z

    def sample(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            # sample from the code distribution, which should be the standard normal
            Z_sample = torch.normal(0, 1, size=(n, self.out_size()), device=DEVICE)

            # decode
            return self.backward(Z_sample)

    def train(self, dataset: Dataset, batch_size: int, test_size: int, epochs: int, lr: float, log_period: int) \
            -> dict[int, TrainingSnapshot]:
        """Train the SurVAE model on the given dataset."""
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        def run(data):
            z, ll = self(data, return_log_likelihood=True)
            loss = (0.5 * torch.sum(z ** 2) - ll) / len(data)

            return loss

        x_test = dataset(test_size)
        trained_models = {}

        for epoch in range(epochs):
            optimizer.zero_grad()
            x_train = dataset(batch_size)

            loss = run(x_train)
            loss.backward()

            optimizer.step()

            # log possibly for period and always for the last iteration
            if (log_period != 0 and epoch % log_period == 0) or epoch == epochs - 1:
                loss_test = run(x_test)

                trained_models[epoch + 1] = TrainingSnapshot(
                    copy.deepcopy(self.state_dict()), loss.item(), loss_test.item())

        return trained_models

    def in_size(self) -> int | None:
        return self.in_s

    def out_size(self) -> int | None:
        return self.out_s
