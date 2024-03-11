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
    def __init__(self, layers: list[Layer | list], name: str | None = None, condition_size: int = 0, summary=lambda x: x) -> None:
        """
        General framework for the SurVAE-Flow architecture.

        ### Inputs:
        * layers: (Possibly nested) list of layers.
        * name: Human-recognizable name of the network.
        * condition_size: Size of the conditional input. 0 by default, i.e. unconditional.
        * summary: If the network is conditional, the condition gets passed through this function. Identity by default. 'condition_size' must be the output size of 'summary'.
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
        self.summary = summary

        self.condition_size = condition_size

        if condition_size != 0:
            self.make_conditional(condition_size)

    def make_conditional(self, size: int):
        for layer in self.layers:
            layer.make_conditional(size)

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

    def get_name(self) -> str | None:
        """Get the name of the network."""
        return self.name

    def set_name(self, name: str):
        """Set the name of the network."""
        self.name = name

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        if condition is not None:
            condition = self.summary(condition)

        ll_total = 0
        for layer in self.layers:
            if return_log_likelihood:
                X, ll = layer.forward(X, condition, return_log_likelihood=True)
                ll_total += ll
            else:
                X = layer.forward(X, condition, return_log_likelihood=False)

        if return_log_likelihood:
            return X, ll_total
        else:
            return X

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        if condition is not None:
            condition = self.summary(condition)

        for layer in reversed(self.layers):
            Z = layer.backward(Z, condition)

        return Z

    def sample(self, n: int, condition: torch.Tensor | int | None = None) -> torch.Tensor:
        # TODO: Tom did this and will do it again
        if condition is not None and not isinstance(condition, torch.Tensor):
            condition = torch.nn.functional.one_hot(
                torch.tensor([condition] * n, dtype=torch.long),
                num_classes=self.condition_size
            )

        with torch.no_grad():
            # sample from the code distribution, which should be the standard normal
            Z_sample = torch.normal(0, 1, size=(n, self.out_size()), device=DEVICE)

            # decode
            return self.backward(Z_sample, condition)

    def train(self, dataset: Dataset, batch_size: int, test_size: int, epochs: int, lr: float, log_period: int) \
            -> dict[int, TrainingSnapshot]:
        """Train the SurVAE model on the given dataset."""
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        def run(data, labels):
            if labels is not None:
                labels = Dataset.label_to_one_hot(labels.type(torch.long), self.condition_size)

            z, ll = self.forward(data, return_log_likelihood=True, condition=labels)
            loss = (0.5 * torch.sum(z ** 2) - ll) / len(data)

            return loss

        labels = self.condition_size != 0

        if labels:
            x_test, y_test = dataset.sample(test_size, labels=True)
        else:
            x_test = dataset.sample(test_size, labels=False)
            y_test = None

        trained_models = {}

        for epoch in range(epochs):
            optimizer.zero_grad()

            if labels:
                x_train, y_train = dataset.sample(batch_size, labels=True)
            else:
                x_train = dataset.sample(batch_size, labels=False)
                y_train = None

            loss = run(x_train, y_train)
            loss.backward()

            optimizer.step()

            # log possibly for period and always for the last iteration
            if (log_period != 0 and epoch % log_period == 0) or epoch == epochs - 1:
                loss_test = run(x_test, y_test)

                trained_models[epoch + 1] = TrainingSnapshot(
                    copy.deepcopy(self.state_dict()), loss.item(), loss_test.item())

        return trained_models

    def in_size(self) -> int | None:
        return self.in_s

    def out_size(self) -> int | None:
        return self.out_s
