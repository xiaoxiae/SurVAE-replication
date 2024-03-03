import copy
from time import time

import torch
import torch.nn as nn

from survae.data import Dataset
from survae.layer import Layer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.double)


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

        self.size = None
        for l in reversed(layers):
            if l.out_size() is not None:
                self.size = l.out_size()
                break
        else:
            raise ValueError(f"SurVAE has layers TODO im lazey")

        self.layers = nn.ModuleList(layers)

    def get_name(self) -> str | None:
        return self.name

    def set_name(self, name: str):
        self.name = name

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        # TODO: optimize me (don't compute likelihood if it's not needed)
        ll_total = 0
        for layer in self.layers:
            X, ll = layer.forward(X, return_log_likelihood=True)
            ll_total += ll

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

    def train(self, dataset: Dataset, batch_size=1000, test_size=10000, epochs=1000, lr=0.01, log_count=10):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        # TODO: remove the duplicate code in this function
        start_time = time()
        x_test = dataset(test_size)
        trained_models = {}

        for epoch in range(epochs):
            optimizer.zero_grad()
            x_train = dataset(batch_size)
            z, ll = self(x_train, return_log_likelihood=True)

            loss = (0.5 * torch.sum(z ** 2) - ll) / batch_size
            loss.backward()

            optimizer.step()

            if epochs // log_count != 0 and (epoch + 1) % (epochs // log_count) == 0:
                z, ll = self(x_test, return_log_likelihood=True)
                loss_test = (0.5 * torch.sum(z ** 2) - ll) / test_size

                trained_models[epoch + 1] = (copy.deepcopy(self.state_dict()), loss.item(), loss_test.item())

        # save the last one regardless
        z, ll = self(x_test, return_log_likelihood=True)
        loss_test = (0.5 * torch.sum(z ** 2) - ll) / test_size
        trained_models[epoch + 1] = (self.state_dict(), loss.item(), loss_test.item())

        end_time = time()  # Record end time
        duration = end_time - start_time

        return trained_models

    def in_size(self) -> int | None:
        # TODO: this is broken
        return self.size

    def out_size(self) -> int | None:
        return self.size
