from __future__ import annotations

from functools import cache

import numpy as np
import torch
from sklearn.datasets import fetch_openml


# TODO: documentation

def ngon(n: int, k: int = 8, noise: float = 0.01):
    indexes = np.floor(np.random.rand(n) * k)

    cov = np.array([[noise, 0], [0, noise]])

    X = np.array([(np.cos(index * 2 * np.pi / k), np.sin(index * 2 * np.pi / k)) for index in indexes])  # exact corners

    X = X + np.random.multivariate_normal([0.0, 0.0], cov, n)  # corners + deviation

    return torch.tensor(X)


def corners(n, r: float = 1, w: float = .5, l: float = 2):
    assert n % 2 == 0

    points = []

    for a, b in [(l, w), (w, l)]:
        p = np.column_stack((
            np.random.uniform(-1, 1, size=n // 2),
            np.random.uniform(-1, 1, size=n // 2)))

        p[:, 0] *= a
        p[:, 1] *= b

        p[:, 0][p[:, 0] < 0] -= r
        p[:, 0][p[:, 0] > 0] += r
        p[:, 1][p[:, 1] < 0] -= r
        p[:, 1][p[:, 1] > 0] += r

        points.append(p)

    return torch.tensor(np.concatenate(points))


def _circle(n, noise, radius):
    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, n)

    # Convert polar coordinates to Cartesian coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Add Gaussian noise to coordinates
    x += np.random.normal(0, noise, n)
    y += np.random.normal(0, noise, n)

    return np.column_stack((x, y)) * radius


def circles(n: int, k: int = 4, r1: int = 1, r2: int = 1.25, noise=0.025):
    assert n % k == 0

    points = []

    for i in range(k):
        alpha = (i / k) * 2 * np.pi

        x = np.cos(alpha) * r1
        y = np.sin(alpha) * r1

        p = _circle(n // k, noise=noise, radius=r2)
        p[:, 0] += x
        p[:, 1] += y

        points.append(p)

    return torch.tensor(np.concatenate(points))


def checkerboard(n: int, k: int = 4):
    assert k % 2 == 0

    # local tile coordinates
    x_coords = np.random.uniform(0, 1, size=n)
    y_coords = np.random.uniform(0, 1, size=n)

    points = np.column_stack((x_coords, y_coords))

    # move from local to global coordinates randomly
    for i in range(n):
        row_offset = np.random.randint(0, k)
        column_offset = ((np.random.randint(0, k)) * 2 + (row_offset % 2)) % k

        points[i][0] += row_offset
        points[i][1] += column_offset

    # center to origin
    points -= k / 2

    return torch.tensor(points)


@cache
def _get_mnist(name='mnist_784', version=1):
    # Fetch the MNIST dataset
    mnist = fetch_openml(name, version=version, cache=True)

    # Convert to numpy array
    mnist_images = np.array(mnist.data)

    # Normalize
    mnist_images_normalized = mnist_images / mnist_images.sum(axis=1, keepdims=True)

    return mnist_images_normalized


def _sample_points(image: np.ndarray, n: int = 50):
    """
    Expects a normalized & flattened image.
    """

    # Sample n points using the pixel values as weights
    sampled_indices = np.random.choice(len(image), size=n, p=image)

    # Assume the image is square
    n = int((image.shape[0]) ** (1 / 2))

    assert n ** 2 == image.shape[0]

    # Convert sampled indices back to coordinates in the original image
    sampled_points = np.unravel_index(sampled_indices, (n, n))

    sampled_points_array = np.column_stack(sampled_points)[:, ::-1] \
        .astype(float)

    # We want to uniformly sample from each pixel so offset randomly by +-0.5
    noise = np.random.uniform(-0.5, 0.5, size=sampled_points_array.shape)
    sampled_points_array += noise

    return sampled_points_array


def get_spatial_mnist(size: int, n: int = 50, flatten=True):
    images = _get_mnist()[:size]

    # Sample 'size' images randomly
    images = images[np.random.choice(len(images), size=size, replace=False)]

    sampled_points_all_images = []

    for i, image in enumerate(images):
        sampled_points = _sample_points(image, n)

        if flatten:
            sampled_points = sampled_points.flatten()

        sampled_points_all_images.append(sampled_points)

    return torch.tensor(np.array(sampled_points_all_images))


def spiral(n: int, radius: int = 2, angle: float = 5.1, noise: float = 0.2):
    # uniformly sample points on a line
    x_axis = torch.rand(n) * radius

    # add y-axis with gaussian noise
    line = torch.stack((x_axis, x_axis * torch.normal(torch.zeros_like(x_axis), noise * torch.ones_like(x_axis))),
                       dim=1)

    # add rotation
    angles = line.norm(dim=1) * angle
    points_cos = torch.cos(angles)
    points_sin = torch.sin(angles)

    points_x = points_cos * line[:, 0] + points_sin * line[:, 1]
    points_y = points_sin * line[:, 0] - points_cos * line[:, 1]

    points = torch.stack((points_x, points_y), dim=1)

    return points


class Dataset:
    """
    A dataset function. Includes convenient calling and cool augmentations.

    Batteries included!
    """

    @staticmethod
    def _skew(data, axis=0, q=0.5):
        """
        Take a dataset and randomly (with probability q) abs() values in the given axis.
        """
        mask = torch.rand(data.size()[0]) < q
        data[mask, axis] = torch.abs(data[mask, axis])

        return data

    def __init__(self, function, name: str | None = None):
        self.function = function
        self.modifiers = []

        self.name = name or function.__name__

    def __call__(self, n: int, **kwargs):
        data = self.function(n, **kwargs)

        for modifier in self.modifiers:
            data = modifier(data)

        return data

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    def skew(self, axis=0, q=0.5) -> Dataset:
        self.modifiers.append(lambda data: self._skew(data, axis=axis, q=q))
        return self

    def offset(self, vector: torch.Tensor) -> Dataset:
        self.modifiers.append(lambda data: data + vector)
        return self