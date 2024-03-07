from __future__ import annotations

from functools import cache

import numpy as np
import torch
from sklearn.datasets import fetch_openml, make_moons


def ngon(n: int, k: int = 8, noise: float = 0.01, labels=False):
    """
    N Gaussians in a spherical pattern.

     o o
    o   o
    o   o
     o o

    :param k: How many dots.
    :param noise: Gaussian noise to add to the dots.
    """
    indexes = np.floor(np.random.rand(n) * k)

    cov = np.array([[noise, 0], [0, noise]])

    X = np.array([(np.cos(i * 2 * np.pi / k), np.sin(i * 2 * np.pi / k)) for i in indexes])  # exact corners

    X += np.random.multivariate_normal([0.0, 0.0], cov, n)

    y = np.array([i % k for i in indexes], dtype=int)

    if labels:
        return torch.tensor(X), torch.tensor(y).int()
    else:
        return torch.tensor(X)


def corners(n: int, middle_space: float = 1, width: float = .5, length: float = 2, labels=False):
    """
    4 corners.

      |   |
    --#   #--

    --#   #--
      |   |

    :param middle_space: Spacing of the rectangles (i.e. space in the middle).
    :param width:  Width of each rectangle.
    :param length: Length of each rectangle.
    """
    points = []

    for i, (a, b) in enumerate([(length, width), (width, length)]):
        if n % 2 == 1 and i == 1:
            m = n // 2 + 1
        else:
            m = n // 2

        p = np.column_stack((
            np.random.uniform(-1, 1, size=m),
            np.random.uniform(-1, 1, size=m)))

        p[:, 0] *= a
        p[:, 1] *= b

        p[:, 0][p[:, 0] < 0] -= middle_space
        p[:, 0][p[:, 0] > 0] += middle_space
        p[:, 1][p[:, 1] < 0] -= middle_space
        p[:, 1][p[:, 1] > 0] += middle_space

        points.append(p)

    X = torch.tensor(np.concatenate(points))

    if labels:
        signs = torch.sign(X)
        y = (signs[:, 0] >= 0) + 2 * (signs[:, 1] < 0)

        return X, y.int()
    else:
        return X


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


def circles(n: int, k: int = 4, circle_center_radius: int = 1, circle_radius: int = 1.25, noise=0.025, labels=False):
    """
    K overlapping circles.

    :param k: Number of circles
    :param circle_center_radius: Distance of circle centers to origin.
    :param circle_radius: The radius of the actual circles.
    """
    remainder = n % k

    points = []
    categories = []
    for i in range(k):
        alpha = (i / k) * 2 * np.pi

        # circle position
        x = np.cos(alpha) * circle_center_radius
        y = np.sin(alpha) * circle_center_radius

        if remainder == 0:
            m = n // k
        else:
            m = n // k + 1
            remainder -= 1

        # circle points
        p = _circle(m, noise=noise, radius=circle_radius)
        p[:, 0] += x
        p[:, 1] += y

        points.append(p)
        categories += [i] * len(p)

    if labels:
        return torch.tensor(np.concatenate(points)), torch.tensor(categories).int()
    else:
        return torch.tensor(np.concatenate(points))


def checkerboard(n: int, k: int = 2, scale: float = 1, labels=False):
    """A checkerboard of size k^2."""
    k **= 2

    # local tile coordinates
    x_coords = np.random.uniform(0, 1, size=n)
    y_coords = np.random.uniform(0, 1, size=n)

    points = np.column_stack((x_coords, y_coords))
    categories = []

    # move from local to global coordinates randomly
    for i in range(n):
        row_offset = np.random.randint(0, k)
        column_offset = ((np.random.randint(0, k)) * 2 + (row_offset % 2)) % k

        categories.append(row_offset * k + column_offset)

        points[i][0] += row_offset
        points[i][1] += column_offset

    # center to origin
    points -= k / 2

    X = torch.tensor(points) * scale

    if labels:
        return X, torch.tensor(categories).int()
    else:
        return X


@cache
def _get_mnist(name='mnist_784', version=1):
    # Fetch the MNIST dataset
    mnist = fetch_openml(name, version=version, cache=True)

    # Convert to numpy array
    mnist_images = np.array(mnist.data)
    mnist_labels = np.array(mnist.target.astype(int))

    # Normalize
    mnist_images_normalized = mnist_images / mnist_images.sum(axis=1, keepdims=True)

    return mnist_images_normalized, mnist_labels


def _sample_points(image: np.ndarray, n: int = 50):
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


def spatial_mnist(n: int, k: int = 50, flatten=True, labels=False):
    """
    Generate a spatial MNIST dataset by taking normalized MNIST images as density and sampling k points.

    :param k: How many points to sample.
    :param flatten: Whether to flatten the result.
    """
    images, y = _get_mnist()

    # Sample 'size' images randomly
    idxs = np.random.choice(len(images), size=n, replace=False)
    images = images[idxs]
    y = y[idxs]

    sampled_points_all_images = []

    for i, image in enumerate(images):
        sampled_points = _sample_points(image, k)

        if flatten:
            sampled_points = sampled_points.flatten()

        sampled_points_all_images.append(sampled_points)

    X = torch.tensor(np.array(sampled_points_all_images))
    y = torch.tensor(y)

    if labels:
        return X, y.int()
    else:
        return X


def spiral(n, noise=0.1, labels=False):
    """
    Generate a double spiral.
    """
    t = (torch.linspace(0, 1, n // 2 + n % 2)) ** (1 / 2) * np.pi * 3
    r = (torch.linspace(0, 1, n // 2 + n % 2)) ** (1 / 2) * 5  # Linearly increasing radius

    # Spiral 1
    x1 = (r * torch.cos(t))
    y1 = (r * torch.sin(t))

    # Spiral 2, rotating by 180 degrees
    x2 = r * torch.cos(t - np.pi)
    y2 = r * torch.sin(t - np.pi)

    # Add Gaussian noise if specified
    if noise > 0.0:
        x1 += torch.randn_like(t) * noise
        y1 += torch.randn_like(t) * noise
        x2 += torch.randn_like(t) * noise
        y2 += torch.randn_like(t) * noise

    data1 = torch.stack((x1, y1), dim=1)
    data2 = torch.stack((x2, y2), dim=1)
    double_spiral_data = torch.cat((data1, data2), dim=0)

    X = double_spiral_data[:n]

    if labels:
        y = torch.tensor([0] * len(data1) + [1] * len(data2))[:n]
        return X, y.int()
    else:
        return X


def moons(n, delta: float = 0.1, labels=False):
    X, y = make_moons(n_samples=n, noise=delta, random_state=42)

    if labels:
        return torch.tensor(X), torch.tensor(y).int()
    else:
        return torch.tensor(X)


def split_line(n, delta: float = 0.005, k: float = 1, labels=False):
    mu = np.array([0, 0])
    sigma = np.array([[1, 1], [-delta, delta]])

    data = np.random.multivariate_normal(mu, sigma, n)

    X = torch.tensor(data)
    y = torch.zeros(n)

    x_values = X[:, 0]
    y_values = X[:, 1]

    y_values[x_values >= 0] += k
    y_values[x_values < 0] -= k

    y[x_values >= 0] = 1

    X[:, 1] = y_values

    if labels:
        return X, y.int()
    else:
        return X


class Dataset:
    """
    A dataset function. Includes convenient calling and cool augmentations.

    Batteries included!
    """

    def __init__(self, function, name: str | None = None, labels=False, shuffle=True, **kwargs):
        self.function = function
        self.modifiers = []

        self.kwargs = kwargs
        self.labels = labels
        self.shuffle = shuffle

        self.name = name or function.__name__

    @staticmethod
    def _skew(data, axis=0, q=0.5):
        """
        Take a dataset and randomly (with probability q) abs() values in the given axis.
        """
        mask = torch.rand(data.size()[0]) < q
        data[mask, axis] = torch.abs(data[mask, axis])

        return data

    def __call__(self, n: int):
        result = self.function(n, labels=self.labels, **self.kwargs)

        # Get data / data and labels
        if self.labels:
            data, labels = result
        else:
            data = result

        # Possibly shuffle data
        if self.shuffle:
            data = data[torch.randperm(n)]

        # Possibly modify data
        for modifier in self.modifiers:
            data = modifier(data)

        # Return data / data + labels

        if self.labels:
            return data, labels
        else:
            return data

    def get_name(self) -> str:
        """Get the dataset name."""
        return self.name

    def set_name(self, name: str):
        """Set the dataset name."""
        self.name = name

    def skew(self, axis=0, q=0.5) -> Dataset:
        """Apply a modifier that flips value of the dataset in the specified axis with probability q."""
        self.modifiers.append(lambda data: self._skew(data, axis=axis, q=q))
        return self

    def offset(self, vector: torch.Tensor) -> Dataset:
        """Apply a modifier that offsets all values of the dataset by a given vector."""
        self.modifiers.append(lambda data: data + vector)
        return self
