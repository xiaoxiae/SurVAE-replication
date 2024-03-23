from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache

import numpy as np
import torch
from sklearn.datasets import fetch_openml, make_moons
from torchvision import datasets

import survae


class Dataset(ABC):
    """
    A dataset function. Includes convenient calling and cool augmentations.

    Batteries included!
    """

    def __init__(self, shuffle=True, name=None):
        self.modifiers = []
        self.shuffle = shuffle

        self._name = name

    def get_name(self) -> str:
        """Get the dataset name."""
        return self._name or self.__class__.__name__

    @abstractmethod
    def get_categories(self) -> int:
        """Get the number of categories of the dataset."""
        pass

    @abstractmethod
    def __call__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample(self, n: int, labels=False):
        data, y = self(n)

        # Possibly shuffle data
        if self.shuffle:
            permutation = torch.randperm(n)
            data = data[permutation]
            y = y[permutation]

        # Possibly modify data
        for modifier in self.modifiers:
            data = modifier(data)

        # Return data / data + labels
        if labels:
            return data, y
        else:
            return data

    def skew(self, axis=0, q=0.5) -> Dataset:
        """Apply a modifier that flips value of the dataset in the specified axis with probability q."""

        def _skew(data, axis=0, q=0.5):
            """
            Take a dataset and randomly (with probability q) abs() values in the given axis.
            """
            mask = torch.rand(data.size()[0]) < q
            data[mask, axis] = torch.abs(data[mask, axis])

            return data

        self.modifiers.append(lambda data: _skew(data, axis=axis, q=q))
        return self

    def offset(self, vector: torch.Tensor) -> Dataset:
        """Apply a modifier that offsets all values of the dataset by a given vector."""
        self.modifiers.append(lambda data: data + vector)
        return self

    @classmethod
    def label_to_one_hot(cls, labels: torch.Tensor, size: int) -> torch.Tensor:
        """Convert labels to one-hot encoding."""
        return torch.nn.functional.one_hot(labels, num_classes=size)


class Ngon(Dataset):
    """
    N Gaussians in a spherical pattern.

     o o
    o   o
    o   o
     o o
    """

    def __init__(self, k: int = 8, noise: float = 0.01, **kwargs):
        """
        :param k: How many dots.
        :param noise: Gaussian noise to add to the dots.
        """
        super().__init__(**kwargs)

        self._k = k
        self._noise = noise

    def get_categories(self) -> int:
        return self._k

    def __call__(self, n: int):
        indexes = np.floor(np.random.rand(n) * self._k)

        cov = np.array([[self._noise, 0], [0, self._noise]])

        X = np.array(
            [(np.cos(i * 2 * np.pi / self._k), np.sin(i * 2 * np.pi / self._k)) for i in indexes])  # exact corners

        X += np.random.multivariate_normal([0.0, 0.0], cov, n)
        y = np.array([i % self._k for i in indexes], dtype=int)

        return torch.tensor(X), torch.tensor(y).int()


class Corners(Dataset):
    """
    4 corners.

      |   |
    --#   #--

    --#   #--
      |   |
    """

    def __init__(self, middle_space: float = 1, width: float = .5, length: float = 2, **kwargs):
        """
        :param middle_space: Spacing of the rectangles (i.e. space in the middle).
        :param width:  Width of each rectangle.
        :param length: Length of each rectangle.
        """
        super().__init__(**kwargs)

        self._middle_space = middle_space
        self._width = width
        self._length = length

    def get_categories(self) -> int:
        return 4

    def __call__(self, n: int):
        points = []

        for i, (a, b) in enumerate([(self._length, self._width), (self._width, self._length)]):
            if n % 2 == 1 and i == 1:
                m = n // 2 + 1
            else:
                m = n // 2

            p = np.column_stack((
                np.random.uniform(-1, 1, size=m),
                np.random.uniform(-1, 1, size=m)))

            p[:, 0] *= a
            p[:, 1] *= b

            p[:, 0][p[:, 0] < 0] -= self._middle_space
            p[:, 0][p[:, 0] > 0] += self._middle_space
            p[:, 1][p[:, 1] < 0] -= self._middle_space
            p[:, 1][p[:, 1] > 0] += self._middle_space

            points.append(p)

        X = torch.tensor(np.concatenate(points))

        signs = torch.sign(X)
        y = (signs[:, 0] >= 0) + 2 * (signs[:, 1] < 0)

        return X, y.int()


class Circles(Dataset):
    """
    K overlapping circles.
    """

    def __init__(self, k: int = 4, circle_center_radius: float = 1,
                 circle_radius: float = 1.25, noise: float = 0.025, **kwargs):
        """
        :param k: Number of circles
        :param circle_center_radius: Distance of circle centers to origin.
        :param circle_radius: The radius of the actual circles.
        :param noise: Gaussian nosie to add to the circles.
        """
        super().__init__(**kwargs)

        self._k = k
        self._circle_center_radius = circle_center_radius
        self._circle_radius = circle_radius
        self._noise = noise

    def get_categories(self) -> int:
        return self._k

    def __call__(self, n: int):
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

        remainder = n % self._k

        points = []
        categories = []
        for i in range(self._k):
            alpha = (i / self._k) * 2 * np.pi

            # circle position
            x = np.cos(alpha) * self._circle_center_radius
            y = np.sin(alpha) * self._circle_center_radius

            if remainder == 0:
                m = n // self._k
            else:
                m = n // self._k + 1
                remainder -= 1

            # circle points
            p = _circle(m, noise=self._noise, radius=self._circle_radius)
            p[:, 0] += x
            p[:, 1] += y

            points.append(p)
            categories += [i] * len(p)

        return torch.tensor(np.concatenate(points)), torch.tensor(categories).int()


class Checkerboard(Dataset):
    """
    K overlapping circles.
    """

    def __init__(self, k: int = 2, **kwargs):
        """
        :param k: sqrt(k) of the keyboard.
        """
        super().__init__(**kwargs)

        self._k = k ** 2

    def get_categories(self) -> int:
        return self._k ** 2

    def __call__(self, n: int):
        # local tile coordinates
        x_coords = np.random.uniform(0, 1, size=n)
        y_coords = np.random.uniform(0, 1, size=n)

        points = np.column_stack((x_coords, y_coords))
        categories = []

        # move from local to global coordinates randomly
        for i in range(n):
            row = np.random.randint(0, self._k)
            column = np.random.randint(0, self._k)

            row_offset = row
            column_offset = (column * 2 + (row_offset % 2)) % self._k

            categories.append(row * self._k + column)

            points[i][0] += row_offset
            points[i][1] += column_offset

        # center to origin
        points -= self._k / 2

        X = torch.tensor(points)

        return X, torch.tensor(categories).int()


def _load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function for MNIST_784 and SpatialMNIST dataset.
    This is needed because torchvision.datasets.MNIST doesn't work for Mary and
    fetch_openml doesn't work for Jannis, so this function attempts both.

    Returns MNIST image data (flattened) and target as numpy arrays.
    """
    try:
        # Fetch the MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, parser='auto', cache=True)

        # Convert to numpy array
        mnist_images = np.array(mnist.data)
        mnist_labels = np.array(mnist.target.astype(int))
    except Exception:
        # Fetch the MNIST dataset
        mnist = datasets.MNIST(root='./data', train=True, download=True)

        mnist_images = np.array((mnist.data).double().flatten(start_dim=1))
        mnist_labels = np.array(mnist.targets)
    
    return mnist_images, mnist_labels



class MNIST_784(Dataset):
    """
    Load the 28x28 MNIST dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # immediately load the dataset for later use
        # mnist = datasets.MNIST(root='./data', train=True, download=True)

        # self.mnist_images = (mnist.data).double().flatten(start_dim=1).to(survae.DEVICE)
        # self.mnist_labels = mnist.targets.to(survae.DEVICE)
        mnist_images, mnist_labels = _load_mnist()
        self.mnist_images = torch.tensor(mnist_images, dtype=torch.double)
        self.mnist_labels = torch.tensor(mnist_labels, dtype=torch.long)

        self.size = len(self.mnist_labels)
    
    def get_categories(self) -> int:
        return 10
    
    def __call__(self, n: int):
        # if n > self.size, elements may be sampled multiple times
        perm = (torch.randperm(max(n, self.size)) % self.size)[:n]
        X = self.mnist_images[perm]
        y = self.mnist_labels[perm]
        return X, y


class SpatialMNIST(Dataset):
    """
    Generate a spatial MNIST dataset by taking normalized MNIST images as density and sampling k points.
    """

    def __init__(self, k: int = 2, flatten: bool = True, **kwargs):
        """
        :param k: How many points to sample.
        :param flatten: Whether to flatten the result.
        """
        super().__init__(**kwargs)

        self._k = k
        self._flatten = flatten

        # Fetch the MNIST dataset
        # mnist = fetch_openml('mnist_784', version=1, parser='auto', cache=True)

        # # Convert to numpy array
        # mnist_images = np.array(mnist.data)
        # mnist_labels = np.array(mnist.target.astype(int))
        mnist_images, mnist_labels = _load_mnist()

        # Normalize
        mnist_images_normalized = mnist_images / mnist_images.sum(axis=1, keepdims=True)

        # store
        self.mnist_images_normalized = mnist_images_normalized
        self.mnist_labels = mnist_labels

    def get_categories(self) -> int:
        return 10

    def _sample_points(self, image: np.ndarray, n: int = 50):
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

    def __call__(self, n: int):
        images, y = self.mnist_images_normalized, self.mnist_labels

        # Sample n images randomly
        idxs = np.random.choice(len(images), size=n, replace=False)
        images = images[idxs]
        y = y[idxs]

        sampled_points_all_images = []

        for i, image in enumerate(images):
            sampled_points = self._sample_points(image, self._k)

            if self._flatten:
                sampled_points = sampled_points.flatten()

            sampled_points_all_images.append(sampled_points)

        X = torch.tensor(np.array(sampled_points_all_images))
        y = torch.tensor(y).int()

        return X, y


class Spiral(Dataset):
    """
    Generate a double spiral.
    """

    def __init__(self, noise: float = 0.1, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self._noise = noise

    def get_categories(self) -> int:
        return 2

    def __call__(self, n: int):
        t = (torch.linspace(0, 1, n // 2 + n % 2)) ** (1 / 2) * np.pi * 3
        r = (torch.linspace(0, 1, n // 2 + n % 2)) ** (1 / 2) * 5  # Linearly increasing radius

        # Spiral 1
        x1 = (r * torch.cos(t))
        y1 = (r * torch.sin(t))

        # Spiral 2, rotating by 180 degrees
        x2 = r * torch.cos(t - np.pi)
        y2 = r * torch.sin(t - np.pi)

        # Add Gaussian noise if specified
        x1 += torch.randn_like(t) * self._noise
        y1 += torch.randn_like(t) * self._noise
        x2 += torch.randn_like(t) * self._noise
        y2 += torch.randn_like(t) * self._noise

        data1 = torch.stack((x1, y1), dim=1)
        data2 = torch.stack((x2, y2), dim=1)
        double_spiral_data = torch.cat((data1, data2), dim=0)

        X = double_spiral_data[:n]
        y = torch.tensor([0] * len(data1) + [1] * len(data2))[:n].int()

        return X, y


class Moons(Dataset):
    """
    Two moons dataset.

     ___
    |   |
      |___|
    """

    def __init__(self, noise: float = 0.1, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self._noise = noise

    def get_categories(self) -> int:
        return 2

    def __call__(self, n: int):
        X, y = make_moons(n_samples=n, noise=self._noise, random_state=42)
        return torch.tensor(X), torch.tensor(y).int()


class SplitLine(Dataset):
    """
    A split line dataset.

      /
     o

     o
    /
    """

    def __init__(self, noise: float = 0.005, offset: float = 1, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self._noise = noise
        self._offset = offset

    def get_categories(self) -> int:
        return 2

    def __call__(self, n: int):
        mu = np.array([0, 0])
        sigma = np.array([[1, 1], [self._noise, -self._noise]])

        data = np.random.multivariate_normal(mu, sigma, n)

        X = torch.tensor(data)
        y = torch.zeros(n)

        x_values = X[:, 0]
        y_values = X[:, 1]

        y_values[x_values >= 0] += self._offset
        y_values[x_values < 0] -= self._offset

        y[x_values >= 0] = 1
        X[:, 1] = y_values

        return X, y.int()


class Gradient(Dataset):
    """
    Gradient.

    ###
    mmm
    ...
    """

    def __init__(self, cool: bool = False, **kwargs):
        """
        :param cool: Whether the gradient is cool or not.
        """
        super().__init__(**kwargs)

        self._cool = cool

    def get_categories(self) -> int:
        return 2

    def __call__(self, n: int):
        exponents = [1, 1 / 2]
        if self._cool:
            exponents = [1 / 2.4, 1 / 1.4]

        x_axis = torch.rand(n) ** exponents[0]
        y_axis = torch.rand(n) ** exponents[1]

        X = torch.stack((x_axis, y_axis), dim=1)

        y = torch.zeros(n)
        y[X[:, 1] >= .5] = 1

        return X, y.int()


class SIR_Basic(Dataset):
    def __init__(self, shuffle=True, name=None, **kwargs):
        """kwargs are passed to the simulation."""
        super().__init__(shuffle, name)
        self.kwargs = kwargs
    
    def get_categories(self) -> int:
        """Not every kind of label is a one-hot encoding, Tom."""
        raise TypeError("SIR dataset labels don't have categories!")
    
    def _sample_parameters(self, n_samples: int) -> torch.Tensor:
        '''
        Generate sensible parameter values for the simulation. Uses Gaussian distribution for all three parameters.

        ### Input:
        * n_samples: Number of samples to be generated

        ### Output:
        Tensor with shape (n_sample) x 3
        '''
        # Priors of each parameter
        lam = torch.normal(0.5, 0.1, size=(n_samples,))
        mu  = torch.normal(0.25, 0.1, size=(n_samples,))
        I_0 = torch.normal(0.06, 0.05, size=(n_samples,))

        # I_0 must not be larger than 1
        I_0[I_0 > 1] = 1

        Y = torch.stack((lam, mu, I_0)).T.to(survae.DEVICE)

        # make sure all parameters are non-negative
        Y = torch.abs(Y)

        return Y
    
    def _simulate(self, Y: torch.Tensor, steps_per_day: int = 10, n_days: int = 100, flatten: bool = True):
        '''
        Simulates the most basic SIR model, given parameters lambda, mu, and I_0.

        ### Inputs:
        * Y: Initial values with shape (N, 3) containing lambda, mu, and I_0 (in this order) for N simulations.
        * steps_per_day: Number of simulation steps per unit of time.
        * n_days: Number of units of time to simulate.
        * flatten: Whether to flatten the output for each simulation.

        ### Output:
        Tensor with shape (N, n_days, 2) containing dS and dR (in this order) for each simulation and unit of time.
        If 'flatten' is True, the tensor will instead have shape (N, n_days * 2)
        '''
        # extract initial values
        lam = Y[:, 0]
        mu = Y[:, 1]
        I_0 = Y[:, 2]

        h = 1 / steps_per_day      # step size
        n = len(Y)                 # number of samples, i.e. simulations
        t = steps_per_day * n_days # number of simulation steps

        C = torch.zeros((n, t+1, 2)) # contains S & R for each simulation and timestep
        C[:, 0, 0] = 1 - I_0

        for i in range(t):
            # define S, I, R
            S = C[:, i, 0]
            R = C[:, i, 1]
            I = 1 - S - R

            # compute derivatives
            dS = -lam * S * I
            dR = mu * I

            # compute one step
            C[:, i+1, 0] = S + h * dS
            C[:, i+1, 1] = R + h * dR
        
        # compute the desired output
        X = C[:, 1::steps_per_day] - C[:, :-1:steps_per_day]
        X[:, :, 0] *= -1 # convention necessitates that dS change sign

        if flatten:
            X = X.flatten(start_dim=1)

        return X
    
    def __call__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        Y = self._sample_parameters(n)
        X = self._simulate(Y, **self.kwargs)

        return Y, X # the parameters are the data and the simulation is the condition!


class SIR_Degenerate(Dataset):
    def __init__(self, shuffle=True, name=None, **kwargs):
        """kwargs are passed to the simulation."""
        super().__init__(shuffle, name)

        self.kwargs = kwargs
    
    def get_categories(self) -> int:
        """Not every kind of label is a one-hot encoding, Tom."""
        raise TypeError("Degenerate SIR dataset labels don't have categories!")
    
    def _sample_parameters(self, n_samples:int) -> torch.Tensor:
        '''
        Generate sensible parameter values for the simulation. Uses Gaussian distribution for all parameters.
        In contrast to the non-degenerate case, this function generates two values for lambda.

        ### Input:
        - n_samples: Number of samples to be generated

        ### Output:
        Tensor with shape (n_sample) x 4
        '''
        # Priors of each parameter
        # The values for lam1 and lam2 are chosen such that the distribution of lam1 * lam2
        # closely resembles that of lam from SIR_Basic
        lam1 = torch.normal(8 / 10, 0.06, size=(n_samples,))
        lam2 = torch.normal(5 /  8, 0.11, size=(n_samples,))
        mu  = torch.normal(0.25, 0.1, size=(n_samples,))
        I_0 = torch.normal(0.06, 0.05, size=(n_samples,))

        # I_0 must not be larger than 1
        I_0[I_0 > 1] = 1

        Y = torch.stack((lam1, lam2, mu, I_0)).T.to(survae.DEVICE)

        # make sure all parameters are non-negative
        Y = torch.abs(Y)

        return Y
    
    def _simulate(self, Y: torch.Tensor, steps_per_day: int = 10, n_days: int = 100, flatten: bool = True):
        '''
        Simulates the most basic SIR model, given parameters lambda1, lambda2, mu, and I_0.

        ### Inputs:
        * Y: Initial values with shape (N, 4) containing lambda1, lambda2, mu, and I_0 (in this order) for N simulations.
        * steps_per_day: Number of simulation steps per unit of time.
        * n_days: Number of units of time to simulate.
        * flatten: Whether to flatten the output for each simulation.

        ### Output:
        Tensor with shape (N, n_days, 2) containing dS and dR (in this order) for each simulation and unit of time.
        If 'flatten' is True, the tensor will instead have shape (N, n_days * 2)
        '''
        # extract initial values
        lam1 = Y[:, 0]
        lam2 = Y[:, 1]
        lam = lam1 * lam2
        mu = Y[:, 2]
        I_0 = Y[:, 3]

        h = 1 / steps_per_day      # step size
        n = len(Y)                 # number of samples, i.e. simulations
        t = steps_per_day * n_days # number of simulation steps

        C = torch.zeros((n, t+1, 2)) # contains S & R for each simulation and timestep
        C[:, 0, 0] = 1 - I_0

        for i in range(t):
            # define S, I, R
            S = C[:, i, 0]
            R = C[:, i, 1]
            I = 1 - S - R

            # compute derivatives
            dS = -lam * S * I
            dR = mu * I

            # compute one step
            C[:, i+1, 0] = S + h * dS
            C[:, i+1, 1] = R + h * dR
        
        # compute the desired output
        X = C[:, 1::steps_per_day] - C[:, :-1:steps_per_day]
        X[:, :, 0] *= -1 # convention necessitates that dS change sign

        if flatten:
            X = X.flatten(start_dim=1)

        return X
    
    def __call__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        Y = self._sample_parameters(n)
        X = self._simulate(Y, **self.kwargs)

        return Y, X # the parameters are the data and the simulation is the condition!


# (Jannis) I had to change these lists to only be created when they're actually needed because for some reason
# my PC can't load MNIST via _fetch_openml, and since this file is loaded in survae.__init__ I couldn't do
# anything without it crashing :(
TWO_D_DATASETS = lambda: [Ngon(), Corners(), Circles(), Checkerboard(), Spiral(), Moons(), SplitLine(), Gradient()]
MULTI_D_DATASETS = lambda: [SpatialMNIST()]
ALL_DATASETS = lambda: TWO_D_DATASETS() + MULTI_D_DATASETS()
