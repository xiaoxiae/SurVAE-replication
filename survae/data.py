import numpy as np
import torch

# TODO: documentatio

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
