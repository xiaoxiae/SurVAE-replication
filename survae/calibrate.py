from bisect import bisect_left

import numpy as np
import torch
from scipy.stats import norm as scipy_norm
from matplotlib import pyplot as plt

from survae import SurVAE


def compute_calibration_values(model: SurVAE, X: torch.Tensor, Y: torch.Tensor | None = None, n_samples: int = 10):
    '''
    Calculate the c_i values with respect to each index of X, which are needed for calibration.

    ### Input:
    - model:     SurVAE model to be checked.
    - X:         Ground truth.
    - Y:         Condition ground truth. None by default.
    - n_samples: Number of samples with respect to which each entry in the ground truth X is sorted.

    ### Output:
    A list containing a list of c_i values for each index of X.
    '''
    n_indices = X.shape[1]
    cs = [[] for _ in range(n_indices)]

    has_condition = (Y is not None)

    # copy condition for number of samples
    if has_condition:
        Y = Y.unsqueeze(1).expand((-1, n_samples, -1))

    _iter = zip(X, Y) if has_condition else X

    for _X in _iter:
        # possibly unpack
        if has_condition:
            _X, _Y = _X

        # sample comparison values
        if has_condition:
            X_comparison = model.sample(n_samples, _Y)
        else:
            X_comparison = model.sample(n_samples)

        for i in range(n_indices):
            # sort comparison values with respect to current index
            X_sorted = sorted(X_comparison, key=lambda x: x[i])

            # find index k of ground truth
            k = bisect_left(X_sorted, _X[i], key=lambda x: x[i])

            # record c value
            cs[i].append(k / n_samples)

    return torch.tensor(cs).cpu()


def plot_calibration_history(cs: torch.Tensor, n_bins: int, title: str):
    bins = torch.zeros(n_bins).cpu()

    for c in cs:
        bin_index = min(int(c * n_bins), n_bins - 1)
        bins[bin_index] += 1

    barwidth = 1 / n_bins

    x = np.arange(n_bins) / n_bins + barwidth / 2
    y = bins / sum(bins)
    mean = 1 / n_bins

    plt.bar(x, y, width=0.9 * barwidth)
    plt.plot((0, 1), (mean, mean), ls='--', color='darkred')

    plt.xlim(0, 1)
    plt.title(title)
    plt.show()

def plot_calibration_cdf(cs:torch.Tensor):
    n = len(cs)

    # define CDF(t) and r(t) as in the lecture
    cdf = lambda t: 1/n * sum(cs <= t)
    r = lambda t: cdf(t) - t

    t = torch.linspace(0, 1, 200).cpu()
    y = [r(_t) for _t in t]

    stddev = torch.sqrt(t*(1-t))

    plt.plot(t, y)
    plt.plot(t, stddev, ls='--', color='darkred')
    plt.plot(t, -stddev, ls='--', color='darkred')
    plt.axis('equal')
    plt.show()


def plot_learned_distribution(
        Y: torch.Tensor,
        title: str,
        plotsize: float = 2,
        axis_scale: float = 3.6,
        bins: int = 40,
        alpha: float = 0.15,
):
    '''
    Check if a distribution looks gaussian in 1D and 2D.
    A circle is drawn in each scatter plot as a visual cue to more easily compare the spreads.
    The radius is chosen such that 90% of the samples of a gaussian distribution are expected
    to lie inside the circle.

    ### Inputs:
    * Y: Output of distribution.
    * plotsize: Size of each subplot.
    * axis_scale: "Zoom factor" of the plots.
    * bins: Number of bins in the 1D plots.
    '''
    n = Y.shape[-1] # number of parameters

    fig, ax = plt.subplots(n, n, figsize=(plotsize * n, plotsize * n))

    x_axis = np.linspace(-axis_scale, axis_scale, 100)
    y_gaussian = scipy_norm.pdf(x_axis)
    circle_radius = np.sqrt(-2 * np.log(0.1))

    for i in range(n):
        for k in range(n):
            _ax = ax[i][k]

            _ax.set_xticks([])
            _ax.set_yticks([])

            if i == k:
                _ax.hist(Y[:, i], bins=bins, density=True, range=(-axis_scale, axis_scale))
                _ax.plot(x_axis, y_gaussian, color='r')

                _ax.set_xlim(-axis_scale, axis_scale)
            else:
                # the y axis is flipped so that the graphs are mirrored on the diagonal
                # (e.g. subplot (1, 2) is a mirror image of subplot (2, 1))
                _ax.scatter(Y[:, i], -Y[:, k], s=1, alpha=alpha)
                _ax.plot(0, 0, 'ro', markersize=2)
                _ax.add_patch(plt.Circle((0, 0), radius=circle_radius, color='r', fill=False))

                _ax.set_xlim(-axis_scale, axis_scale)
                _ax.set_ylim(-axis_scale, axis_scale)
                _ax.set_aspect('equal')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
