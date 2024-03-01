from bisect import bisect_left

import numpy as np
import torch
from matplotlib import pyplot as plt

from survae import SurVAE


def calc_cs(model: SurVAE, X: torch.Tensor, n_samples: int):
    '''
    Calculate the c_i values with respect to each index of X, which are needed for calibration.

    ### Input:
    - model:     SurVAE model to be checked.
    - X:         Ground truth.
    - n_samples: Number of samples with respect to which each entry in the ground truth X is sorted.

    ### Output:
    A list containing a list of c_i values for each index of X.
    '''
    n_indices = X.shape[1]
    cs = [[] for _ in range(n_indices)]

    for _X in X:
        # sample comparison values
        X_comparison = model.sample(n_samples)

        for i in range(n_indices):
            # sort comparison values with respect to current index
            X_sorted = sorted(X_comparison, key=lambda x: x[i])

            # find index k of ground truth
            k = bisect_left(X_sorted, _X[i], key=lambda x: x[i])

            # record c value
            cs[i].append(k / n_samples)

    return torch.tensor(cs).cpu()


def plot_histogram(cs: torch.Tensor, n_bins: int, title: str):
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

def plot_cdf(cs:torch.Tensor):
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
