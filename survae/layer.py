from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from scipy.stats import ortho_group
from torch import exp, tanh, log


class FFNN(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: list[int], out_size: int) -> None:
        '''
        Standard feed-forward neural network, used in 'BijectiveLayer'.

        ### Inputs:
        * in_size: Size of input.
        * hidden_sizes: List containing the sizes of the hidden layers.
        * out_size: Size of output.
        '''
        super().__init__()

        # append all layer sizes
        sizes = [in_size] + list(hidden_sizes) + [out_size]

        # initialize linear layers
        layers = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]

        # splice in ReLU layers
        for i in range(len(layers) - 1):
            layers.insert(i * 2 + 1, nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        return self.layers(X)


class Layer(nn.Module, ABC):
    """
    Abstract class used as the framework for all types of layers used in the SurVAE-Flow architecture.
    All layers are defined in the inference direction, i.e. the 'forward' method sends elements of the
    data space X to the latent space Z, whereas the 'backward' method goes from Z to X.
    """

    # TODO: later abc __init__

    @abstractmethod
    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        """
        Computes the forward pass of the layer and, optionally, the log likelihood contribution as a scalar (i.e. already summed).
        """
        pass

    @abstractmethod
    def backward(self, Z: torch.Tensor):
        """
        Computes the backward pass of the layer.
        """
        pass

    @abstractmethod
    def in_size(self) -> int | None:
        pass

    @abstractmethod
    def out_size(self) -> int | None:
        pass


class OrthonormalLayer(Layer):
    def __init__(self, size: int):
        '''
        Performs an orthonormal transformation. The transformation is randomly initialized and fixed,
        in particular it cannot be learned.

        ### Inputs:
        * size: Size of input, which is the same for the output.
        '''
        super().__init__()

        self.size = size
        self.o = torch.tensor(ortho_group.rvs(size))

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        if return_log_likelihood:
            return X @ self.o, 0
        else:
            return X @ self.o

    def backward(self, Z: torch.Tensor):
        return Z @ self.o.T

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return self.size


class BijectiveLayer(Layer):
    def __init__(self, size: int, hidden_sizes: list[int]) -> None:
        '''
        Standard bijective block from normalizing flow architecture.

        ### Inputs:
        * size: Size of input, which is the same for the output.
        * hidden_sizes: Sizes of hidden layers of the nested FFNN.
        '''
        super().__init__()

        assert size > 1, "Bijective layer size must be at least 2!"

        self.size = size

        # The size of the skip connection is half the input size, rounded down
        self.skip_size = size // 2
        self.non_skip_size = size - self.skip_size

        # The nested FFNN takes the skip connection as input and returns
        # the translation t (of same size as the non-skip connection) and
        # scaling factor s (which is a scalar) for the linear transformation
        self.ffnn = FFNN(self.skip_size, hidden_sizes, self.non_skip_size + 1)

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        # split input into skip and non-skip
        skip_connection = X[:, :self.skip_size]
        non_skip_connection = X[:, self.skip_size:]

        # compute coefficients for linear transformation
        coeffs = self.ffnn(skip_connection)
        # split output into t and pre_s
        t = coeffs[:, :-1]
        pre_s = coeffs[:, -1]
        # compute s_log for log-likelihood contribution and fix dimension; shape is (N,) but should be (N, 1)
        s_log = tanh(pre_s).unsqueeze(1)
        # compute s
        s = exp(s_log)

        # apply transformation
        new_connection = s * non_skip_connection + t
        # stack skip connection and transformed non-skip connection
        Z = torch.cat((skip_connection, new_connection), dim=1)

        if return_log_likelihood:
            return Z, torch.sum(s_log)
        else:
            return Z

    def backward(self, Z: torch.Tensor):
        # split input into skip and non-skip
        skip_connection = Z[:, :self.skip_size]
        non_skip_connections = Z[:, self.skip_size:]

        # compute coefficients for linear transformation
        coeffs = self.ffnn(skip_connection)
        # split output into t and pre_s
        t = coeffs[:, :-1]
        pre_s = coeffs[:, -1]
        # compute s and fix dimension; shape is (N,) but should be (N, 1)
        s = exp(tanh(pre_s)).unsqueeze(1)

        # apply inverse transformation
        new_connection = (non_skip_connections - t) / s
        # stack skip connection and transformed non-skip connection
        X = torch.cat((skip_connection, new_connection), dim=1)

        return X

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return self.size


class AbsoluteUnit(Layer):
    def __init__(self, q: torch.Tensor, learn_q: bool = False):
        '''
        Performs the absolute value inference surjection.

        ### Inputs:
        * q: Initial probabilities to choose each entry's positive representative over the negative. Must be of the same size as the input.
        * learn_q: Whether q should be learned alongside the rest of the SurVAE flow instead of fixed.
        '''
        super().__init__()

        self.size = q.shape[0]

        if learn_q:
            q = nn.Parameter(q)

        self.q = q

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        Z = abs(X)
        if return_log_likelihood:
            pos_count = (X > 0).sum(dim=0)
            neg_count = X.shape[0] - pos_count

            ll = torch.sum(
                pos_count * torch.log(self.q)
                +
                neg_count * torch.log(1 - self.q)
            )
            
            return Z, ll # TODO: do we have a problem if q is learned to be less than 0? also we should put an upper limit on it
        else:
            return Z

    def backward(self, Z: torch.Tensor):
        s = torch.sign(torch.rand_like(Z) - (1 - self.q))
        return Z * s

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return self.size


# TODO think about putting gamma instead of exponential distribution
class MaxTheLayer(Layer):
    def __init__(self, size: int, learn_index_probs: bool = False, learn_lambda: bool = False):
        super().__init__()

        self.size = size

        # if unspecified we use the categorical distribution with equal prob for all the categories
        index_probs = torch.tensor([1 / size for _ in range(size)])

        lam = torch.tensor([0.1])

        if learn_index_probs:
            index_probs = nn.Parameter(index_probs)

        if learn_lambda:
            lam = nn.Parameter(lam)

        self.index_probs = index_probs
        self.lam = lam

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):

        max_val, max_index = torch.max(X, dim=-1)

        if return_log_likelihood:

            exp_distr = torch.distributions.exponential.Exponential(self.lam)

            ll_q = - torch.sum(log(torch.ones_like(X) * self.index_probs[max_index])) - torch.sum(
                exp_distr.log_prob(torch.cat((X[:max_index], X[max_index + 1:]))))

            return max_val, ll_q
        else:
            return max_val

    def backward(self, Z: torch.Tensor):
        # sample smaller values for the indices from exponential distribution
        exp_distr = torch.distributions.exponential.Exponential(self.lam)

        X_shape = torch.cat((torch.tensor([len(Z)]), torch.tensor([self.size])))

        X = exp_distr.sample(X_shape).squeeze()
        X = Z.view(-1, 1).expand(tuple(X_shape)) - X

        # sample index for max_val
        k = torch.distributions.categorical.Categorical(self.index_probs)  # probability distribution
        indices = k.sample((X.shape[0],))
        X[torch.arange(X.shape[0]), indices] = Z.squeeze()

        return X

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return 1


class DequantizationLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        if return_log_likelihood:
            return X + torch.rand_like(X), 0
        else:
            return X + torch.rand_like(X)

    def backward(self, Z: torch.Tensor):
        return torch.floor(Z)

    def in_size(self) -> int | None:
        return None

    def out_size(self) -> int | None:
        return None


class SortingLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        Z, _ = torch.sort(X, dim=-1)
        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor):
        X = torch.zeros_like(Z)
        # randomly permute Z
        for i in range(Z.shape[0]):
            X[i] = Z[i][torch.randperm(Z.shape[1])]

        return X

    def in_size(self) -> int | None:
        return None

    def out_size(self) -> int | None:
        return None


class PermutationLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, return_log_likelihood: bool = False):
        Z = torch.zeros_like(X)
        # randomly permute X
        for i in range(X.shape[0]):
            Z[i] = X[i][torch.randperm(X.shape[1])]

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor):
        X = torch.zeros_like(Z)
        # randomly permute Z
        for i in range(Z.shape[0]):
            X[i] = Z[i][torch.randperm(Z.shape[1])]

        return X

    def in_size(self) -> int | None:
        return None

    def out_size(self) -> int | None:
        return None


