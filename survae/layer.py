from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from scipy.stats import ortho_group
from torch import exp, tanh, log
import numpy as np


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

    @abstractmethod
    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        """
        Computes the forward pass of the layer and, optionally, the log likelihood contribution as a scalar (i.e. already summed).
        """
        pass

    @abstractmethod
    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
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

    def make_conditional(self, size: int):
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

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = X @ self.o

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        return Z @ self.o.T

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return self.size


class BijectiveLayer(Layer):
    def __init__(self, shape: tuple[int] | int, hidden_sizes: list[int]) -> None:
        '''
        Standard bijective block from normalizing flow architecture.

        ### Inputs:
        * shape: Shape of input entries, which is the same for the output.
        * hidden_sizes: Sizes of hidden layers of the nested FFNN.
        '''
        super().__init__()

        # transform shape variable into a more usable form
        if isinstance(shape, int):
            shape = (shape,)

        self.shape = shape

        self.size = torch.prod(torch.tensor(shape)).item()

        assert self.size > 1, "Bijective layer size must be at least 2!"

        # The size of the skip connection is half the input size, rounded down
        self.skip_size = self.size // 2
        self.non_skip_size = self.size - self.skip_size

        self.hidden_sizes = hidden_sizes

        self.ffnn = FFNN(self.skip_size, self.hidden_sizes, self.non_skip_size + 1)

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        # flatten input
        X = X.flatten(start_dim=1)

        # split input into skip and non-skip
        skip_connection = X[:, :self.skip_size]
        non_skip_connection = X[:, self.skip_size:]

        # add conditional input
        if condition is not None:
            ffnn_input = torch.cat((skip_connection, condition), dim=1)
        else:
            ffnn_input = skip_connection

        # compute coefficients for linear transformation
        coeffs = self.ffnn(ffnn_input)
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

        # reshape output
        Z = Z.reshape(-1, *self.shape)

        if return_log_likelihood:
            return Z, torch.sum(s_log)
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        # flatten input
        Z = Z.flatten(start_dim=1)

        # split input into skip and non-skip
        skip_connection = Z[:, :self.skip_size]
        non_skip_connections = Z[:, self.skip_size:]

        # add conditional input
        if condition is not None:
            ffnn_input = torch.cat((skip_connection, condition), dim=1)
        else:
            ffnn_input = skip_connection

        # compute coefficients for linear transformation
        coeffs = self.ffnn(ffnn_input)
        # split output into t and pre_s
        t = coeffs[:, :-1]
        pre_s = coeffs[:, -1]
        # compute s and fix dimension; shape is (N,) but should be (N, 1)
        s = exp(tanh(pre_s)).unsqueeze(1)

        # apply inverse transformation
        new_connection = (non_skip_connections - t) / s
        # stack skip connection and transformed non-skip connection
        X = torch.cat((skip_connection, new_connection), dim=1)

        # reshape output
        X = X.reshape(-1, *self.shape)

        return X

    def make_conditional(self, size: int):
        self.ffnn = FFNN(self.skip_size + size, self.hidden_sizes, self.non_skip_size + 1)

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

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = abs(X)
        if return_log_likelihood:
            pos_count = (X > 0).sum(dim=0)
            neg_count = X.shape[0] - pos_count

            ll = torch.sum(
                pos_count * torch.log(self.q)
                +
                neg_count * torch.log(1 - self.q)
            )

            return Z, ll  # TODO: do we have a problem if q is learned to be less than 0? also we should put an upper limit on it
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
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

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):

        max_val, max_index = torch.max(X, dim=-1)

        if return_log_likelihood:

            exp_distr = torch.distributions.exponential.Exponential(self.lam)

            ll_q = - torch.sum(log(torch.ones_like(X) * self.index_probs[max_index])) - torch.sum(
                exp_distr.log_prob(torch.cat((X[:max_index], X[max_index + 1:]))))

            return max_val, ll_q
        else:
            return max_val

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
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


class MaxPoolingLayer(Layer):
    '''
    MaxPoolingLayer: Layer that performs max pooling on the input data.

    size: input size, i.e. flattened picture
    width: input width, i.e. the width of the picture, sqrt of size

    out_width: width of the output picture, sqrt of out_size
    out_size: size of flattened output picture

    stride: the stride of the max pooling operation 

    Distribution choices: 
        - standard half-normal distribution (default)
        - exponential distribution 

    '''
    def __init__(self, size: int, stride: int, exponential_distribution: bool = False, learn_distribution_parameter: bool = False):
        super().__init__()

        self.size = size
        self.width = np.sqrt(size).astype(int) 

        assert self.width % stride == 0, "Stride must be a divisor of size!"
        self.stride = stride

        self.out_width = int(self.width / self.stride)

        
        if exponential_distribution:
            self.distribution = "exponential"
            lam = torch.tensor(0.1)
            if learn_distribution_parameter:
                lam = nn.Parameter(lam)
            self.lam = lam
        else:
            self.distribution = "half-normal"
            sigma = torch.tensor(1.0)
            if learn_distribution_parameter:
                sigma = nn.Parameter(sigma)
            self.sigma = sigma


        self.index_probs = torch.tensor([1 / self.stride**2 for _ in range(self.stride**2)])


    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):

        X = X.view(-1, self.width, self.width) # reshape to 2D
        
        l = []
        for i in range(self.stride):
            for j in range(self.stride):
                l.append(X[:, i::self.stride,j::self.stride])

        combined_tensor = torch.stack(l, dim=0)
        Z, _ = torch.max(combined_tensor, dim=0)
        Z = Z.flatten(start_dim=1)

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        Z = Z.view(-1, self.out_width, self.out_width)
        
        # expand matrix containing local maxima (by repeating local max)
        X_hat = Z.repeat_interleave(self.stride,dim=2).repeat_interleave(self.stride,dim=1)

        # mask for the indices of the local maxima
        k = torch.distributions.categorical.Categorical(self.index_probs) 
        indices = k.sample(Z.shape)

        indices_repeated = indices.repeat_interleave(self.stride, dim=2).repeat_interleave(self.stride, dim=1)
        index_places = torch.arange(self.stride**2).reshape(self.stride, self.stride).repeat(self.out_width, self.out_width)

        index_mask = (index_places == indices_repeated)

        # sample values in (- infty, 0]) with respective distribution
        # TODO (Jannis): to make the parameter sigma learnable, I had to redo the sampling so that the
        # influence of sigma is more obvious to autograd. Maybe I will be bothered to do something similar
        # for the exponential distribution, but probably not

        # if self.distribution == "half-normal":
        #     distr = torch.distributions.half_normal.HalfNormal(self.sigma)
        # else:
        #     distr = torch.distributions.exponential.Exponential(self.lam)
        if self.distribution != "half-normal":
            raise NotImplementedError("Currently no distribution other than half-normal is supported!")
        # samples = -distr.sample(X_hat.shape)
        samples = -torch.randn(X_hat.shape).abs() * self.sigma.abs()

        X_hat = X_hat + samples * ~index_mask
        
        return X_hat.flatten(start_dim=1)

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return int(self.out_width ** 2)


class MaxPoolingLayerWithHop(Layer):
    '''
    MaxPoolingLayerWithHop: Layer that performs max pooling on the input data.

    size: input size, i.e. flattened picture
    width: input width, i.e. the width of the picture, sqrt of size
    hop: defines the distance between the blocks of size stride for which the maxima is taken
        we require that hop >= 1

    out_width: width of the output picture, sqrt of out_size
    out_size: size of flattened output picture

    stride: the stride of the max pooling operation

    Distribution choices: 
        - standard half-normal distribution (default)
        - exponential distribution 

    '''
    def __init__(self, size: int, stride: int, hop: int, exponential_distribution: bool = False, learn_distribution_parameter: bool = False):
        super().__init__()

        self.size = size

        self.width = np.sqrt(size).astype(int) 

        assert stride <= self.width, "Stride must be smaller than the width of the picture!"
        assert 0 < hop and hop <= stride, "Hop must be smaller than the stride!"
        assert (self.width - stride) % hop == 0, "Stride and hop must be chosen such that the picture is fully covered!"

        self.stride = stride
        self.hop = hop

        self.out_width = (self.width - stride) // hop + 1 # = the number of blocks considered (possible overlap)
        
        if exponential_distribution:
            self.distribution = "exponential"
            lam = torch.tensor(0.1)
            if learn_distribution_parameter:
                lam = nn.Parameter(lam)
            self.lam = lam
        else:
            self.distribution = "half-normal"
            sigma = torch.tensor(1.0)
            if learn_distribution_parameter:
                sigma = nn.Parameter(sigma)
            self.sigma = sigma


    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):

        X = X.view(-1, self.width, self.width) # reshape to 2D
        
        l = []
        for i in range(self.stride):
            for j in range(self.stride):
                l.append(X[:, i:i+self.width-self.stride + 1:self.hop,j:j+self.width-self.stride + 1:self.hop])

        combined_tensor = torch.stack(l, dim=0)
        Z, _ = torch.max(combined_tensor, dim=0)
        Z = Z.flatten(start_dim=1)

        if return_log_likelihood:
            return Z, 0
        else:
            return Z
    
    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        Z = Z.view(-1, self.out_width, self.out_width)

        _max = (Z.max() + 1).item()
        max_max = torch.full((len(Z), self.out_width * self.out_width, self.width, self.width), _max)

        batch_indices = torch.arange(len(Z))
        for i in range(self.out_width):
            for j in range(self.out_width):
                max_max[batch_indices, j + i * (self.out_width), (self.hop * i):(self.hop * i + self.stride), (self.hop * j):(self.hop * j + self.stride)] = Z[batch_indices, i, j].unsqueeze(1).unsqueeze(2)

        # This is the best possible name. I will not elaborate.
        min_max = max_max.min(dim=1)[0]

        block_mask = torch.isclose(min_max.unsqueeze(1), max_max)
        _rand = torch.rand(block_mask.shape)
        _rand[~block_mask] = -1
        arg_max = _rand.flatten(start_dim=2).argmax(dim=2)

        noise_mask = torch.ones((len(Z), self.size), dtype=torch.bool)
        for idx in arg_max.T:
            noise_mask[batch_indices, idx] = False

        # sample values in (- infty, 0]) with respective distribution
        # TODO (Jannis): to make the parameter sigma learnable, I had to redo the sampling so that the
        # influence of sigma is more obvious to autograd. Maybe I will be bothered to do something similar
        # for the exponential distribution, but probably not

        # if self.distribution == "half-normal":
        #     distr = torch.distributions.half_normal.HalfNormal(self.sigma)
        # else:
        #     distr = torch.distributions.exponential.Exponential(self.lam)
        if self.distribution != "half-normal":
            raise NotImplementedError("Currently no distribution other than half-normal is supported!")
        # samples = -distr.sample(noise_mask.shape)
        samples = -torch.randn(noise_mask.shape).abs() * self.sigma

        X_hat = min_max.flatten(start_dim=1) + (samples * noise_mask)

        return X_hat

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return int(self.out_width ** 2)


class DequantizationLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = X + torch.rand_like(X)

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        return torch.floor(Z)

    def in_size(self) -> int | None:
        return None

    def out_size(self) -> int | None:
        return 


class SortingLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z, _ = torch.sort(X, dim=-1)
        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
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

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = torch.zeros_like(X)
        # randomly permute X
        for i in range(X.shape[0]):
            Z[i] = X[i, ..., torch.randperm(X.shape[-1])]

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        X = torch.zeros_like(Z)
        # randomly permute Z
        for i in range(Z.shape[0]):
            X[i] = Z[i, ..., torch.randperm(Z.shape[-1])]

        return X

    def in_size(self) -> int | None:
        return None

    def out_size(self) -> int | None:
        return None


class Augment(Layer):
    def __init__(self, size, augmented_size):
        super().__init__()

        self.size = size
        self.augmented_size = augmented_size

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = torch.empty((len(X), self.augmented_size))
        Z[:, :self.size] = X
        Z[:, self.size:] = torch.randn(len(X), self.augmented_size - self.size)

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        return Z[:, :self.size]

    def in_size(self) -> int | None:
        return self.size

    def out_size(self) -> int | None:
        return self.augmented_size


class SliceLayer(Layer):
    '''Opposite of Augment, i.e. goes from highdim to lowdim'''

    def __init__(self, original_size, new_size):
        super().__init__()

        assert original_size >= new_size, f"Invalid inputs to Slice layer: ({original_size}, {new_size})"

        self.new_size = new_size
        self.original_size = original_size

    def forward(self, X: torch.Tensor, condition: torch.Tensor | None = None, return_log_likelihood: bool = False):
        Z = X[..., :self.new_size]

        if return_log_likelihood:
            return Z, 0
        else:
            return Z

    def backward(self, Z: torch.Tensor, condition: torch.Tensor | None = None):
        X = torch.cat((Z, torch.randn((len(Z), self.original_size - self.new_size))), dim=1)
        return X

    def in_size(self) -> int | None:
        return self.original_size

    def out_size(self) -> int | None:
        return self.new_size
