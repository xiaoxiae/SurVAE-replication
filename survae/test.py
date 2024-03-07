import unittest

from survae import SurVAE
from survae.data import *
from survae.layer import *


class LayerTest(unittest.TestCase):
    def test_bijective(self):
        b = BijectiveLayer(23, [25, 50])
        for _ in range(1000):
            a = torch.randn(300, 23) * 3
            self.assertTrue(torch.allclose(a, b.backward(b.forward(a)), atol=1e-5))
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_abs(self):
        b = AbsoluteUnit(torch.zeros(23) + 0.01)
        for _ in range(1000):
            a = torch.abs(torch.randn(300, 23) * 3)  # TODO: it's not nice to have to call abs here...
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_orthonormal(self):
        b = OrthonormalLayer(23)
        for _ in range(1000):
            a = torch.randn(300, 23) * 3
            self.assertTrue(torch.allclose(a, b.backward(b.forward(a)), atol=1e-5))
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_max(self):
        L = MaxTheLayer(3)
        for _ in range(1):
            _X = torch.rand(10, 3)
            vals, indices = torch.max(_X, dim=-1)

            _Z = L.forward(_X)

            for i in range(len(_X)):
                self.assertEqual(_Z[i], vals[i])

            X_hat = L.backward(_Z)
            vals_hat, indices_hat = torch.max(X_hat, dim=-1)
            for i in range(len(_X)):
                self.assertEqual(vals_hat[i], vals[i])

    def test_sorting(self):
        L = SortingLayer()

        for _ in range(10):
            _X = torch.rand(300, 23)
            _Z = L.forward(_X)  # sorted

            self.assertTrue((_X != _Z).any())  # different
            self.assertTrue((torch.allclose(torch.sort(_X, dim=-1).values, _Z)))  # actually sorted

            X_hat = L.backward(_Z)  # desorted

            self.assertTrue((X_hat != _Z).any())  # actually desorted

            # set equality
            self.assertTrue(torch.allclose(torch.sort(X_hat, dim=-1).values, _Z))

    def test_dequantization(self):
        L = DequantizationLayer()
        for _ in range(10):
            _X = torch.rand(300, 23)

            _X_rounded = torch.floor(_X)
            Z = L.forward(_X_rounded)  # not rounded
            _X_hat = L.backward(Z)  # rounded
            self.assertTrue(torch.allclose(_X_rounded, _X_hat))
            self.assertTrue(torch.allclose(_X_rounded, Z, atol=1))

    def test_permutation(self):
        L = PermutationLayer()
        for _ in range(10):
            _X = torch.rand(300, 23)
            _Z = L.forward(_X)

            self.assertTrue((_X != _Z).any())  # actually permuted
            self.assertTrue(
                torch.allclose(torch.sort(_X, dim=-1).values, torch.sort(_Z, dim=-1).values))  # set equality

            _X_hat = L.backward(_Z)  # permute again

            self.assertTrue((_X_hat != _Z).any())  # actually permuted
            self.assertTrue(
                torch.allclose(torch.sort(_X_hat, dim=-1).values, torch.sort(_Z, dim=-1).values))  # set equality
            self.assertTrue(torch.allclose(torch.sort(_X, dim=-1).values, torch.sort(_X_hat, dim=-1).values))


class DatasetTest(unittest.TestCase):
    def test_functions_no_labels(self):
        # 2D datasets
        for function in [ngon, corners, circles, checkerboard, spiral, moons, split_line]:
            for n in [2, 31, 500]:
                # Default is no label
                data = function(n)
                self.assertTrue(data.shape == (n, 2), function.__name__)

        # Spatial MNIST
        for function in [spatial_mnist]:
            for n in [2, 31, 500]:
                data = function(n)
                self.assertTrue(data.shape == (n, 100), function.__name__)

    def test_functions_labels(self):
        # all datasets
        for function in [ngon, corners, circles, checkerboard, spiral, moons, split_line, spatial_mnist]:
            for n in [2, 31, 500]:
                data, labels = function(n, labels=True)

                self.assertTrue(labels.dtype == torch.int, function.__name__)
                self.assertTrue(len(data) == len(labels), function.__name__)

    def test_datasets_no_labels(self):
        # 2D datasets
        for function in [ngon, corners, circles, checkerboard, spiral, moons, split_line]:
            dataset = Dataset(function)

            for n in [2, 31, 500]:
                # Default is no label
                data = dataset(n)
                self.assertTrue(data.shape == (n, 2), function.__name__)

        # Spatial MNIST
        for function in [spatial_mnist]:
            dataset = Dataset(function)

            for n in [2, 31, 500]:
                data = dataset(n)
                self.assertTrue(data.shape == (n, 100), function.__name__)

            # check that it also works when applying modifiers
            dataset.skew()

            for n in [2, 31, 500]:
                data = dataset(n)
                self.assertTrue(data.shape == (n, 100), function.__name__)

    def test_dataset_labels(self):
        # all datasets
        for function in [ngon, corners, circles, checkerboard, spiral, moons, split_line, spatial_mnist]:
            dataset = Dataset(function, labels=True)

            for n in [2, 31, 500]:
                data, labels = dataset(n)

                self.assertTrue(labels.dtype == torch.int, function.__name__)
                self.assertTrue(len(data) == len(labels), function.__name__)

            # check that it also works when applying modifiers
            dataset.skew()

            for n in [2, 31, 500]:
                data, labels = dataset(n)

                self.assertTrue(labels.dtype == torch.int, function.__name__)
                self.assertTrue(len(data) == len(labels), function.__name__)


class SurvaeTest(unittest.TestCase):
    def test_survae(self):
        b = SurVAE(
            [
                BijectiveLayer(23, [5, 5]),
                OrthonormalLayer(23),
            ] * 3,
        )

        for _ in range(1000):
            a = torch.rand(300, 23)
            assert torch.allclose(a, b.backward(b.forward(a)), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
