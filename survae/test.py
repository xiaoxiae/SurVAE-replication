import unittest

from survae import SurVAE
from survae.data import *
from survae.layer import *


class LayerTest(unittest.TestCase):
    def test_bijective(self):
        b = BijectiveLayer(23, [25, 50])
        for _ in range(100):
            a = torch.randn(300, 23) * 3
            self.assertTrue(torch.allclose(a, b.backward(b.forward(a)), atol=1e-5))
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_bijective_conditional(self):
        b = BijectiveLayer(23, [25, 50])
        b.make_conditional(5)
        for _ in range(100):
            a = torch.randn(300, 23) * 3
            cond = torch.randn(300, 5) * 6
            self.assertTrue(torch.allclose(a, b.backward(b.forward(a, cond), cond), atol=1e-5))
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a, cond), cond), atol=1e-5))

    def test_abs(self):
        b = AbsoluteUnit(torch.zeros(23) + 0.01)
        for _ in range(100):
            a = torch.abs(torch.randn(300, 23) * 3)  # TODO: it's not nice to have to call abs here...
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_orthonormal(self):
        b = OrthonormalLayer(23)
        for _ in range(100):
            a = torch.randn(300, 23) * 3
            self.assertTrue(torch.allclose(a, b.backward(b.forward(a)), atol=1e-5))
            self.assertTrue(torch.allclose(a, b.forward(b.backward(a)), atol=1e-5))

    def test_max(self):
        L = MaxTheLayer(3)
        for _ in range(100):
            _X = torch.randn(10, 3) * 3
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

        for _ in range(100):
            _X = torch.randn(300, 23) * 3
            _Z = L.forward(_X)  # sorted

            self.assertTrue((_X != _Z).any())  # different
            self.assertTrue((torch.allclose(torch.sort(_X, dim=-1).values, _Z)))  # actually sorted

            X_hat = L.backward(_Z)  # desorted

            self.assertTrue((X_hat != _Z).any())  # actually desorted

            # set equality
            self.assertTrue(torch.allclose(torch.sort(X_hat, dim=-1).values, _Z))

    def test_dequantization(self):
        L = DequantizationLayer()
        for _ in range(100):
            _X = torch.randn(300, 23) * 3

            _X_rounded = torch.floor(_X)
            Z = L.forward(_X_rounded)  # not rounded
            _X_hat = L.backward(Z)  # rounded
            self.assertTrue(torch.allclose(_X_rounded, _X_hat))
            self.assertTrue(torch.allclose(_X_rounded, Z, atol=1))

    def test_permutation(self):
        L = PermutationLayer()
        for _ in range(100):
            _X = torch.randn(300, 23) * 3
            _Z = L.forward(_X)

            self.assertTrue((_X != _Z).any())  # actually permuted
            self.assertTrue(
                torch.allclose(torch.sort(_X, dim=-1).values, torch.sort(_Z, dim=-1).values))  # set equality

            _X_hat = L.backward(_Z)  # permute again

            self.assertTrue((_X_hat != _Z).any())  # actually permuted
            self.assertTrue(
                torch.allclose(torch.sort(_X_hat, dim=-1).values, torch.sort(_Z, dim=-1).values))  # set equality
            self.assertTrue(torch.allclose(torch.sort(_X, dim=-1).values, torch.sort(_X_hat, dim=-1).values))
    
    def test_maxpooling(self):
        L = MaxPoolingLayer(18*18, 3)
        for _ in range(100):
            Z = torch.randn(6*6) * 3

            X = L.backward(Z)
            Z_hat = L.forward(X)
            self.assertTrue(torch.allclose(Z, Z_hat, atol=1e-5))

            # test that replacing all elements with their appropriate maximum gives the same output
            X_tilde = Z.view(6, 6).repeat_interleave(3, dim=0).repeat_interleave(3, dim=1).flatten()
            Z_tilde = L.forward(X_tilde)
            self.assertTrue(torch.allclose(Z, Z_tilde, atol=1e-5))


class DatasetTest(unittest.TestCase):
    def test_dataset(self):
        # all datasets
        for dataset in ALL_DATASETS:
            for f in [lambda d: d, lambda d: d.skew()]:
                f(dataset)

                for n in [2, 31, 500]:
                    data, labels = dataset.sample(n, labels=True)

                    self.assertTrue(labels.dtype == torch.int, dataset.get_name())
                    self.assertTrue(len(data) == len(labels), dataset.get_name())

                    # should also work without labels
                    data2 = dataset.sample(n, labels=False)

                    self.assertTrue(data.shape == data2.shape, dataset.get_name())

                    # if the labels are not sequential then we are sad
                    self.assertTrue(torch.max(labels) < dataset.get_categories())


class SurvaeTest(unittest.TestCase):
    def test_survae(self):
        b = SurVAE(
            [
                BijectiveLayer(23, [5, 5]),
                OrthonormalLayer(23),
            ] * 3,
        )

        for _ in range(100):
            a = torch.randn(300, 23) * 3
            assert torch.allclose(a, b.backward(b.forward(a)), atol=1e-5)

    def test_dimensions(self):
        a = SurVAE(
            [
                [BijectiveLayer(2, [64] * 5), OrthonormalLayer(2)]
                for _ in range(10)
            ],
            name="NF",
        )

        self.assertEqual(a.in_size(), 2)
        self.assertEqual(a.out_size(), 2)

        b = SurVAE(
            [
                Augment(2, 4),
                BijectiveLayer(4, [64] * 5), OrthonormalLayer(4),
                BijectiveLayer(4, [64] * 5), OrthonormalLayer(4),
                BijectiveLayer(4, [64] * 5), OrthonormalLayer(4),
                BijectiveLayer(4, [64] * 5),
            ],
            name="NF-augmented",
        )

        self.assertEqual(b.in_size(), 2)
        self.assertEqual(b.out_size(), 4)

    def test_survae_conditional(self):
        b = SurVAE(
            [
                BijectiveLayer(23, [5, 5]),
                OrthonormalLayer(23),
            ] * 3,
            condition_size=5,
        )

        for _ in range(100):
            a = torch.randn(300, 23) * 3
            cond = torch.randn(300, 5) * 6
            assert torch.allclose(a, b.backward(b.forward(a, cond), cond), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
