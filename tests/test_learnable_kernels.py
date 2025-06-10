import torch

from kta import (
    KernelCombiner,
    LearnablePolynomial,
    LearnableRBF,
    LearnableSigmoid,
    kta_torch,
)


def test_learnable_rbf_alignment():
    torch.manual_seed(0)
    X = torch.randn(15, 4)
    y = torch.randint(0, 2, (15,)).float() * 2 - 1
    model = LearnableRBF(gamma_init=0.5)
    K = model(X)
    a = kta_torch(K, y)
    assert K.shape == (15, 15)
    assert torch.allclose(K, K.T, atol=1e-5)
    assert isinstance(a, torch.Tensor)


def test_learnable_rbf_cross_gram():
    X1 = torch.randn(10, 4)
    X2 = torch.randn(15, 4)
    model = LearnableRBF(gamma_init=0.5)
    K = model(X1, X2)
    assert K.shape == (10, 15)


def test_learnable_polynomial_alignment():
    X = torch.randn(10, 3)
    y = torch.randint(0, 2, (10,)).float() * 2 - 1
    model = LearnablePolynomial(degree=2, c_init=1.0)
    K = model(X)
    assert K.shape == (10, 10)
    assert torch.allclose(K, K.T, atol=1e-5)
    assert isinstance(kta_torch(K, y), torch.Tensor)


def test_learnable_poly_cross_gram():
    X1 = torch.randn(6, 3)
    X2 = torch.randn(4, 3)
    model = LearnablePolynomial(degree=2, c_init=0.0)
    K = model(X1, X2)
    assert K.shape == (6, 4)


def test_learnable_sigmoid_alignment():
    X = torch.randn(12, 5)
    y = torch.randint(0, 2, (12,)).float() * 2 - 1
    model = LearnableSigmoid(gamma_init=0.1, c_init=1.0)
    K = model(X)
    assert K.shape == (12, 12)
    assert torch.allclose(K, K.T, atol=1e-5)
    assert isinstance(kta_torch(K, y), torch.Tensor)


def test_learnable_sigmoid_cross_gram():
    X1 = torch.randn(12, 6)
    X2 = torch.randn(8, 6)
    model = LearnableSigmoid(gamma_init=0.1, c_init=0.5)
    K = model(X1, X2)
    assert K.shape == (12, 8)


def test_kernel_combiner_alignment():
    X = torch.randn(20, 6)
    y = torch.randint(0, 2, (20,)).float() * 2 - 1
    model = KernelCombiner(
        [LearnableRBF(1.0), LearnablePolynomial(2, 1.0), LearnableSigmoid(0.05, 0.0)],
    )
    K = model(X)
    a = kta_torch(K, y)
    assert K.shape == (20, 20)
    assert torch.allclose(K, K.T, atol=1e-5)
    assert a.item() >= -1 and a.item() <= 1
