import numpy as np
import torch

from kta import (
    linear,
    linear_torch,
    polynomial,
    polynomial_torch,
    rbf,
    rbf_torch,
    sigmoid,
    sigmoid_torch,
)


def test_rbf_kernel_consistency():
    X = np.random.randn(25, 4)
    gamma = 0.7

    K_np = rbf(X, gamma=gamma)
    K_torch = rbf_torch(torch.tensor(X, dtype=torch.float32), gamma=gamma)

    assert np.allclose(K_np, K_torch.numpy(), rtol=1e-5, atol=1e-7)


def test_polynomial_kernel_consistency():
    X = np.random.randn(15, 3)
    degree = 3
    c = 2.0

    K_np = polynomial(X, degree=degree, c=c)
    K_torch = polynomial_torch(torch.tensor(X, dtype=torch.float32), degree=degree, c=c)

    assert np.allclose(K_np, K_torch.numpy(), rtol=1e-5, atol=1e-7)


def test_sigmoid_kernel_consistency():
    X = np.random.randn(10, 5)
    gamma = 0.1
    c = 0.0

    K_np = sigmoid(X, gamma=gamma, c=c)
    K_torch = sigmoid_torch(torch.tensor(X, dtype=torch.float32), gamma=gamma, c=c)

    assert np.allclose(K_np, K_torch.numpy(), rtol=1e-5, atol=1e-7)


def test_linear_kernel_consistency():
    X = np.random.randn(12, 6)

    K_np = linear(X)
    K_torch = linear_torch(torch.tensor(X, dtype=torch.float32))

    assert np.allclose(K_np, K_torch.numpy(), rtol=1e-5, atol=1e-7)
