# src/kta/kernels.py

from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


def rbf(X: NDArray, Y: Optional[NDArray] = None, gamma: float = 1.0) -> NDArray:
    if Y is None:
        Y = X
    D2 = cdist(X, Y, "sqeuclidean")
    return np.exp(-gamma * D2)


def polynomial(
    X: NDArray,
    Y: Optional[NDArray] = None,
    degree: int = 2,
    c: float = 1.0,
) -> NDArray:

    if Y is None:
        Y = X
    return (X @ Y.T + c) ** degree


def linear(X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
    if Y is None:
        Y = X
    return X @ Y.T


def sigmoid(
    X: NDArray,
    Y: Optional[NDArray] = None,
    gamma: float = 0.01,
    c: float = 1.0,
) -> NDArray:
    if Y is None:
        Y = X
    return np.tanh(gamma * (X @ Y.T) + c)


def rbf_torch(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    if Y is None:
        Y = X
    D2 = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-gamma * D2)


def polynomial_torch(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    degree: int = 2,
    c: float = 1.0,
) -> torch.Tensor:
    if Y is None:
        Y = X
    return (X @ Y.T + c) ** degree


def linear_torch(X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
    if Y is None:
        Y = X
    return X @ Y.T


def sigmoid_torch(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    gamma: float = 0.01,
    c: float = 1.0,
) -> torch.Tensor:
    if Y is None:
        Y = X
    return torch.tanh(gamma * (X @ Y.T) + c)
