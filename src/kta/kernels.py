# src/kernels.py

from typing import Optional

import numpy as np
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
