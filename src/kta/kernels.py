# src/kernels.py

import numpy as np
from scipy.spatial.distance import cdist


def rbf(X: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    D2 = cdist(X, X, "sqeuclidean")
    return np.exp(-gamma * D2)


def polynomial(X: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    return (X @ X.T + c) ** degree
