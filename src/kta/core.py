# src/kta/core.py

import numpy as np
from numpy.typing import NDArray


def kta(K: NDArray, y: NDArray) -> float:
    y = y.ravel()
    Y = np.outer(y, y)
    return alignment(K, Y)


def alignment(K1: NDArray, K2: NDArray) -> float:
    """
    Compute the alignment between two kernels.
    """
    numerator = np.sum(K1 * K2)
    denominator = np.linalg.norm(K1) * np.linalg.norm(K2)
    return float(numerator / denominator)
