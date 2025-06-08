# src/kta.py

import numpy as np

from numpy.typing import NDArray


def kta(K: NDArray, y: NDArray) -> float:
    y = y.ravel()
    Y = np.outer(y,y)
    numerator =  np.sum(K * Y)
    denominator = np.linalg.norm(K) * np.linalg.norm(Y)
    return float(numerator / denominator)

