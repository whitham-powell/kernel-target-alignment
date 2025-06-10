# src/kta/core.py

import numpy as np
import torch
from numpy.typing import NDArray
from torch.types import Tensor


def kta(K: NDArray, y: NDArray) -> float:
    """
    Compute the Kernel-Target Alignment (KTA) between a kernel matrix K and target vector y.
    The target vector y is expected to be a 1D array, and it will be reshaped if necessary.
    """
    y = y.ravel()
    Y = np.outer(y, y)
    return alignment(K, Y)


def alignment(K1: NDArray, K2: NDArray) -> float:
    """
    Compute the alignment between two kernels.
    """
    numerator = np.sum(K1 * K2)
    denominator = np.linalg.norm(K1) * np.linalg.norm(K2)
    if denominator == 0:
        return 0.0  # Avoid division by zero
    return float(numerator / denominator)


def alignment_torch(K1: Tensor, K2: Tensor) -> Tensor:
    """
    Compute the alignment between two kernels using PyTorch tensors.
    """
    numerator = torch.sum(K1 * K2)
    denominator = torch.norm(K1) * torch.norm(K2)
    return torch.where(denominator == 0, torch.tensor(0.0), numerator / denominator)


def kta_torch(K: Tensor, y: Tensor) -> Tensor:
    """
    Compute the Kernel-Target Alignment using PyTorch.
    """
    y = y.view(-1).float()  # Flatten in case it's (n,1)
    Y = torch.outer(y, y)
    return alignment_torch(K, Y)
