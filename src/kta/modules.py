# src/kta/modules.py
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.types import Tensor

from kta import polynomial_torch, rbf_torch, sigmoid_torch


class LearnableRBF(nn.Module):
    def __init__(self, gamma_init: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, X1: Tensor, X2: Optional[Tensor] = None) -> Tensor:
        return rbf_torch(X1, X2, gamma=self.gamma)


class LearnablePolynomial(nn.Module):
    def __init__(self, degree: int = 3, c_init: float = 1.0):
        super().__init__()
        self.degree = degree
        self.c = nn.Parameter(torch.tensor(c_init))

    def forward(self, X1: Tensor, X2: Optional[Tensor] = None) -> Tensor:
        return polynomial_torch(X1, X2, degree=self.degree, c=self.c)


class LearnableSigmoid(nn.Module):
    def __init__(self, gamma_init: float = 0.01, c_init: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.c = nn.Parameter(torch.tensor(c_init))

    def forward(self, X1: Tensor, X2: Optional[Tensor] = None) -> Tensor:
        return sigmoid_torch(X1, X2, gamma=self.gamma, c=self.c)


class FixedKernel(nn.Module):
    def __init__(self, kernel_fn: Callable):
        super().__init__()
        self.kernel_fn = kernel_fn

    def forward(self, X1: Tensor, X2: Optional[Tensor] = None) -> Tensor:
        return self.kernel_fn(X1, X2)


class KernelCombiner(nn.Module):
    def __init__(self, kernel_modules: list[nn.Module]):
        super().__init__()
        self.kernels = nn.ModuleList(kernel_modules)
        self.raw_weights = nn.Parameter(torch.ones(len(kernel_modules)))

    def forward(self, X1: Tensor, X2: Optional[Tensor] = None) -> Tensor:
        Ks = [k(X1, X2) for k in self.kernels]
        weights = torch.softmax(self.raw_weights, dim=0)
        out = torch.zeros_like(Ks[0])
        for w, K in zip(weights, Ks):
            out += w * K
        return out
