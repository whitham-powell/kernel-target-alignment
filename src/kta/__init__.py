# src/kta/__init__.py
"""
Kernel-Target Alignment toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convenience re-exports so users can just do

    from kta import kta, alignment, rbf
"""
from .core import alignment, alignment_torch, kta, kta_torch
from .kernels import (
    linear,
    linear_torch,
    polynomial,
    polynomial_torch,
    rbf,
    rbf_torch,
    sigmoid,
    sigmoid_torch,
)
from .modules import (
    FixedKernel,
    KernelCombiner,
    LearnablePolynomial,
    LearnableRBF,
    LearnableSigmoid,
)

__all__ = [
    "alignment",
    "kta",
    "rbf",
    "polynomial",
    "linear",
    "sigmoid",
    "kta_torch",
    "alignment_torch",
    "rbf_torch",
    "polynomial_torch",
    "linear_torch",
    "sigmoid_torch",
    "FixedKernel",
    "KernelCombiner",
    "LearnablePolynomial",
    "LearnableRBF",
    "LearnableSigmoid",
]
