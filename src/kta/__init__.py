# src/kta/__init__.py
"""
Kernel-Target Alignment toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convenience re-exports so users can just do

    from kta import kta, alignment, rbf
"""
from .core import alignment, kta
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

__all__ = [
    "alignment",
    "kta",
    "rbf",
    "polynomial",
    "linear",
    "sigmoid",
    "rbf_torch",
    "polynomial_torch",
    "linear_torch",
    "sigmoid_torch",
]
