# src/kta/__init__.py
"""
Kernel-Target Alignment toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convenience re-exports so users can just do

    from kta import kta, alignment, rbf
"""
from .core import alignment, kta
from .kernels import polynomial, rbf  # add others as you implement

__all__ = ["alignment", "kta", "rbf", "polynomial"]
