import numpy as np
import torch

from kta import alignment, alignment_torch, kta, kta_torch, rbf, rbf_torch


def test_np_vs_torch_rbf_alignment():
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((30, 5))
    y_np = rng.choice([-1, 1], size=30)

    # NumPy
    K_np = rbf(X_np, gamma=0.3)
    kta_val_np = kta(K_np, y_np)
    align_val_np = alignment(K_np, np.outer(y_np, y_np))

    # Torch
    X_torch = torch.tensor(X_np, dtype=torch.float32)
    y_torch = torch.tensor(y_np, dtype=torch.float32)
    K_torch = rbf_torch(X_torch, gamma=0.3)
    kta_val_torch = kta_torch(K_torch, y_torch)
    align_val_torch = alignment_torch(K_torch, torch.outer(y_torch, y_torch))

    assert np.isclose(kta_val_np, kta_val_torch.item(), rtol=1e-5, atol=1e-7)
    assert np.isclose(align_val_np, align_val_torch.item(), rtol=1e-5, atol=1e-7)


def test_np_vs_torch_identity_alignment():
    K_np = np.eye(5)
    y_np = np.array([1, -1, 1, -1, 1])
    expected = 1 / np.sqrt(len(y_np))

    # Torch version
    K_torch = torch.eye(5)
    y_torch = torch.tensor(y_np, dtype=torch.float32)
    val_torch = kta_torch(K_torch, y_torch).item()

    assert np.isclose(kta(K_np, y_np), expected)
    assert np.isclose(val_torch, expected, rtol=1e-5, atol=1e-7)


def test_np_vs_torch_perfect_alignment():
    y_np = np.array([1, -1, 1, -1, 1, 1])
    K_np = np.outer(y_np, y_np)

    y_torch = torch.tensor(y_np, dtype=torch.float32)
    K_torch = torch.outer(y_torch, y_torch)

    assert np.isclose(kta(K_np, y_np), 1.0)
    assert np.isclose(kta_torch(K_torch, y_torch).item(), 1.0)
