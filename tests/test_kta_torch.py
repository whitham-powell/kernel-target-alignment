import torch

from kta import alignment_torch, kta_torch, rbf_torch


# Torch versions of your kernels and kta
def test_torch_random_vs_perfect():
    torch.manual_seed(0)
    X = torch.randn(30, 4)
    y = torch.randint(0, 2, (30,)) * 2 - 1  # Convert {0,1} to {-1,1}

    Kp = torch.outer(y, y).float()  # perfect alignment
    Kr = rbf_torch(X, gamma=0.5)  # realistic kernel, unlikely to align perfectly

    assert kta_torch(Kp, y) > kta_torch(Kr, y)


def test_torch_identity_kernel():
    K = torch.eye(5)
    y = torch.tensor([1, -1, 1, -1, 1], dtype=torch.float32)
    expected = 1 / torch.sqrt(torch.tensor(float(len(y))))
    result = kta_torch(K, y)
    assert torch.isclose(result, expected)


def test_torch_self_alignment_is_one():
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    K = rbf_torch(X, gamma=0.7)
    assert torch.isclose(alignment_torch(K, K), torch.tensor(1.0))


def test_torch_kta_perfect_alignment():
    torch.manual_seed(1)
    y = torch.randint(0, 2, (25,)) * 2 - 1
    K = torch.outer(y, y).float()
    assert torch.isclose(kta_torch(K, y), torch.tensor(1.0))


def test_torch_identity_normalized():
    K = torch.randn(12, 12)
    K = (K + K.T) / 2  # ensure symmetry
    K = K / torch.norm(K)  # normalize
    assert torch.isclose(alignment_torch(K, K), torch.tensor(1.0), atol=1e-6)


def test_alignment_vs_orthogonal():
    A = torch.eye(10)
    B = torch.flip(A, dims=[0])
    assert alignment_torch(A, B) < 0.2  # nearly orthogonal
