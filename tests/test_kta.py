import numpy as np

from kta import alignment, kta, rbf


def test_random_vs_perfect():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    y = rng.choice([-1, 1], size=30)

    Kp = np.outer(y, y)  # perfect alignment
    Kr = rbf(X, gamma=0.5)  # realistic kernel, unlikely to align perfectly

    assert kta(Kp, y) > kta(Kr, y)


def test_identity_kernel():
    K = np.eye(5)
    y = np.array([1, -1, 1, -1, 1])
    expected = 1 / np.sqrt(len(y))
    assert np.isclose(kta(K, y), expected)


def test_self_alignment_is_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    K = rbf(X, gamma=0.7)
    assert np.isclose(alignment(K, K), 1.0)


def test_kta_perfect_alignment():
    rng = np.random.default_rng(1)
    y = rng.choice([-1, 1], size=25)
    K = np.outer(y, y)  # same as label kernel
    assert np.isclose(kta(K, y), 1.0)
