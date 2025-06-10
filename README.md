# Kernel Target Alignment (KTA)

A Python implementation of Kernel Target Alignment for kernel selection and optimization in machine learning.

## Overview

Kernel Target Alignment (KTA) is a metric that measures how well a kernel function aligns with target labels in machine learning tasks. This library provides both NumPy and PyTorch implementations, supporting gradient-based optimization of kernel parameters.

## Features

- **Dual Implementation**: NumPy for traditional ML workflows, PyTorch for gradient-based optimization
- **Multiple Kernel Functions**: RBF, Polynomial, Linear, and Sigmoid kernels
- **Learnable Parameters**: Optimize kernel parameters using gradient descent
- **Kernel Combination**: Combine multiple kernels with learnable weights
- **Comprehensive Testing**: Unit tests ensuring consistency between implementations
- **Educational Notebooks**: Examples demonstrating various use cases

## Installation

```bash
# Clone the repository
git clone https://github.com/whitham-powell/kernel-target-alignment.git
cd kernel-target-alignment

# If using uv (recommended)
uv sync              # Sync dependencies
make env             # Install package with dev extras

# Or with standard pip
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from kta import kta
from kta.kernels import rbf, polynomial

# Generate sample data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Compute RBF kernel matrix
K_rbf = rbf(X, gamma=0.1)

# Calculate kernel-target alignment
alignment_score = kta(K_rbf, y)
print(f"RBF KTA: {alignment_score:.3f}")

# Compare different kernels
K_poly = polynomial(X, degree=3)
print(f"Polynomial KTA: {kta(K_poly, y):.3f}")
```

## PyTorch Integration for Learning

```python
import torch
from kta.modules import LearnableRBF, KernelCombiner
from kta import kta_torch

# Convert to PyTorch tensors
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.long)

# Create learnable RBF kernel
rbf_kernel = LearnableRBF(gamma_init=0.1)

# Optimize kernel parameters
optimizer = torch.optim.Adam(rbf_kernel.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    K = rbf_kernel(X_torch)
    loss = -kta_torch(K, y_torch)  # Maximize KTA
    loss.backward()
    optimizer.step()

# Combine multiple kernels
combiner = KernelCombiner([
    LearnableRBF(gamma_init=0.1),
    LearnablePolynomial(degree=3, coef0_init=1.0)
])
K_combined = combiner(X_torch)
```

## Project Structure

```
kernel-target-alignment/
├── src/kta/              # Core implementation
│   ├── core.py          # KTA computation functions
│   ├── kernels.py       # Kernel implementations
│   └── modules.py       # PyTorch learnable modules
├── notebooks/           # Example notebooks
│   ├── demos/          # Basic demonstrations
│   ├── learnable/      # Learning kernel parameters
│   └── sweeps/         # Parameter sweep experiments
├── tests/              # Unit tests
└── Makefile           # Development commands
```

## Examples

### Notebooks

1. **Iris Demo** (`notebooks/demos/01_iris_demo.ipynb`): Basic KTA demonstration on the Iris dataset
2. **Comprehensive Demo** (`notebooks/demos/02_kta_comprehensive_demo.ipynb`): Detailed exploration of KTA properties
3. **Single Kernel Optimization** (`notebooks/learnable/01_optimize_single_kernel.ipynb`): Learning optimal kernel parameters
4. **Kernel Combination** (`notebooks/learnable/03_learnable_kernel_weights.ipynb`): Combining multiple kernels with learnable weights

### Running Examples

```bash
# Run all tests
make test

# Convert notebooks to markdown for viewing
make md

# Extract figures from notebooks
make plots
```

## API Reference

### Core Functions

- `kta(K, y)`: Compute kernel-target alignment (NumPy)
- `kta_torch(K, y)`: Compute kernel-target alignment (PyTorch)
- `alignment(K1, K2)`: Compute alignment between two kernels

### Kernel Functions

All kernels support both square (Gram) matrices and cross-kernel matrices:

- `rbf(X, Y=None, gamma=1.0)`: Radial Basis Function kernel
- `polynomial(X, Y=None, degree=3, gamma=1.0, coef0=0.0)`: Polynomial kernel
- `linear(X, Y=None)`: Linear kernel
- `sigmoid(X, Y=None, gamma=1.0, coef0=0.0)`: Sigmoid kernel

### PyTorch Modules

- `LearnableRBF`: RBF kernel with learnable gamma
- `LearnablePolynomial`: Polynomial kernel with learnable coefficient
- `LearnableSigmoid`: Sigmoid kernel with learnable parameters
- `KernelCombiner`: Combine multiple kernels with learnable weights

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kta

# Run specific test file
pytest tests/test_kta.py
```

## Development

This project uses:
- `pytest` for testing
- `jupytext` for notebook synchronization
- `black` and `isort` for code formatting

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kta2024,
  title={Kernel Target Alignment Implementation},
  author={Elijah Whitham-Powell},
  year={2024},
  url={https://github.com/whitham-powell/kernel-target-alignment}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation was developed as part of STAT 673 coursework at Portland State University.
