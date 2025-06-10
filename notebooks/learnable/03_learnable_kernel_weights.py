# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %%
# notebooks/learnable/03_learnable_kernel_weights.py
# %% [markdown]
# # Jointly Learn Kernel Parameters and Weights
# - Dataset: Breast Cancer
# - Kernels: Learnable RBF, Polynomial, Sigmoid
# - Goal: Learn both parameters and convex weights to maximize KTA

# %%
try:
    from kta import (
        KernelCombiner,
        LearnablePolynomial,
        LearnableRBF,
        LearnableSigmoid,
        kta_torch,
    )
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "git+https://github.com/whitham-powell/kernel-target-alignment.git",
        ],
    )
    from kta import (
        LearnableRBF,
        LearnablePolynomial,
        LearnableSigmoid,
        KernelCombiner,
        kta_torch,
    )

import matplotlib.pyplot as plt
import torch
from sklearn import datasets, model_selection, preprocessing
from sklearn.svm import SVC

# %% [markdown]
# ## 1. Load and preprocess data

# %%
X, y = datasets.load_breast_cancer(return_X_y=True)
y = (y * 2 - 1).astype(float)

X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y,
)

scaler = preprocessing.StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

X = torch.tensor(X_tr, dtype=torch.float32)
y = torch.tensor(y_tr, dtype=torch.float32)
X_test = torch.tensor(X_te, dtype=torch.float32)
y_test = y_te

# %% [markdown]
# ## 2. Build learnable kernel combiner

# %%
kernels = [
    LearnableRBF(gamma_init=1.0),
    LearnablePolynomial(degree=2, c_init=1.0),
    LearnableSigmoid(gamma_init=0.01, c_init=0.0),
]

model = KernelCombiner(kernels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# %% [markdown]
# ## 3. Optimize KTA

# %%
alignments = []
weights_history = []
accuracies = []
params_history = []
accuracy_freq = 1  # Check accuracy every `accuracy_freq` epochs

for epoch in range(100):
    K_train = model(X)
    loss = -kta_torch(K_train, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    alignments.append(-loss.item())
    with torch.no_grad():
        weights_history.append(torch.softmax(model.raw_weights, dim=0).cpu().tolist())
        params = [p.item() for k in model.kernels for p in k.parameters()]
        params_history.append(params)

    if epoch % accuracy_freq == 0:
        with torch.no_grad():
            K_train_np = K_train.detach().cpu().numpy()
            K_test_np = model(X_test, X).detach().cpu().numpy()
        clf = SVC(kernel="precomputed")
        clf.fit(K_train_np, y.cpu().numpy())
        acc = clf.score(K_test_np, y_test)
        accuracies.append((epoch, acc))

# %% [markdown]
# ## 4. Plot alignment and weight evolution

# %%
fig, ax1 = plt.subplots()
ax1.plot(alignments, label="KTA")
ax1.set_ylabel("Alignment")
ax1.set_xlabel("Epoch")
ax1.set_title("Learnable Kernels: Alignment")
plt.show()

# %%
weights_history = torch.tensor(weights_history)
fig, ax = plt.subplots()
for i in range(weights_history.shape[1]):
    ax.plot(weights_history[:, i], label=f"Kernel {i+1}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Weight")
ax.set_title("Kernel Weights Over Time")
ax.legend()
plt.show()

# %% [markdown]
# ## 5. Accuracy vs Epoch

# %%
if accuracies:
    epochs, accs = zip(*accuracies)
    plt.plot(epochs, accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"SVC Accuracy (Every {accuracy_freq} Epochs)")
    plt.grid(True)
    plt.show()

# %% [markdown]
# ## 6. Final Weights, Alignment, Parameters

# %%
print("Final weights:", torch.softmax(model.raw_weights, dim=0))
print(f"Final alignment: {alignments[-1]:.4f}")
print("Final kernel parameters:")
for i, k in enumerate(model.kernels):
    for name, param in k.named_parameters():
        print(f"Kernel {i+1} - {name}: {param.item():.4f}")

# %% [markdown]
# ## 7. Plot Kernel Parameter Evolution
# Tracks each learnable parameter (e.g., gamma, c) over time.

# %%
params_history = torch.tensor(params_history)
param_labels = ["RBF gamma", "Poly c", "Sigmoid gamma", "Sigmoid c"]

fig, ax = plt.subplots()
for i in range(params_history.shape[1]):
    ax.plot(params_history[:, i], label=param_labels[i])
ax.set_xlabel("Epoch")
ax.set_ylabel("Parameter Value")
ax.set_title("Learnable Kernel Parameters Over Time")
ax.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
