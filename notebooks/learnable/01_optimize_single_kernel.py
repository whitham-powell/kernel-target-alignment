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
# notebooks/learnable/01_optimize_single_kernel.py
# %% [markdown]
# # Optimize Single Learnable Kernel (RBF)
# - Dataset: Breast Cancer
# - Model: LearnableRBF
# - Goal: Maximize KTA over time

# %%

import matplotlib.pyplot as plt
import torch
from sklearn import datasets, model_selection, preprocessing

try:
    from kta import LearnableRBF, kta_torch
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
    from kta import LearnableRBF, kta_torch

# %% [markdown]
# ## 1. Load and preprocess data

# %%
X, y = datasets.load_breast_cancer(return_X_y=True)
y = (y * 2 - 1).astype(float)  # convert to {-1, 1}

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

# %% [markdown]
# ## 2. Initialize model and optimizer

# %%
model = LearnableRBF(gamma_init=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# %% [markdown]
# ## 3. Train loop: maximize KTA

# %%
alignments = []
gammas = []

for epoch in range(100):
    K = model(X)
    loss = -kta_torch(K, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    alignments.append(-loss.item())
    gammas.append(model.gamma.item())

# %% [markdown]
# ## 4. Plot results

# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alignments, label="KTA", color="tab:blue")
ax2.plot(gammas, label="gamma", color="tab:orange")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Alignment", color="tab:blue")
ax2.set_ylabel("Gamma", color="tab:orange")
fig.suptitle("Learnable RBF: KTA vs Gamma")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Final gamma and alignment
# %%
print(f"Final gamma: {model.gamma.item():.4f}")
print(f"Final alignment: {alignments[-1]:.4f}")

# %%
