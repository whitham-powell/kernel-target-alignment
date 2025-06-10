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

# %% [markdown]
# # Comprehensive Analysis of Kernel-Target Alignment (KTA)
#
# **Goals:**
# - Demonstrate Kernel-Target Alignment across multiple datasets
# - Evaluate multiple kernel types (Linear, Polynomial, RBF, Sigmoid)
# - Compare accuracy with Kernel-Target Alignment (KTA)
#

# %%
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, svm
from sklearn.preprocessing import StandardScaler

try:
    from kta import kta  # noqa: F401
    from kta import linear, polynomial, rbf, sigmoid  # noqa: F401
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
    from kta import kta, linear, polynomial, rbf, sigmoid

# %% [markdown]
# ## 1️⃣ Setup Datasets

# %%
dataset_loaders = {
    "Iris": datasets.load_iris,
    "Digits": datasets.load_digits,
    "Breast Cancer": datasets.load_breast_cancer,
}

# %% [markdown]
# ## 2️⃣ Define Kernels and Hyperparameters

# %%
kernels = {
    "Linear": lambda X, Xp=None: linear(X, Xp),
    "Polynomial": lambda X, Xp=None: polynomial(X, Xp, degree=3, c=1),
    "RBF": lambda X, Xp=None: rbf(X, Xp, gamma=1),
    "Sigmoid": lambda X, Xp=None: sigmoid(X, Xp, gamma=0.01, c=0),
}

# %% [markdown]
# ## 3️⃣ Run Experiments and Collect Results

# %%
results = {}

for ds_name, loader in dataset_loaders.items():
    X, y = loader(return_X_y=True)
    X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    results[ds_name] = {}

    for kernel_name, kernel_fn in kernels.items():
        K_tr = kernel_fn(X_tr)
        K_te = kernel_fn(X_te, X_tr)

        align_score = kta(K_tr, y_tr)

        clf = svm.SVC(kernel="precomputed", C=1.0)
        clf.fit(K_tr, y_tr)
        accuracy = clf.score(K_te, y_te)

        results[ds_name][kernel_name] = {"Alignment": align_score, "Accuracy": accuracy}

# %% [markdown]
# ## 4️⃣ Plot Alignment vs Accuracy

# %%
fig, axs = plt.subplots(1, len(dataset_loaders), figsize=(16, 5), sharey=True)

for i, (ds_name, kernels_result) in enumerate(results.items()):
    alignments = [res["Alignment"] for res in kernels_result.values()]
    accuracies = [res["Accuracy"] for res in kernels_result.values()]
    kernel_labels = list(kernels_result.keys())

    axs[i].scatter(alignments, accuracies, s=100)

    for j, kernel_label in enumerate(kernel_labels):
        axs[i].annotate(
            kernel_label,
            (alignments[j], accuracies[j]),
            xytext=(5, -5),
            textcoords="offset points",
            fontsize=9,
        )

    axs[i].set_title(ds_name)
    axs[i].set_xlabel("Alignment")
    if i == 0:
        axs[i].set_ylabel("Accuracy")

    axs[i].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Kernel-Target Alignment vs. Accuracy across Datasets")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [markdown]
# ## 5️⃣ Insights
# - Kernels with higher alignment generally yield higher accuracy.
# - KTA can guide effective kernel selection.
