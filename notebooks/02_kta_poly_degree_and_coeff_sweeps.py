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
# # Polynomial Degree Sweep: Kernel–Target Alignment vs. SVM Accuracy
# *Dataset*: Breast-Cancer (binary, 30 features)
# *Message*: The Degree that **maximizes KTA** almost always maximizes test accuracy.


# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, svm
from sklearn.preprocessing import StandardScaler

try:
    from kta import kta  # noqa: F401
    from kta.kernels import polynomial  # noqa: F401
except ModuleNotFoundError:
    print("Installing kta package...")
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
    from kta import kta
    from kta.kernels import polynomial

# 1️⃣ data
X, y = datasets.load_breast_cancer(return_X_y=True)
y = np.where(y == 0, -1, 1)
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y,
)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# %%
# 2️⃣ sweep degrees constant c
degrees = [2, 3, 4, 5, 6]
c = 1.0

alignments, accuracies = [], []
for d in degrees:
    K_tr = polynomial(X_tr, None, degree=d, c=c)
    alignments.append(kta(K_tr, y_tr))

    K_te = polynomial(X_te, X_tr, degree=d, c=c)

    clf = svm.SVC(kernel="precomputed", C=1.0)
    clf.fit(K_tr, y_tr)
    accuracies.append(clf.score(K_te, y_te))


# %%
# 3️⃣ plot Degree vs Alignment & Accuracy
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

ax1.plot(degrees, alignments, marker="o", label="KTA")
ax2.plot(degrees, accuracies, marker="x", color="tab:orange", label="Accuracy")

ax1.set_xlabel("Polynomial degree")
ax1.set_ylabel("Alignment")
ax2.set_ylabel("Test accuracy")
fig.legend()
fig.suptitle("Polynomial degree sweep — KTA vs Accuracy")
fig.tight_layout()
plt.show()

# %%
# 4️⃣ sweep coefficient c
cs = np.logspace(-2, 2, 40)
degree = 2
alignments, accuracies = [], []
for c in cs:
    K_tr = polynomial(X_tr, None, degree=degree, c=c)
    alignments.append(kta(K_tr, y_tr))

    K_te = polynomial(X_te, X_tr, degree=degree, c=c)

    clf = svm.SVC(kernel="precomputed", C=1.0)
    clf.fit(K_tr, y_tr)
    accuracies.append(clf.score(K_te, y_te))

# %%
# 5️⃣ plot c vs Alignment & Accuracy
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
ax1.plot(np.log10(cs), alignments, marker="o", label="KTA")
ax2.plot(np.log10(cs), accuracies, marker="x", color="tab:orange", label="Accuracy")
ax1.set_xlabel("log₁₀ c")
ax1.set_ylabel("Alignment")
ax2.set_ylabel("Test accuracy")
fig.legend()
fig.suptitle(f"Polynomial c sweep — KTA vs Accuracy (degree={degree})")
fig.tight_layout()
plt.show()

# %%
