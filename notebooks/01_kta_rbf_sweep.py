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
# # RBF γ Sweep: Kernel–Target Alignment vs. SVM Accuracy
# *Dataset*: Breast-Cancer (binary, 30 features)
# *Message*: The γ that **maximises KTA** almost always maximises test accuracy.


# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, model_selection, svm

from kta import kta
from kta.kernels import rbf

# %%
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

# 2️⃣ sweep γ
gammas = np.logspace(-5, 2, 40)
alignment = []
accuracy = []

for g in gammas:
    K_tr = rbf(X_tr, gamma=g)
    alignment.append(kta(K_tr, y_tr))

    clf = svm.SVC(kernel="rbf", gamma=g, C=1.0)
    clf.fit(X_tr, y_tr)
    accuracy.append(clf.score(X_te, y_te))

alignment = np.array(alignment)
accuracy = np.array(accuracy)

# %%
# 3️⃣ plot
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

ax1.plot(np.log10(gammas), alignment, marker="o", label="KTA")
ax2.plot(np.log10(gammas), accuracy, marker="x", color="tab:orange", label="Accuracy")

ax1.set_xlabel("log₁₀ γ")
ax1.set_ylabel("Alignment")
ax2.set_ylabel("Test accuracy")
fig.suptitle("RBF γ sweep — alignment closely tracks accuracy")
fig.tight_layout()
plt.show()

# %%
