# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv (3.14.3.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pattern Recognition and Machine Learning
#
# ## Project
#
# ### Comparative Study of kNN and Weighted kNN Classification on the Palmer Penguins Dataset

# %% [markdown]
# #### Completed by: **Group 6**
#
# | Index | Full Name |
# | ----- | ----- |
# | 66m/25 | Ivona Petrović |
# | 144m/25 | Vojin Đorđević |

# %% [markdown]
# **Imports and data loading.** We use `pandas` for loading and manipulating the tabular dataset, `os` for file system operations, and `numpy` for saving the final feature/label arrays. `load_penguins_data` fetches the Palmer Penguins dataset from a public CSV source and we save an untouched raw copy to `../data/raw/penguins_.csv` before any cleaning.

# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


# %%
def load_penguins_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    df = pd.read_csv(url)
    
    print("Raw shape:", df.shape)
    
    return df

df = load_penguins_data()
df.to_csv("../data/raw/penguins_.csv", index=False)

# %% [markdown]
# **Inspecting the raw data.** We preview the first rows to check column names and types, count missing values per column, and look at the rows/indices where values are missing, to decide how to handle them.

# %%
df.head()

# %%
missing_counts = df.isnull().sum().to_frame(name="missing_count")
missing_counts

# %%
rows_with_missing = df[df.isnull().any(axis=1)]
rows_with_missing

# %%
missing_indices = df[df.isnull().any(axis=1)].index
missing_indices 

# %% [markdown]
# **Cleaning the data.** Since missing entries are a small fraction of the dataset, we drop them with `dropna()` rather than imputing, producing `df_clean`. Shapes before/after are printed to quantify how much data was lost.

# %%
df_clean = df.dropna()

print("Before:", df.shape)
print("After:", df_clean.shape)

# %% [markdown]
# **Feature/target preparation and saving.** We select the four numeric measurements (bill length, bill depth, flipper length, body mass) as features `X`, and species as the target. Species names are mapped to integers $\{0,1,2\}$ to produce `y_encoded`, while `df_eda` is kept as an independent copy for later exploratory analysis. Finally, the cleaned dataset and feature/label arrays are saved to disk for use in subsequent notebook stages.

# %%
X = df_clean[[ "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df_clean["species"]

# %%
label_map = {
    "Adelie": 0,
    "Chinstrap": 1,
    "Gentoo": 2
}

y_encoded = y.map(label_map)

df_eda = df_clean.copy()  # separate dataset for EDA (kept independent from modeling data)

# %%
os.makedirs("../data", exist_ok=True)

df_clean.to_csv("../data/penguins_clean.csv", index=False)
np.save("../data/X.npy", X.values)
np.save("../data/y.npy", y_encoded.values)

# %% [markdown]
# ##  Exploratory Data Analysis
#
# We explore the cleaned dataset (`df_eda`) before modeling: summary statistics, class balance, feature correlations, distributions by species, a from-scratch PCA projection, and outlier inspection via z-scores. Figures are saved to `../figures`.

# %%
feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
species_list = sorted(df_eda["species"].unique())
colors = {"Adelie": "tab:blue", "Chinstrap": "tab:orange", "Gentoo": "tab:green"}

# %% [markdown]
# **Summary statistics by species.** We compute mean, standard deviation, min, and max for each of the four numeric features, grouped by species, to get a quantitative sense of how the species differ before any modeling.

# %%
summary_stats = df_eda.groupby("species")[feature_cols].agg(["mean", "std", "min", "max"])
summary_stats = summary_stats.round(2)
summary_stats

# %% [markdown]
# **Class distribution.** We plot the number of samples per species after cleaning, to check whether the dataset is balanced or skewed toward one class — class imbalance can affect both the choice of evaluation metric and kNN's neighbor voting.

# %%
class_counts = df_eda["species"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values, color=[colors[s] for s in class_counts.index])
plt.title("Class distribution after cleaning")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../figures/class_distribution.png", dpi=150)
plt.show()

print(class_counts)
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio (largest/smallest class): {imbalance_ratio:.2f}")


# %% [markdown]
# **Note on class imbalance.** The dataset shows moderate class imbalance — Adelie (146 samples) has roughly 2.15× more samples than Chinstrap (68 samples), with Gentoo (119 samples) in between. This is accounted for during model evaluation, where overall accuracy alone may be misleading and per-class metrics (e.g., precision/recall) are also reported.

# %% [markdown]
# **Correlation heatmap.** We compute the pairwise Pearson correlation among the four numeric features to identify redundant or strongly related measurements, which is relevant both for interpreting the data and for understanding what PCA will later capture.

# %%
def correlation_matrix(X):
    n_features = X.shape[1]
    corr = np.zeros((n_features, n_features))
    
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    
    for i in range(n_features):
        for j in range(n_features):
            cov_ij = np.mean((X[:, i] - means[i]) * (X[:, j] - means[j]))
            corr[i, j] = cov_ij / (stds[i] * stds[j])
    
    return corr

X_features = df_eda[feature_cols].values
corr_array = correlation_matrix(X_features)

corr_matrix = pd.DataFrame(corr_array, index=feature_cols, columns=feature_cols).round(2)
corr_matrix

# %%
corr_no_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
strongest = corr_no_diag.abs().unstack().idxmax()
strongest_val = corr_matrix.loc[strongest]
print(f"Strongest correlation: {strongest[0]} vs {strongest[1]} (r = {strongest_val:.2f})")

# %% [markdown]
# **Interpretation.** The strongest correlation is between flipper_length_mm and body_mass_g (r = 0.87), indicating that larger-bodied penguins tend to have proportionally longer flippers. This strong positive relationship suggests these two features carry overlapping information, which is consistent with the high variance explained by the first principal component in the PCA projection below.

# %% [markdown]
# **Feature distributions by species.** We use overlapping histograms for each of the four features, split by species, to see how much the distributions overlap or separate — features with clearly separated distributions are likely to be more informative for classification.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, feat in zip(axes.flatten(), feature_cols):
    for s in species_list:
        ax.hist(df_eda[df_eda["species"] == s][feat], alpha=0.5, label=s, color=colors[s], bins=15)
    ax.set_title(feat)
    ax.legend()

plt.tight_layout()
plt.savefig("../figures/distributions_by_species.png", dpi=150)
plt.show()

# %% [markdown]
# **Pairwise feature relationships.** We plot the two most relevant feature pairs (selected based on the correlation analysis above), colored by species, to visually inspect class separability.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pairs = [("flipper_length_mm", "body_mass_g"), ("bill_length_mm", "bill_depth_mm")]

for ax, (x_feat, y_feat) in zip(axes, pairs):
    for s in species_list:
        subset = df_eda[df_eda["species"] == s]
        ax.scatter(subset[x_feat], subset[y_feat], color=colors[s], label=s, alpha=0.6)
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    ax.legend()

plt.tight_layout()
plt.savefig("../figures/key_scatter_pairs.png", dpi=150)
plt.show()

# %% [markdown]
# **PCA projection.** We standardize the four features (zero mean, unit variance) and project them onto the first two principal components, to visualize class separability in 2D before modeling. We also report the explained variance ratio of each component.

# %%
X = df_eda[feature_cols].values
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

cov_matrix = (X_scaled.T @ X_scaled) / X_scaled.shape[0]
eigvals, eigvecs = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

X_pca = X_scaled @ eigvecs[:, :2]
explained_var = eigvals / eigvals.sum()

plt.figure(figsize=(6, 5))
for s, c in colors.items():
    mask = (df_eda["species"] == s).values
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=s, color=c, alpha=0.7)

plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("PCA projection")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/pca_projection.png", dpi=150)
plt.show()

print("Explained variance ratio:", explained_var[:2])

# %% [markdown]
# **Outlier inspection via z-scores.** We compute per-feature z-scores on the standardized data and flag any sample with |z| > 3 on at least one feature as a potential outlier. We list the flagged rows and record an explicit decision on whether to keep or remove them.

# %%
z_scores = pd.DataFrame(X_scaled, columns=feature_cols, index=df_eda.index)
outliers = df_eda[(z_scores.abs() > 3).any(axis=1)]

print(f"Flagged outliers (|z| > 3): {len(outliers)}")
outliers

# %% [markdown]
# **Decision on flagged outliers.** No samples exceeded |z| > 3 on any of the four features, indicating no extreme outliers in the cleaned dataset. No additional rows were removed beyond the missing-value cleaning already performed.

# %% [markdown]
# ## Hyperparameter Search: Choosing k via Cross-Validation
#
# Before evaluating on the held-out test set, we tune `k` for both classic kNN and weighted kNN
# using the generalised `cross_validate(X, y, k_values, method)`. We sweep `k = 1...20` for both methods, record the mean CV accuracy
# per `k`, and select the optimal `k` for each method to be used in all subsequent test-set
# evaluations.
#
# **Tie-breaking rule:** if multiple `k` values tie for the best mean CV accuracy (compared after
# rounding to 6 decimal places, to ignore floating-point noise), the **smallest `k`** is selected, favouring the simpler, less variance-prone model.

# %%
sys.path.append("../src")

from cross_validation import cross_validate
from knn import knn_predict, weighted_knn_predict

K_MIN = 1
K_MAX = 20          # search k = 1...20 (sqrt(n_train) ~ 16, so this comfortably covers it)
N_FOLDS = 5
RANDOM_SEED = 42
ROUND_DECIMALS = 6   # precision used when checking for ties in mean accuracy

METHODS = {"knn": knn_predict, "weighted_knn": weighted_knn_predict}

X = np.load("../data/X.npy")
y = np.load("../data/y.npy")


# %% [markdown]
# **Running the search.** `run_search` calls `cross_validate` once per method over `k = 1...20`
# and reshapes the results into a tidy, plottable DataFrame.

# %%
def run_search(X, y, k_values=range(K_MIN, K_MAX + 1), methods=METHODS, n_folds=N_FOLDS, random_seed=RANDOM_SEED, verbose=True):
    raw_results = {}

    for name, method_fn in methods.items():
        if verbose:
            print(f"\n=== {name} ===")
        raw_results[name] = cross_validate(X, y, k_values, method=method_fn, n_folds=n_folds, random_seed=random_seed, verbose=verbose)

    rows = []
    for method_name, per_k in raw_results.items():
        for k, stats in per_k.items():
            rows.append({
                "method": method_name,
                "k": k,
                "mean_accuracy": stats["mean_accuracy"],
                "std_accuracy": stats["std_accuracy"]
            })
    results_df = pd.DataFrame(rows)
    return results_df  

results_df = run_search(X, y)


# %% [markdown]
# **Selecting the optimal k.** `select_optimal_k` finds the best mean CV accuracy for a given
# method and applies the explicit tie-breaking rule (smallest `k` among ties), so the result is
# deterministic and reproducible.

# %%
def select_optimal_k_simple(results_df, method_name, round_decimals=6):
    rows = results_df[results_df["method"] == method_name]
    candidates = list(zip(rows["k"], rows["mean_accuracy"], rows["std_accuracy"]))

    best_acc = max(round(acc, round_decimals) for k, acc, std in candidates)

    
    tied = [(k, acc, std) for k, acc, std in candidates if round(acc, round_decimals) == best_acc]

    
    best_k, best_acc_real, best_std = min(tied, key=lambda row: row[0])

    return {"method": method_name, "optimal_k": int(best_k),
            "mean_accuracy": float(best_acc_real), "std_accuracy": float(best_std)}

# %%
optimal_results = [select_optimal_k_simple(results_df, name) for name in METHODS]
optimal_df = pd.DataFrame(optimal_results)

print("\n=== Optimal k per method (tie-break: smallest k) ===")
optimal_df

# %% [markdown]
# **Accuracy vs k, both methods.** Visualising the full sweep alongside the selected optimum makes
# it easy to see how sensitive each method is to `k`, and whether the chosen `k` sits in a flat,
# stable region of the curve or a sharp, noise-driven peak.

# %%
results_df = pd.read_csv("../data/cv_results_long.csv")

fig, ax = plt.subplots(figsize=(8, 5))
for method, color, label in zip(
    ["knn", "weighted_knn"],
    ["tab:blue", "tab:orange"],
    ["kNN", "Weighted kNN"],
):
    sub = results_df[results_df["method"] == method]
    ax.plot(sub["k"], sub["mean_accuracy"], marker="o", color=color, label=label)
    opt_row = optimal_df[optimal_df["method"] == method]
    opt_k   = int(opt_row["optimal_k"].values[0])
    opt_acc = float(opt_row["mean_accuracy"].values[0])
    ax.scatter([opt_k], [opt_acc], color=color, s=150, zorder=5,
               edgecolor="black", label=f"{label} optimal (k={opt_k})")

ax.set_xlabel("k")
ax.set_ylabel("Mean CV accuracy (5-fold stratified)")
ax.set_title("Cross-validation accuracy vs k")
ax.set_xticks(results_df["k"].unique())
ax.legend()
plt.tight_layout()
plt.savefig("../figures/cv_accuracy_vs_k.png", dpi=150)
plt.show()

# %% [markdown]
# **Saving results for later use.** The full per-k results and the selected optimal `k` per method
# are saved to `data/` so the test-set evaluation stage can load them directly without re-running
# cross-validation.

# %%
results_df.to_csv("../data/cv_results_long.csv", index=False)
optimal_df.to_csv("../data/cv_optimal_k.csv", index=False)

print("Saved: data/cv_results_long.csv (full per-k results), "
      "data/cv_optimal_k.csv (optimal k per method)")

# %% [markdown]
# ## Test-Set Evaluation
#
# Before evaluating on the held-out test set, we perform a stratified train/test
# split (80/20, same `random_seed=42` as cross-validation to guarantee no data
# leakage), standardise using a `StandardScaler` fit on training data only, and
# load the optimal `k` per method from the cross-validation results saved in previous section.

# %%
from preprocessing import stratified_train_test_split, StandardScaler


LABEL_NAMES = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
COLORS      = ["tab:blue", "tab:orange", "tab:green"]
N_CLASSES   = 3

X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

X_train, X_test, y_train, y_test = stratified_train_test_split(
    X, y, test_size=0.2, random_seed=42
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)   # uses train mean/std 
optimal_df = pd.read_csv("../data/cv_optimal_k.csv")
k_knn  = int(optimal_df[optimal_df["method"] == "knn"]["optimal_k"].values[0])
k_wknn = int(optimal_df[optimal_df["method"] == "weighted_knn"]["optimal_k"].values[0])

print(f"Optimal k  —  kNN: {k_knn},  Weighted kNN: {k_wknn}")

y_pred_knn  = knn_predict(X_train_s, y_train, X_test_s, k=k_knn)
y_pred_wknn = weighted_knn_predict(X_train_s, y_train, X_test_s, k=k_wknn)


# %% [markdown]
# ### Accuracy, Precision, Recall, F1
#
# We compute all metrics from scratch using the confusion matrix. Macro-averaging
# treats each class equally regardless of size, which is appropriate given the
# moderate class imbalance noted in some previous section.

# %%
def confusion_matrix_scratch(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true, y_pred, n_classes):
    cm = confusion_matrix_scratch(y_true, y_pred, n_classes)
    per_class = {}
    for c in range(n_classes):
        tp   = cm[c, c]
        fp   = cm[:, c].sum() - tp
        fn   = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1}
    macro = {m: float(np.mean([per_class[c][m] for c in range(n_classes)]))
             for m in ["precision", "recall", "f1"]}
    return cm, per_class, macro


cm_knn,  per_knn,  macro_knn  = precision_recall_f1(y_test, y_pred_knn,  N_CLASSES)
cm_wknn, per_wknn, macro_wknn = precision_recall_f1(y_test, y_pred_wknn, N_CLASSES)

acc_knn  = float(np.mean(y_pred_knn  == y_test))
acc_wknn = float(np.mean(y_pred_wknn == y_test))

rows = []
for c in range(N_CLASSES):
    rows.append({
        "class":          LABEL_NAMES[c],
        "knn_precision":  round(per_knn[c]["precision"],  4),
        "knn_recall":     round(per_knn[c]["recall"],     4),
        "knn_f1":         round(per_knn[c]["f1"],         4),
        "wknn_precision": round(per_wknn[c]["precision"], 4),
        "wknn_recall":    round(per_wknn[c]["recall"],    4),
        "wknn_f1":        round(per_wknn[c]["f1"],        4),
    })
rows.append({
    "class":          "macro",
    "knn_precision":  round(macro_knn["precision"],  4),
    "knn_recall":     round(macro_knn["recall"],     4),
    "knn_f1":         round(macro_knn["f1"],         4),
    "wknn_precision": round(macro_wknn["precision"], 4),
    "wknn_recall":    round(macro_wknn["recall"],    4),
    "wknn_f1":        round(macro_wknn["f1"],        4),
})

metrics_df = pd.DataFrame(rows).set_index("class")
print(f"Test accuracy  —  kNN: {acc_knn:.4f},  Weighted kNN: {acc_wknn:.4f}\n")
metrics_df

# %% [markdown]
# ### Confusion Matrices

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = [LABEL_NAMES[c] for c in range(N_CLASSES)]

for ax, cm, title in zip(
    axes,
    [cm_knn, cm_wknn],
    [f"kNN (k={k_knn})", f"Weighted kNN (k={k_wknn})"],
):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im, ax=axes[-1])
plt.tight_layout()
plt.savefig("../figures/confusion_matrices.png", dpi=150)
plt.show()

# %% [markdown]
# Both confusion matrices are identical: both methods correctly classify all 67
# test points at `k = 4`. The diagonal entries confirm perfect per-class recall
# for all three species.

# %% [markdown]
# ### Decision Boundaries on PCA Projection
#
# **Note:** boundaries are trained on the first two PCA components only (not the
# full 4-D feature space) — included for visual illustration, not as a
# performance measure.

# %%
cov     = (X_train_s.T @ X_train_s) / X_train_s.shape[0]
eigvals, eigvecs = np.linalg.eigh(cov)
eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
W       = eigvecs[:, :2]

X_train_pca = X_train_s @ W
explained   = np.sort(eigvals)[::-1] / eigvals.sum()

margin = 0.5
x_min, x_max = X_train_pca[:, 0].min() - margin, X_train_pca[:, 0].max() + margin
y_min, y_max = X_train_pca[:, 1].min() - margin, X_train_pca[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))
grid_pca = np.c_[xx.ravel(), yy.ravel()]

Z_knn  = knn_predict(X_train_pca, y_train, grid_pca, k=k_knn).reshape(xx.shape)
Z_wknn = weighted_knn_predict(X_train_pca, y_train, grid_pca, k=k_wknn).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, Z, title in zip(
    axes,
    [Z_knn, Z_wknn],
    [f"kNN (k={k_knn})", f"Weighted kNN (k={k_wknn})"],
):
    ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORS)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="gray", linewidths=0.8, linestyles="--")
    for c, color in enumerate(COLORS):
        mask = y_train == c
        ax.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                   color=color, alpha=0.6, s=30, label=LABEL_NAMES[c])
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../figures/decision_boundaries_pca.png", dpi=150)
plt.show()

# %% [markdown]
# ### Disagreement Analysis
#
# We check where the two methods give different predictions on the test set.

# %%
disagree_mask = y_pred_knn != y_pred_wknn
n_disagree    = int(disagree_mask.sum())

print(f"Test set size : {len(y_test)}")
print(f"Disagreements : {n_disagree}  ({n_disagree / len(y_test) * 100:.1f}%)")

if n_disagree > 0:
    disagree_df = pd.DataFrame({
        "true_label": [LABEL_NAMES[c] for c in y_test[disagree_mask]],
        "knn_pred":   [LABEL_NAMES[c] for c in y_pred_knn[disagree_mask]],
        "wknn_pred":  [LABEL_NAMES[c] for c in y_pred_wknn[disagree_mask]],
    })
    disagree_df

# %% [markdown]
# Both methods agree on every test point at `k = 4`. This is consistent with the
# EDA findings: Gentoo is clearly separable on flipper length and
# body mass, and while Adelie/Chinstrap overlap on bill depth, the four nearest
# neighbours at `k = 4` are unambiguous enough that both voting schemes reach the
# same decision. The low CV standard deviation (0.006) confirms the result is
# stable across all five folds, not a lucky split.
