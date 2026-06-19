"""
cross_validation.py
--------------------
Reusable 5-fold stratified cross-validation harness, parameterised by `k`
and `method`. Used identically to tune classic kNN and weighted kNN: pass
in `knn_predict` or `weighted_knn_predict` (or any callable with the same
signature) and a range of k values, and get back per-k mean/std accuracy
across folds.

Exposes:
    make_stratified_folds(y, n_folds, random_seed)
    cross_validate(X, y, k_values, method, n_folds, random_seed)
"""

import numpy as np

from preprocessing import StandardScaler


def make_stratified_folds(y, n_folds=5, random_seed=42):
    """
    Build n_folds stratified folds: each fold gets roughly the same
    per-class proportions as the full dataset.

    Parameters
    ----------
    y : array-like, shape (n_samples,)  -- integer class labels
    n_folds : int
    random_seed : int

    Returns
    -------
    list of np.ndarray
        fold_indices[i] = indices of samples assigned to fold i
    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)

    fold_indices = [[] for _ in range(n_folds)]
    for c in classes:
        c_idx = rng.permutation(np.flatnonzero(y == c))
        # as-even-as-possible split of this class's indices across folds
        splits = np.array_split(c_idx, n_folds)
        for fold_i, split in enumerate(splits):
            fold_indices[fold_i].extend(split.tolist())

    return [rng.permutation(np.array(fold, dtype=int)) for fold in fold_indices]


def cross_validate(X, y, k_values, method, n_folds=5, random_seed=42, verbose=False):
    """
    5-fold (by default) stratified cross-validation.

    For every k in k_values, and for every fold: the other (n_folds - 1)
    folds are used as training data, a StandardScaler is fit on that
    training data only and applied to the held-out fold (no leakage),
    `method` is called to predict on the held-out fold, and accuracy is
    recorded. Mean and std accuracy across folds are reported per k.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)         -- integer class labels
    k_values : iterable of int
    method : callable(X_train, y_train, X_val, k) -> predicted labels
        e.g. knn_predict or weighted_knn_predict
    n_folds : int, default 5
    random_seed : int, for reproducible folds
    verbose : bool, if True print mean/std accuracy as each k finishes

    Returns
    -------
    dict[int, dict]
        {k: {"mean_accuracy": float,
             "std_accuracy": float,
             "fold_accuracies": list[float]}}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    folds = make_stratified_folds(y, n_folds=n_folds, random_seed=random_seed)
    results = {}

    for k in k_values:
        fold_accuracies = []

        for fold_i in range(n_folds):
            val_idx = folds[fold_i]
            train_idx = np.concatenate(
                [folds[j] for j in range(n_folds) if j != fold_i]
            )

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)   # fit on train fold(s) only
            X_val_scaled = scaler.transform(X_val)            # apply same stats to val fold

            y_pred = method(X_train_scaled, y_train, X_val_scaled, k)
            fold_accuracies.append(float(np.mean(y_pred == y_val)))

        fold_accuracies = np.array(fold_accuracies)
        results[k] = {
            "mean_accuracy": float(fold_accuracies.mean()),
            "std_accuracy": float(fold_accuracies.std()),
            "fold_accuracies": fold_accuracies.tolist(),
        }

        if verbose:
            print(f"k={k:<3} mean_acc={results[k]['mean_accuracy']:.4f} "
                  f"std_acc={results[k]['std_accuracy']:.4f}")

    return results
