"""
preprocessing.py
-----------------
Stratified train/test split and a from-scratch StandardScaler. NumPy only.

    stratified_train_test_split(X, y, test_size, random_seed)
    StandardScaler
"""

import numpy as np


def stratified_train_test_split(X, y, test_size=0.2, random_seed=42, verbose=True):
    """
    Splits each class independently by test_size, then concatenates and
    shuffles -- keeps class proportions roughly equal in both splits.

    Returns X_train, X_test, y_train, y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_seed)

    classes = np.unique(y)
    train_idx_parts = []
    test_idx_parts = []

    for c in classes:
        c_idx = rng.permutation(np.flatnonzero(y == c))
        n_test = int(round(len(c_idx) * test_size))
        test_idx_parts.append(c_idx[:n_test])
        train_idx_parts.append(c_idx[n_test:])

    train_idx = rng.permutation(np.concatenate(train_idx_parts))
    test_idx = rng.permutation(np.concatenate(test_idx_parts))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if verbose:
        print(f"{'class':<8}{'train':<8}{'test':<8}")
        for c in classes:
            print(f"{c:<8}{np.sum(y_train == c):<8}{np.sum(y_test == c):<8}")
        print(f"{'total':<8}{len(y_train):<8}{len(y_test):<8}")

    return X_train, X_test, y_train, y_test


class StandardScaler:
    """Zero mean, unit variance per feature. fit() on train data only;
    transform() reuses the stored mean/std, so it never recomputes stats
    from whatever it's given."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # constant feature -> avoid divide by zero
        self.std_ = std
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    y_demo = np.array([0] * 146 + [1] * 68 + [2] * 119)
    X_demo = rng.normal(size=(len(y_demo), 4))
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_demo, y_demo, test_size=0.2, random_seed=42
    )
    for c in np.unique(y_demo):
        assert abs(np.mean(y_demo == c) - np.mean(y_train == c)) < 0.02
        assert abs(np.mean(y_demo == c) - np.mean(y_test == c)) < 0.03

    X_train_demo = rng.normal(loc=0.0, scale=1.0, size=(80, 4))
    X_test_demo = rng.normal(loc=5.0, scale=2.0, size=(20, 4))

    scaler = StandardScaler()
    scaler.fit(X_train_demo)
    X_test_scaled = scaler.transform(X_test_demo)

    using_train_stats = (X_test_demo - scaler.mean_) / scaler.std_
    using_test_stats = (X_test_demo - X_test_demo.mean(axis=0)) / X_test_demo.std(axis=0)

    assert np.allclose(X_test_scaled, using_train_stats)
    assert not np.allclose(X_test_scaled, using_test_stats)