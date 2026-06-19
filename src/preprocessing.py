"""
preprocessing.py
-----------------
Shared preprocessing infrastructure used identically by the baseline
(classic kNN) and extension (weighted kNN) pipelines, and by the
cross-validation harness in cross_validation.py:

    stratified_train_test_split(X, y, test_size, random_seed)
    StandardScaler                      -- from-scratch, fit on train only

Both pieces are deliberately dependency-free (NumPy only), consistent with
the project's "no external ML libraries for core logic" constraint.
"""

import numpy as np


def stratified_train_test_split(X, y, test_size=0.2, random_seed=42, verbose=True):
    """
    Stratified train/test split that preserves per-class proportions.

    Each class is split independently in the given test_size proportion,
    then the per-class index sets are concatenated and shuffled. This keeps
    class balance in both splits close to the balance in the full dataset,
    regardless of how imbalanced the classes are.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)        -- integer class labels
    test_size : float in (0, 1), proportion of each class held out for test
    random_seed : int, for reproducibility
    verbose : bool, if True print per-class counts in each split

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_seed)

    classes = np.unique(y)
    train_idx_parts = []
    test_idx_parts = []

    for c in classes:
        c_idx = np.flatnonzero(y == c)
        c_idx = rng.permutation(c_idx)
        n_test = int(round(len(c_idx) * test_size))
        test_idx_parts.append(c_idx[:n_test])
        train_idx_parts.append(c_idx[n_test:])

    train_idx = rng.permutation(np.concatenate(train_idx_parts))
    test_idx = rng.permutation(np.concatenate(test_idx_parts))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if verbose:
        print("Stratified train/test split -- class counts per split")
        print(f"{'class':<8}{'train':<8}{'test':<8}")
        for c in classes:
            print(f"{c:<8}{np.sum(y_train == c):<8}{np.sum(y_test == c):<8}")
        print(f"{'total':<8}{len(y_train):<8}{len(y_test):<8}")

    return X_train, X_test, y_train, y_test


class StandardScaler:
    """
    From-scratch standard scaler: removes the mean and scales to unit
    variance, one column (feature) at a time.

    Must be fit on training data only -- transform() then applies the
    *training* mean/std to whatever is passed in (validation, test, or
    new data), so no information about the held-out set leaks into the
    scaling statistics.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # guard against division by zero for any (degenerate) constant feature
        std[std == 0] = 1.0
        self.std_ = std
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("StandardScaler instance is not fitted yet. "
                                "Call fit() before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    # Quick sanity checks -- run this file directly (`python preprocessing.py`)
    # to verify the split stays stratified and the scaler doesn't leak.
    rng = np.random.default_rng(0)

    # --- stratified split check, using an imbalanced class setup like the
    # penguins dataset (146 / 68 / 119) ---
    y_demo = np.array([0] * 146 + [1] * 68 + [2] * 119)
    X_demo = rng.normal(size=(len(y_demo), 4))

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_demo, y_demo, test_size=0.2, random_seed=42
    )

    for c in np.unique(y_demo):
        full_prop = np.mean(y_demo == c)
        train_prop = np.mean(y_train == c)
        test_prop = np.mean(y_test == c)
        assert abs(full_prop - train_prop) < 0.02, f"class {c} proportion drifted in train split"
        assert abs(full_prop - test_prop) < 0.03, f"class {c} proportion drifted in test split"
    print("OK -- stratified split preserves class proportions.\n")

    # --- no-leakage check: fit on one distribution, transform a deliberately
    # differently-shifted/scaled "test" set, and confirm the result uses the
    # TRAIN statistics, not statistics recomputed from the test data ---
    X_train_demo = rng.normal(loc=0.0, scale=1.0, size=(80, 4))
    X_test_demo = rng.normal(loc=5.0, scale=2.0, size=(20, 4))  # shifted on purpose

    scaler = StandardScaler()
    scaler.fit(X_train_demo)
    X_test_scaled = scaler.transform(X_test_demo)

    expected_using_train_stats = (X_test_demo - scaler.mean_) / scaler.std_
    leaky_using_test_stats = (X_test_demo - X_test_demo.mean(axis=0)) / X_test_demo.std(axis=0)

    assert np.allclose(X_test_scaled, expected_using_train_stats), \
        "transform() is not using the training statistics it was fit with!"
    assert not np.allclose(X_test_scaled, leaky_using_test_stats), \
        "transform() result matches test-set-derived statistics -- leakage!"
    print("OK -- StandardScaler.transform() uses train statistics only, no leakage.")