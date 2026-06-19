"""
knn.py
------
Baseline k-Nearest Neighbours classifier, implemented from scratch with
NumPy only.

Supports multi-class targets (e.g. the 3 Palmer Penguins species), encoded
as integer labels 0..C-1.

Exposes:
    euclidean_distance(x1, x2)         -- shared distance utility
    knn_predict_point(X_train, y_train, x_query, k)
    knn_predict(X_train, y_train, X_test, k)
"""

import numpy as np


def euclidean_distance(x1, x2):
    """
    Euclidean (L2) distance between two 1D vectors.

    d(x1, x2) = sqrt( sum_i (x1_i - x2_i)^2 )
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict_point(X_train, y_train, x_query, k=5):
    """
    Predict the class of a single query point using majority-vote kNN.

    Steps:
      1. Compute the Euclidean distance from x_query to every training point.
      2. Sort distances and take the indices of the k closest points.
      3. Count class votes among those k neighbours.
      4. Return the majority class.

    Tie-breaking rule
    ------------------
    If two or more classes receive the same (maximum) number of votes among
    the k neighbours, the tie is broken by:
        (a) first, the class with the smallest SUM of distances among its
            tied, voting neighbours (i.e. the class whose neighbours are,
            on average, closest to the query wins), and
        (b) if that is still tied, the class with the LOWEST class index
            (deterministic, reproducible fallback).
    This avoids relying on the arbitrary ordering np.argmax(np.bincount(...))
    would otherwise impose, and keeps the rule explicit and reproducible.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    x_query = np.asarray(x_query, dtype=float)

    # 1. distance to every training point
    distances = np.array([euclidean_distance(x, x_query) for x in X_train])

    # 2. indices of the k nearest neighbours
    nn_idx = np.argsort(distances)[:k]
    nn_labels = y_train[nn_idx]
    nn_distances = distances[nn_idx]

    # 3. majority vote (count occurrences of each class among neighbours)
    counts = np.bincount(nn_labels)
    max_votes = counts.max()
    tied_classes = np.flatnonzero(counts == max_votes)  # classes with top vote count

    if tied_classes.size == 1:
        return int(tied_classes[0])

    # 4. tie-break (a): smallest total distance among each tied class's neighbours
    sum_dist_per_class = {}
    for c in tied_classes:
        mask = nn_labels == c
        sum_dist_per_class[c] = nn_distances[mask].sum()

    min_sum = min(sum_dist_per_class.values())
    closest_tied = [c for c, s in sum_dist_per_class.items() if s == min_sum]

    # tie-break (b): lowest class index, if still tied after (a)
    return int(min(closest_tied))


def knn_predict(X_train, y_train, X_test, k=5):
    """
    Predict classes for an entire test set.

    Parameters
    ----------
    X_train : array-like, shape (n_train, n_features)
    y_train : array-like, shape (n_train,)   -- integer class labels 0..C-1
    X_test  : array-like, shape (n_test, n_features)
    k       : int, number of neighbours to use

    Returns
    -------
    np.ndarray, shape (n_test,) -- predicted integer class labels
    """
    X_test = np.asarray(X_test, dtype=float)
    return np.array(
        [knn_predict_point(X_train, y_train, x, k=k) for x in X_test]
    )