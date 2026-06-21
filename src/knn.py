"""
knn.py
------
Both kNN methods compared in this project: majority-vote (baseline) and
inverse-distance weighted (extension). NumPy only.

    knn_predict_point / knn_predict
    weighted_knn_predict_point / weighted_knn_predict
"""

import numpy as np

from distance import euclidean_distance_matrix


def knn_predict_point(X_train, y_train, x_query, k=5, distances=None):
    """
    Majority-vote prediction for a single query point.

    Tie-break if two+ classes get the same vote count: smallest summed
    distance among the tied class's neighbours wins; if still tied, lowest
    class index wins.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    x_query = np.asarray(x_query, dtype=float)

    if distances is None:
        distances = euclidean_distance_matrix(x_query.reshape(1, -1), X_train)[0]

    nn_idx = np.argsort(distances)[:k]
    nn_labels = y_train[nn_idx]
    nn_distances = distances[nn_idx]

    counts = np.bincount(nn_labels)
    max_votes = counts.max()
    tied_classes = np.flatnonzero(counts == max_votes)

    if tied_classes.size == 1:
        return int(tied_classes[0])

    sum_dist_per_class = {
        c: nn_distances[nn_labels == c].sum() for c in tied_classes
    }
    min_sum = min(sum_dist_per_class.values())
    closest_tied = [c for c, s in sum_dist_per_class.items() if s == min_sum]

    return int(min(closest_tied))


def weighted_knn_predict_point(X_train, y_train, x_query, k=5, distances=None):
    """
    Inverse-distance weighted prediction for a single query point. Each
    neighbour votes with weight 1/d; the class with the highest summed
    weight wins.

    Zero-distance handling: if x_query exactly matches one or more of the
    k nearest training points, 1/d is undefined. In that case only the
    zero-distance neighbour(s) vote, each with equal full weight, and the
    rest of the k neighbours are ignored -- an exact match shouldn't be
    diluted by farther points.

    Tie-break on summed weight: lowest class index wins (rare in practice
    since weights are continuous, but kept for determinism).
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    x_query = np.asarray(x_query, dtype=float)

    if distances is None:
        distances = euclidean_distance_matrix(x_query.reshape(1, -1), X_train)[0]

    nn_idx = np.argsort(distances)[:k]
    nn_labels = y_train[nn_idx]
    nn_distances = distances[nn_idx]

    zero_mask = nn_distances == 0.0
    if np.any(zero_mask):
        voting_labels = nn_labels[zero_mask]
        weights = np.ones_like(voting_labels, dtype=float)
    else:
        voting_labels = nn_labels
        weights = 1.0 / nn_distances

    n_classes = int(y_train.max()) + 1
    weight_per_class = np.zeros(n_classes)
    for label, w in zip(voting_labels, weights):
        weight_per_class[label] += w

    max_weight = weight_per_class.max()
    tied_classes = np.flatnonzero(weight_per_class == max_weight)

    return int(tied_classes.min())


def _predict_batch(predict_point_fn, X_train, y_train, X_test, k):
    """Compute the X_test/X_train distance matrix once, then call
    predict_point_fn (knn_predict_point or weighted_knn_predict_point)
    per row."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    dist_matrix = euclidean_distance_matrix(X_test, X_train)

    return np.array([
        predict_point_fn(X_train, y_train, X_test[i], k=k, distances=dist_matrix[i])
        for i in range(X_test.shape[0])
    ])


def knn_predict(X_train, y_train, X_test, k=5):
    """Majority-vote kNN over a whole test set. Returns predicted labels,
    shape (n_test,)."""
    return _predict_batch(knn_predict_point, X_train, y_train, X_test, k)


def weighted_knn_predict(X_train, y_train, X_test, k=5):
    """Inverse-distance weighted kNN over a whole test set. Same signature
    as knn_predict, so cross_validate(..., method=weighted_knn_predict)
    works directly."""
    return _predict_batch(weighted_knn_predict_point, X_train, y_train, X_test, k)


if __name__ == "__main__":
    X_train = np.array([
        [1.0, 0.0],   # class 0
        [0.0, 1.0],   # class 0
        [0.1, 0.0],   # class 1, much closer to the query below
    ])
    y_train = np.array([0, 0, 1])
    query = np.array([0.0, 0.0])

    print("majority vote:", knn_predict_point(X_train, y_train, query, k=3))
    print("weighted vote:", weighted_knn_predict_point(X_train, y_train, query, k=3))

    exact_match_query = np.array([1.0, 0.0])  # == X_train[0], class 0
    print("exact match:", weighted_knn_predict_point(X_train, y_train, exact_match_query, k=3))