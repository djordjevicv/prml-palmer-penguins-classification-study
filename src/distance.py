"""
distance.py
-----------
Shared distance utility, used by both kNN methods and by the
cross-validation harness.

    euclidean_distance_matrix(X1, X2)
"""

import numpy as np


def euclidean_distance_matrix(X1, X2):
    """
    Pairwise Euclidean distance between two sets of points, vectorised.

    dist[i, j] = ||X1[i] - X2[j]||

    Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, computed with one matrix
    multiplication instead of looping over rows.

    X1 : array-like, shape (n1, n_features)
    X2 : array-like, shape (n2, n_features)
    returns : np.ndarray, shape (n1, n2)
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    sq1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
    sq2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
    cross_term = X1 @ X2.T

    sq_dists = sq1 + sq2 - 2 * cross_term
    sq_dists = np.maximum(sq_dists, 0.0)  # clip floating-point noise near 0

    return np.sqrt(sq_dists)


if __name__ == "__main__":
    # hand-computed check
    # (0,0)-(0,0)=0  (0,0)-(3,0)=3  (0,0)-(0,4)=4
    # (3,4)-(0,0)=5  (3,4)-(3,0)=4  (3,4)-(0,4)=3
    X1 = np.array([[0.0, 0.0], [3.0, 4.0]])
    X2 = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    expected = np.array([[0.0, 3.0, 4.0],
                          [5.0, 4.0, 3.0]])

    result = euclidean_distance_matrix(X1, X2)
    print(result)
    assert np.allclose(result, expected)