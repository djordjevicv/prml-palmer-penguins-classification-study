"""
distance.py
-----------
Single, shared distance utility used identically by the baseline (classic
kNN) and the extension (weighted kNN) methods, and by the cross-validation
harness in cross_validation.py.

Exposes:
    euclidean_distance_matrix(X1, X2)  -- vectorised pairwise Euclidean distance
"""

import numpy as np


def euclidean_distance_matrix(X1, X2):
    """
    Vectorised pairwise Euclidean distance between two sets of points.

    d(X1_i, X2_j) = sqrt( sum_f (X1_i,f - X2_j,f)^2 )

    Computed via the expansion
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    using a single matrix multiplication, so the entire pairwise distance
    matrix is produced without any Python-level loop over samples. This
    is the only distance routine used by knn_predict (majority vote) and
    weighted_knn_predict (inverse-distance vote) -- both call this function
    rather than each maintaining its own loop-based implementation.

    Parameters
    ----------
    X1 : array-like, shape (n1, n_features)
    X2 : array-like, shape (n2, n_features)

    Returns
    -------
    np.ndarray, shape (n1, n2)
        dist[i, j] = Euclidean distance between X1[i] and X2[j]
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    sq1 = np.sum(X1 ** 2, axis=1)[:, np.newaxis]   # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1)[np.newaxis, :]   # (1, n2)
    cross_term = X1 @ X2.T                          # (n1, n2)

    sq_dists = sq1 + sq2 - 2 * cross_term
    # floating point error can push values just below zero on the diagonal
    # (or when X1 is X2); clip before taking the square root.
    sq_dists = np.maximum(sq_dists, 0.0)

    return np.sqrt(sq_dists)


if __name__ == "__main__":
    # Quick sanity check against a hand-computed example -- run this file
    # directly (`python distance.py`) to verify the vectorised result
    # matches what you'd get computing each distance by hand.
    #
    #   X1 = [(0,0), (3,4)]
    #   X2 = [(0,0), (3,0), (0,4)]
    #
    #   (0,0) to (0,0) -> 0
    #   (0,0) to (3,0) -> 3
    #   (0,0) to (0,4) -> 4
    #   (3,4) to (0,0) -> sqrt(3^2 + 4^2) = 5
    #   (3,4) to (3,0) -> sqrt(0^2 + 4^2) = 4
    #   (3,4) to (0,4) -> sqrt(3^2 + 0^2) = 3
    X1 = np.array([[0.0, 0.0], [3.0, 4.0]])
    X2 = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    expected = np.array([[0.0, 3.0, 4.0],
                          [5.0, 4.0, 3.0]])

    result = euclidean_distance_matrix(X1, X2)
    print("euclidean_distance_matrix output:\n", result)
    print("hand-computed expected:\n", expected)

    assert np.allclose(result, expected), "vectorised distance does not match hand-computed example!"
    print("\nOK -- matches hand-computed example.")