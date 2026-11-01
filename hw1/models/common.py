"""Общие функции моделей: бинарные CSR-матрицы, top-k и семплирование.

Исключение известных товаров и разрешение равных score используются одинаково:
товары из seen исключаются, при равных score выбирается меньший индекс.
Семплирование ненаблюдаемых пар использует только обучающую историю.
"""

import numpy as np
from scipy.sparse import csr_matrix


def binary_csr(matrix):
    matrix = csr_matrix(matrix, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Expected finite, nonnegative interactions")
    matrix.data.fill(1)
    matrix.sort_indices()
    return matrix


def topk_indices(scores, seen, k=10):
    """Exclude seen items; break score ties by ascending item index."""
    scores = np.asarray(scores).ravel().copy()
    scores[np.asarray(list(seen), dtype=np.int64)] = -np.inf
    candidates = np.flatnonzero(np.isfinite(scores))
    count = min(k, len(candidates))
    if count <= 0:
        return candidates[:0]

    values = scores[candidates]
    threshold = np.partition(values, len(values) - count)[len(values) - count]
    above = candidates[values > threshold]
    tied = candidates[values == threshold]
    selected = np.concatenate([above, tied[:count - len(above)]])
    return selected[np.lexsort((selected, -scores[selected]))]


def sample_unseen(users, seen, rng):
    """Sample one uniformly drawn unseen item for each eligible user occurrence."""
    users = np.asarray(users, dtype=np.int64)
    n_items = seen.shape[1]
    users = users[np.diff(seen.indptr)[users] < n_items]
    if not len(users):
        return users, np.empty(0, dtype=np.int64)

    items = rng.integers(n_items, size=len(users))
    for _ in range(20):
        invalid = np.asarray(seen[users, items]).ravel() > 0
        if not invalid.any():
            return users, items
        items[invalid] = rng.integers(n_items, size=invalid.sum())

    invalid = np.asarray(seen[users, items]).ravel() > 0
    catalog = np.arange(n_items)
    for user in np.unique(users[invalid]):
        positions = np.flatnonzero(invalid & (users == user))
        available = np.setdiff1d(catalog, seen[user].indices, assume_unique=True)
        items[positions] = rng.choice(available, size=len(positions))
    return users, items
