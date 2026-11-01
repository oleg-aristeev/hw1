"""SLIM с неотрицательными весами и собственным координатным спуском.

Вход: бинарная матрица положительных взаимодействий «пользователи × товары».
Модель: score = X @ W, где W >= 0 и diag(W) = 0.
Обучение: минимум .5 ||X-XW||_F^2 + l1 ||W||_1 + .5 l2 ||W||_F^2;
целевая функция не делится на число пользователей. По умолчанию
max_candidates=None: используется только безопасное исключение координат.
Явное ограничение max_candidates задаёт приближённый вариант модели.
Рекомендации: top-k по score с исключением всех ранее оценённых товаров;
при равных score выбирается меньший индекс товара.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from .common import binary_csr, topk_indices


class SLIM:
    def __init__(self, l1=1.0, l2=1.0, max_iter=30, tol=1e-4,
                 block_size=128, max_candidates=None):
        if l1 < 0 or l2 <= 0 or max_iter < 1 or tol < 0 or block_size < 1:
            raise ValueError("Invalid regularization or optimizer parameters")
        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be positive or None")
        self.l1, self.l2 = l1, l2
        self.max_iter, self.tol = max_iter, tol
        self.block_size = block_size
        self.max_candidates = max_candidates

    def _solve(self, X, item, candidates, norms):
        residual = np.zeros(X.shape[0])
        a, b = X.indptr[item:item + 2]
        residual[X.indices[a:b]] = X.data[a:b]
        weights = np.zeros(len(candidates))

        for iteration in range(1, self.max_iter + 1):
            largest_change = 0.0
            for coordinate, source in enumerate(candidates):
                a, b = X.indptr[source:source + 2]
                users, values = X.indices[a:b], X.data[a:b]
                old = weights[coordinate]
                rho = values @ residual[users] + norms[source] * old
                new = max(0.0, (rho - self.l1) / (norms[source] + self.l2))
                change = new - old
                if change:
                    residual[users] -= values * change
                    weights[coordinate] = new
                    largest_change = max(largest_change, abs(change))

            if largest_change <= self.tol * (1 + weights.max(initial=0)):
                return weights, iteration, True

        return weights, self.max_iter, False

    def fit(self, X):
        self.X = binary_csr(X)
        columns = self.X.astype(np.float64).tocsc()
        n_items = X.shape[1]
        norms = np.asarray(columns.power(2).sum(axis=0)).ravel()
        self.n_iter_ = np.zeros(n_items, dtype=np.int32)
        self.converged_ = np.ones(n_items, dtype=bool)
        rows, cols, values = [], [], []

        with tqdm(total=n_items, desc="SLIM: товары", position=1, leave=False) as bar:
            for start in range(0, n_items, self.block_size):
                stop = min(start + self.block_size, n_items)
                cooccurrence = (columns[:, start:stop].T @ columns).tocsr()

                for local, item in enumerate(range(start, stop)):
                    a, b = cooccurrence.indptr[local:local + 2]
                    indices = cooccurrence.indices[a:b]
                    counts = cooccurrence.data[a:b]
                    # For X,W >= 0, X_j.T X_i <= l1 implies optimal W_ji=0.
                    keep = (indices != item) & (counts > self.l1)
                    indices, counts = indices[keep], counts[keep]
                    order = np.lexsort((indices, -counts))
                    if self.max_candidates is not None:
                        order = order[:self.max_candidates]
                    candidates = indices[order]

                    if len(candidates):
                        weights, iterations, converged = self._solve(
                            columns, item, candidates, norms
                        )
                        self.n_iter_[item] = iterations
                        self.converged_[item] = converged
                        nonzero = weights > 0
                        if nonzero.any():
                            rows.append(candidates[nonzero])
                            cols.append(np.full(nonzero.sum(), item, dtype=np.int32))
                            values.append(weights[nonzero].astype(np.float32))
                    bar.update(1)

        self.W = csr_matrix((n_items, n_items), dtype=np.float32)
        if values:
            self.W = csr_matrix(
                (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n_items, n_items),
            )
        return self

    def recommend(self, user, seen, k=10):
        scores = (self.X[user] @ self.W).toarray().ravel()
        return topk_indices(scores, seen, k)
