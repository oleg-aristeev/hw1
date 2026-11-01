"""Implicit ALS с бинарным целевым сигналом.

Вход: бинарная матрица положительных взаимодействий «пользователи × товары».
Модель: score(u, i) = U[u] @ V[i], target = X, confidence = 1 + alpha * X.
Обучение: чередующиеся обновления факторов пакетным решением линейных систем
с L2-регуляризацией reg; начальные факторы определяются seed.
Рекомендации: top-k по score с исключением всех ранее оценённых товаров;
при равных score выбирается меньший индекс товара.
"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class ImplicitALS:
    def __init__(self, factors=32, alpha=40, reg=1.0,
                 iterations=10, seed=42, batch_size=128):
        self.factors = factors
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.seed = seed
        self.batch_size = batch_size

    def _update(self, interactions, other):
        base = other.T @ other + self.reg * np.eye(self.factors)
        result = np.zeros((interactions.shape[0], self.factors))

        for start in range(0, interactions.shape[0], self.batch_size):
            stop = min(start + self.batch_size, interactions.shape[0])
            block = interactions[start:stop]
            active = np.flatnonzero(np.diff(block.indptr))

            if not len(active):
                continue

            vectors = other[block.indices]
            extra = self.alpha * block.data
            offsets = block.indptr[active]

            products = np.einsum(
                "n,ni,nj->nij", extra, vectors, vectors,
                optimize=True,
            )
            systems = base + np.add.reduceat(
                products, offsets, axis=0
            )
            targets = np.add.reduceat(
                (1 + extra)[:, None] * vectors,
                offsets,
                axis=0,
            )

            result[start + active] = np.linalg.solve(
                systems, targets[..., None]
            )[..., 0]

        return result

    def fit(self, X):
        X = csr_matrix(X, dtype=np.float64)
        transposed = X.T.tocsr()
        rng = np.random.default_rng(self.seed)

        self.V = rng.normal(
            scale=0.01, size=(X.shape[1], self.factors)
        )

        for _ in tqdm(
            range(self.iterations),
            desc="Эпохи iALS",
            position=1,
            leave=False,
        ):
            self.U = self._update(X, self.V)
            self.V = self._update(transposed, self.U)

        return self

    def recommend(self, user, seen, k=10):
        scores = self.V @ self.U[user]
        scores[seen] = -np.inf

        candidates = np.flatnonzero(np.isfinite(scores))
        count = min(k, len(candidates))
        if count == 0:
            return candidates[:0]

        values = scores[candidates]
        threshold = np.partition(
            values, len(values) - count
        )[len(values) - count]

        above = candidates[values > threshold]
        tied = candidates[values == threshold]
        selected = np.concatenate([
            above, tied[:count - len(above)]
        ])

        order = np.lexsort((selected, -scores[selected]))
        return selected[order]
