"""Implicit ALS с собственным знаковым нелинейным целевым сигналом.

Вход: разреженная матрица последних обучающих оценок 1–5 с теми же индексами
пользователей и товаров, что у базового iALS.
Модель: score(u, i) = U[u] @ V[i]; target наблюдаемой пары равен
sign(r-3) * (abs(r-3)/2)**target_power, ненаблюдаемой пары — нулю.
При target_power=2 оценки 1–5 преобразуются в [-1, -.25, 0, .25, 1].
Обучение: пакетный ALS с тем же confidence = 1 + alpha * I(r >= 4),
что у базового iALS; изменяется целевой сигнал. Регуляризация — reg.
Рекомендации: top-k по score с исключением всех ранее оценённых товаров;
при равных score выбирается меньший индекс товара.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .ials import ImplicitALS


class CustomImplicitALS(ImplicitALS):
    def __init__(self, factors=32, alpha=40, reg=1.0, iterations=10,
                 seed=42, batch_size=128, target_power=2.0):
        super().__init__(factors, alpha, reg, iterations, seed, batch_size)
        if target_power <= 0:
            raise ValueError("target_power must be positive")
        self.target_power = target_power

    def fit(self, ratings):
        ratings = csr_matrix(ratings, dtype=np.float64, copy=True)
        ratings.sum_duplicates()
        ratings.eliminate_zeros()
        if not np.isin(ratings.data, [1, 2, 3, 4, 5]).all():
            raise ValueError("Expected one rating from 1 to 5 per observed pair")
        return super().fit(ratings)

    def _update(self, ratings, other):
        base = other.T @ other + self.reg * np.eye(self.factors)
        result = np.zeros((ratings.shape[0], self.factors))

        for start in range(0, ratings.shape[0], self.batch_size):
            stop = min(start + self.batch_size, ratings.shape[0])
            block = ratings[start:stop]
            active = np.flatnonzero(np.diff(block.indptr))
            if not len(active):
                continue

            vectors = other[block.indices]
            offsets = block.indptr[active]
            extra = self.alpha * (block.data >= 4)
            centered = block.data - 3
            preference = np.sign(centered) * (np.abs(centered) / 2)**self.target_power

            products = np.einsum(
                "n,ni,nj->nij", extra, vectors, vectors, optimize=True
            )
            systems = base + np.add.reduceat(products, offsets, axis=0)
            targets = np.add.reduceat(
                ((1 + extra) * preference)[:, None] * vectors,
                offsets,
                axis=0,
            )
            result[start + active] = np.linalg.solve(
                systems, targets[..., None]
            )[..., 0]

        return result
