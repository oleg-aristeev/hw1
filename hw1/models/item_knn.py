"""Item-based collaborative filtering с косинусным сходством.

Вход: бинарная матрица положительных взаимодействий «пользователи × товары».
Модель: сумма сходств кандидата с положительно оценёнными товарами пользователя.
Обучение: блочный расчёт сходств и сохранение neighbors соседей каждого товара.
Рекомендации: top-k по score с исключением всех ранее оценённых товаров;
при равных score выбирается меньший индекс товара.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from tqdm.auto import tqdm


class ItemKNN:
    def __init__(self, neighbors=50, block_size=256):
        self.neighbors = neighbors
        self.block_size = block_size

    def fit(self, X):
        self.X = csr_matrix(X, dtype=np.float32)
        n_items = X.shape[1]

        norms = np.sqrt(self.X.power(2).sum(axis=0)).A1
        normalized = (
            self.X @ diags(1 / np.maximum(norms, 1e-12))
        ).tocsc()

        rows, cols, values = [], [], []

        for start in tqdm(
            range(0, n_items, self.block_size),
            desc="Сходства ItemKNN", position=1, leave=False,
        ):
            stop = min(start + self.block_size, n_items)
            block = (normalized[:, start:stop].T @ normalized).tocsr()

            for local, item in enumerate(range(start, stop)):
                a, b = block.indptr[local:local + 2]
                indices = block.indices[a:b]
                weights = block.data[a:b]

                mask = indices != item
                indices, weights = indices[mask], weights[mask]
                order = np.lexsort((indices, -weights))[:self.neighbors]

                if len(order):
                    rows.append(indices[order])
                    cols.append(np.full(len(order), item, dtype=np.int32))
                    values.append(weights[order])

        self.S = csr_matrix((n_items, n_items), dtype=np.float32)
        if values:
            self.S = csr_matrix(
                (
                    np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(cols)),
                ),
                shape=(n_items, n_items),
            )
        return self

    def recommend(self, user, seen, k=10):
        scores = (self.X[user] @ self.S).toarray().ravel()
        scores[seen] = -np.inf

        candidates = np.flatnonzero(np.isfinite(scores))
        order = np.lexsort((candidates, -scores[candidates]))[:k]
        return candidates[order]
