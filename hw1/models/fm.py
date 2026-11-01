"""Факторизационная машина второго порядка с метапризнаками товаров.

Вход: бинарная матрица положительных взаимодействий X, матрица всех известных
взаимодействий seen_matrix и целочисленные категориальные item_features.
Строки item_features соответствуют столбцам X; каждое поле имеет свои коды.
Модель: смещение, линейные признаки и факторизованные попарные взаимодействия
user ID, item ID и хотя бы одного изменяющегося метапризнака товара.
Обучение: бинарная cross-entropy, PyTorch autograd и Adam; ненаблюдаемые
обучающие примеры пересэмплируются на каждой эпохе с исключением seen-пар.
История и преобразования признаков должны соответствовать границе обучения;
историческую доступность метаданных необходимо обосновать в ноутбуке.
Рекомендации: top-k по score с исключением всех ранее оценённых товаров;
при равных score выбирается меньший индекс товара.
"""

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .common import binary_csr, sample_unseen, topk_indices


class _FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, factors)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, features):
        vectors = self.embedding(features)
        interactions = 0.5 * (
            vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        linear = self.linear(features).sum(dim=1).squeeze(-1)
        return self.bias + linear + interactions


class FactorizationMachine:
    def __init__(self, factors=32, lr=0.01, reg=1e-5, epochs=10,
                 batch_size=4096, negative_samples=1, seed=42, device="cpu"):
        if factors < 1 or lr <= 0 or reg < 0 or epochs < 1 or batch_size < 1:
            raise ValueError("Invalid training parameters")
        if not isinstance(negative_samples, int) or negative_samples < 0:
            raise ValueError("negative_samples must be a nonnegative integer")
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.batch_size = epochs, batch_size
        self.negative_samples, self.seed = negative_samples, seed
        self.device = torch.device(device)

    def _features(self, users, items):
        users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        return torch.cat([
            users[:, None],
            (items + self.n_users_)[:, None],
            self.metadata_[items],
        ], dim=1)

    def fit(self, X, item_features, seen_matrix=None):
        X = binary_csr(X)
        seen = binary_csr(X if seen_matrix is None else seen_matrix)
        if seen.shape != X.shape or (X - X.multiply(seen)).nnz:
            raise ValueError("seen_matrix must have X's shape and include all positives")

        metadata = np.asarray(item_features)
        if (metadata.ndim != 2 or metadata.shape[0] != X.shape[1]
                or metadata.shape[1] < 1 or not metadata.shape[0]):
            raise ValueError("Expected one metadata row per item and at least one field")
        if not np.issubdtype(metadata.dtype, np.integer) or (metadata < 0).any():
            raise ValueError("Metadata must contain nonnegative integer category codes")
        if not any(np.unique(metadata[:, j]).size > 1 for j in range(metadata.shape[1])):
            raise ValueError("At least one metadata field must vary across items")
        if not seen.nnz:
            raise ValueError("Training interactions are empty")

        self.n_users_, self.n_items_ = X.shape
        sizes = metadata.max(axis=0).astype(np.int64) + 1
        offsets = self.n_users_ + self.n_items_ + np.r_[0, np.cumsum(sizes[:-1])]
        self.metadata_ = torch.as_tensor(
            metadata + offsets, dtype=torch.long, device=self.device
        )
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.network_ = _FMNetwork(
            int(self.n_users_ + self.n_items_ + sizes.sum()), self.factors
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.lr, weight_decay=self.reg
        )
        criterion = nn.BCEWithLogitsLoss()
        observed, positive = seen.tocoo(), X.tocoo()
        observed_labels = np.asarray(X[observed.row, observed.col]).ravel()
        self.loss_history_ = []

        for _ in tqdm(range(self.epochs), desc="FM: эпохи", position=1, leave=False):
            negative_users, negative_items = sample_unseen(
                np.repeat(positive.row, self.negative_samples), seen, rng
            )
            users = np.concatenate([observed.row, negative_users])
            items = np.concatenate([observed.col, negative_items])
            labels = np.concatenate([
                observed_labels, np.zeros(len(negative_users), dtype=np.float32)
            ])
            order = rng.permutation(len(users))
            epoch_loss = 0.0
            self.network_.train()

            batches = tqdm(
                range(0, len(order), self.batch_size),
                desc="FM: батчи", position=2, leave=False,
            )
            for start in batches:
                batch = order[start:start + self.batch_size]
                features = self._features(users[batch], items[batch])
                target = torch.as_tensor(
                    labels[batch], dtype=torch.float32, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.network_(features), target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
                batches.set_postfix(loss=loss.item(), refresh=False)

            self.loss_history_.append(epoch_loss / len(order))

        self._cache_predictions()
        return self

    def _cache_predictions(self):
        # Exact FM decomposition for static item metadata:
        # score(u,i) = user_bias[u] + item_bias[i] + user_vector[u] @ item_vector[i].
        # This avoids rebuilding feature combinations for the entire catalog per user.
        self.network_.eval()
        vectors, biases = [], []
        with torch.no_grad():
            self.user_vectors_ = (
                self.network_.embedding.weight[:self.n_users_].cpu().numpy().copy()
            )
            self.user_bias_ = (
                self.network_.linear.weight[:self.n_users_, 0].cpu().numpy().copy()
            )
            for start in range(0, self.n_items_, self.batch_size):
                items = torch.arange(
                    start, min(start + self.batch_size, self.n_items_),
                    dtype=torch.long, device=self.device,
                )
                features = torch.cat([
                    (items + self.n_users_)[:, None], self.metadata_[items]
                ], dim=1)
                latent = self.network_.embedding(features)
                summed = latent.sum(dim=1)
                interaction = 0.5 * (
                    summed.square() - latent.square().sum(dim=1)
                ).sum(dim=1)
                bias = (
                    self.network_.bias
                    + self.network_.linear(features).sum(dim=1).squeeze(-1)
                    + interaction
                )
                vectors.append(summed.cpu().numpy())
                biases.append(bias.cpu().numpy())
        self.item_vectors_ = np.concatenate(vectors)
        self.item_bias_ = np.concatenate(biases)

    def recommend(self, user, seen, k=10):
        scores = (
            self.item_vectors_ @ self.user_vectors_[user]
            + self.item_bias_ + self.user_bias_[user]
        )
        return topk_indices(scores, seen, k)
