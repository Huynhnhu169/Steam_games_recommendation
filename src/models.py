"""Recommender models fit only on training interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from .preprocess import STANDARD_ITEM, STANDARD_USER, STANDARD_VALUE


def _to_python_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class Recommendation:
    item_id: object
    score: float

    def as_dict(self) -> dict:
        return {"item_id": _to_python_value(self.item_id), "score": float(self.score)}


class MatrixMixin:
    user_ids: list
    item_ids: list
    user_to_index: dict
    item_to_index: dict
    matrix: csr_matrix
    item_popularity: np.ndarray

    def _build_matrix(self, interactions: pd.DataFrame) -> None:
        self.user_ids = list(pd.unique(interactions[STANDARD_USER]))
        self.item_ids = list(pd.unique(interactions[STANDARD_ITEM]))
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        rows = interactions[STANDARD_USER].map(self.user_to_index).to_numpy()
        cols = interactions[STANDARD_ITEM].map(self.item_to_index).to_numpy()
        values = interactions[STANDARD_VALUE].astype(float).to_numpy()
        self.matrix = csr_matrix((values, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))
        self.item_popularity = np.asarray(self.matrix.sum(axis=0)).ravel()

    def _seen_indices(self, user_id) -> set[int]:
        if user_id not in self.user_to_index:
            return set()
        row = self.matrix[self.user_to_index[user_id]]
        return set(row.indices.tolist())

    def _recommend_from_scores(
        self,
        scores: np.ndarray,
        user_id=None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        scores = np.asarray(scores, dtype=float).copy()
        scores[~np.isfinite(scores)] = -np.inf
        if exclude_seen and user_id is not None:
            for idx in self._seen_indices(user_id):
                scores[idx] = -np.inf

        recommendations: list[Recommendation] = []
        for idx in np.argsort(-scores):
            if len(recommendations) >= top_k:
                break
            score = scores[idx]
            if not np.isfinite(score) or score <= 0:
                continue
            recommendations.append(Recommendation(self.item_ids[int(idx)], float(score)))
        return recommendations


class PopularityRecommender(MatrixMixin):
    name = "popularity"

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        self._build_matrix(interactions)
        return self

    def recommend_for_user(self, user_id, top_k: int = 10) -> list[Recommendation]:
        return self._recommend_from_scores(self.item_popularity, user_id=user_id, top_k=top_k)


class ItemKNNRecommender(MatrixMixin):
    name = "item_knn"

    def __init__(self, k_neighbors: int = 30):
        self.k_neighbors = int(k_neighbors)

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        self._build_matrix(interactions)
        item_user_matrix = self.matrix.T
        similarity = cosine_similarity(item_user_matrix, dense_output=True)
        np.fill_diagonal(similarity, 0.0)
        self.item_similarity = self._keep_top_k(similarity, self.k_neighbors)
        return self

    @staticmethod
    def _keep_top_k(similarity: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or k >= similarity.shape[1]:
            return similarity
        filtered = np.zeros_like(similarity)
        for row_idx, row in enumerate(similarity):
            top_idx = np.argpartition(row, -k)[-k:]
            filtered[row_idx, top_idx] = row[top_idx]
        return filtered

    def recommend_for_user(self, user_id, top_k: int = 10) -> list[Recommendation]:
        if user_id not in self.user_to_index:
            return self._recommend_from_scores(self.item_popularity, user_id=None, top_k=top_k)
        user_vector = self.matrix[self.user_to_index[user_id]].toarray().ravel()
        scores = user_vector @ self.item_similarity
        return self._recommend_from_scores(scores, user_id=user_id, top_k=top_k)

    def recommend_similar_items(self, item_id, top_k: int = 10) -> list[Recommendation]:
        if item_id not in self.item_to_index:
            raise KeyError(f"Unknown item_id: {item_id}")
        item_idx = self.item_to_index[item_id]
        return self._recommend_from_scores(self.item_similarity[item_idx], top_k=top_k, exclude_seen=False)


class UserKNNRecommender(MatrixMixin):
    name = "user_knn"

    def __init__(self, k_neighbors: int = 30):
        self.k_neighbors = int(k_neighbors)

    def fit(self, interactions: pd.DataFrame) -> "UserKNNRecommender":
        self._build_matrix(interactions)
        n_neighbors = min(self.k_neighbors + 1, max(1, len(self.user_ids)))
        self.neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        self.neighbor_model.fit(self.matrix)
        return self

    def recommend_for_user(self, user_id, top_k: int = 10) -> list[Recommendation]:
        if user_id not in self.user_to_index:
            return self._recommend_from_scores(self.item_popularity, user_id=None, top_k=top_k)

        user_idx = self.user_to_index[user_id]
        distances, indices = self.neighbor_model.kneighbors(self.matrix[user_idx], return_distance=True)
        distances = distances.ravel()
        indices = indices.ravel()
        keep = indices != user_idx
        indices = indices[keep]
        weights = 1.0 - distances[keep]
        weights = np.maximum(weights, 0.0)
        if len(indices) == 0 or weights.sum() <= 0:
            scores = self.item_popularity
        else:
            scores = weights @ self.matrix[indices].toarray()
        return self._recommend_from_scores(scores, user_id=user_id, top_k=top_k)


def build_models(config: dict) -> list[MatrixMixin]:
    models: list[MatrixMixin] = [PopularityRecommender()]
    model_cfg = config.get("models", {})
    item_cfg = model_cfg.get("item_knn", {})
    user_cfg = model_cfg.get("user_knn", {})
    if item_cfg.get("enabled", True):
        models.append(ItemKNNRecommender(k_neighbors=item_cfg.get("k_neighbors", 30)))
    if user_cfg.get("enabled", True):
        models.append(UserKNNRecommender(k_neighbors=user_cfg.get("k_neighbors", 30)))
    return models


def fit_models(models: Iterable[MatrixMixin], train_interactions: pd.DataFrame) -> dict[str, MatrixMixin]:
    fitted = {}
    for model in models:
        fitted[model.name] = model.fit(train_interactions)
    return fitted

