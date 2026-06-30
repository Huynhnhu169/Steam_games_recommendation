"""Top-K ranking metrics for implicit recommendation."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .preprocess import STANDARD_ITEM, STANDARD_USER


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    top_items = recommended[:k]
    hits = sum(1 for item in top_items if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_items = recommended[:k]
    hits = sum(1 for item in top_items if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def relevant_by_user(test_interactions: pd.DataFrame) -> dict:
    grouped = defaultdict(set)
    for row in test_interactions[[STANDARD_USER, STANDARD_ITEM]].itertuples(index=False):
        grouped[getattr(row, STANDARD_USER)].add(getattr(row, STANDARD_ITEM))
    return dict(grouped)


def evaluate_model(model, test_interactions: pd.DataFrame, top_k: int = 10) -> dict:
    per_user = relevant_by_user(test_interactions)
    precision_values = []
    recall_values = []
    ndcg_values = []
    ap_values = []
    recommended_items = set()

    for user_id, relevant in per_user.items():
        recommendations = model.recommend_for_user(user_id, top_k=top_k)
        rec_items = [rec.item_id for rec in recommendations]
        recommended_items.update(rec_items)
        precision_values.append(precision_at_k(rec_items, relevant, top_k))
        recall_values.append(recall_at_k(rec_items, relevant, top_k))
        ndcg_values.append(ndcg_at_k(rec_items, relevant, top_k))
        ap_values.append(average_precision_at_k(rec_items, relevant, top_k))

    catalog_size = max(1, len(getattr(model, "item_ids", [])))
    return {
        "precision_at_k": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_at_k": float(np.mean(recall_values)) if recall_values else 0.0,
        "ndcg_at_k": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "map_at_k": float(np.mean(ap_values)) if ap_values else 0.0,
        "coverage": float(len(recommended_items) / catalog_size),
        "evaluated_users": int(len(per_user)),
        "catalog_items": int(catalog_size),
    }
