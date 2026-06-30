"""Preprocessing and leakage-safe interaction splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


STANDARD_USER = "user_id"
STANDARD_ITEM = "item_id"
STANDARD_VALUE = "value"
STANDARD_TITLE = "title"


def _positive_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0) > 0
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "recommended", "positive"})


def prepare_catalog(games: pd.DataFrame, config: dict) -> pd.DataFrame:
    columns = config["columns"]
    item_col = columns.get("item_id", "app_id")
    title_col = columns.get("title", "title")
    popularity_col = columns.get("popularity", "user_reviews")

    required = {item_col, title_col}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Games file is missing required columns: {sorted(missing)}")

    keep = [item_col, title_col]
    if popularity_col in games.columns:
        keep.append(popularity_col)

    catalog = games[keep].copy()
    catalog = catalog.rename(columns={item_col: STANDARD_ITEM, title_col: STANDARD_TITLE})
    if popularity_col in catalog.columns:
        catalog = catalog.rename(columns={popularity_col: "popularity"})
    return catalog.drop_duplicates(subset=[STANDARD_ITEM])


def prepare_interactions(
    reviews: pd.DataFrame,
    games: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return positive implicit interactions and a matching game catalog."""
    columns = config["columns"]
    data_cfg = config["data"]
    user_col = columns.get("user_id", "user_id")
    item_col = columns.get("item_id", "app_id")
    feedback_col = columns.get("feedback", "is_recommended")

    required = {user_col, item_col, feedback_col}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Reviews file is missing required columns: {sorted(missing)}")

    catalog = prepare_catalog(games, config)
    top_n = data_cfg.get("top_n_games_by_reviews")
    if top_n and "popularity" in catalog.columns:
        catalog["popularity"] = pd.to_numeric(catalog["popularity"], errors="coerce").fillna(0)
        top_items = (
            catalog.sort_values("popularity", ascending=False)
            .head(int(top_n))[STANDARD_ITEM]
            .tolist()
        )
        catalog = catalog[catalog[STANDARD_ITEM].isin(top_items)].copy()
        reviews = reviews[reviews[item_col].isin(top_items)].copy()

    interactions = reviews[[user_col, item_col, feedback_col]].copy()
    interactions = interactions.rename(columns={user_col: STANDARD_USER, item_col: STANDARD_ITEM})
    interactions = interactions.dropna(subset=[STANDARD_USER, STANDARD_ITEM, feedback_col])

    if data_cfg.get("implicit_positive_only", True):
        interactions = interactions[_positive_mask(interactions[feedback_col])].copy()
        interactions[STANDARD_VALUE] = 1.0
    else:
        interactions[STANDARD_VALUE] = _positive_mask(interactions[feedback_col]).astype(float)

    interactions = interactions[[STANDARD_USER, STANDARD_ITEM, STANDARD_VALUE]]
    interactions = (
        interactions.groupby([STANDARD_USER, STANDARD_ITEM], as_index=False)[STANDARD_VALUE]
        .max()
        .reset_index(drop=True)
    )

    min_user_interactions = int(data_cfg.get("min_user_interactions", 2))
    if min_user_interactions > 1:
        counts = interactions.groupby(STANDARD_USER)[STANDARD_ITEM].transform("count")
        interactions = interactions[counts >= min_user_interactions].copy()

    catalog = catalog[catalog[STANDARD_ITEM].isin(interactions[STANDARD_ITEM].unique())].copy()
    if interactions.empty:
        raise ValueError("No interactions remain after preprocessing. Check filters and input data.")

    return interactions.reset_index(drop=True), catalog.reset_index(drop=True)


def split_interactions_by_user(
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions before any matrix or similarity is built."""
    rng = np.random.default_rng(random_seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in interactions.groupby(STANDARD_USER, sort=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            train_indices.extend(indices.tolist())
            continue

        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1)
        chosen_test = set(rng.choice(indices, size=n_test, replace=False).tolist())
        for idx in indices:
            if idx in chosen_test:
                test_indices.append(int(idx))
            else:
                train_indices.append(int(idx))

    train = interactions.loc[train_indices].reset_index(drop=True)
    test = interactions.loc[test_indices].reset_index(drop=True)
    overlap = set(map(tuple, train[[STANDARD_USER, STANDARD_ITEM]].to_numpy())).intersection(
        set(map(tuple, test[[STANDARD_USER, STANDARD_ITEM]].to_numpy()))
    )
    if overlap:
        raise RuntimeError("Train/test split contains overlapping user-item interactions.")
    if test.empty:
        raise ValueError("Test split is empty. Increase data size or lower filtering.")
    return train, test
