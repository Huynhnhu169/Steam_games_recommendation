"""Generate demo recommendations from trained or full-data models."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import joblib

from .data import load_config, load_raw_data, resolve_path
from .models import ItemKNNRecommender, PopularityRecommender, fit_models
from .preprocess import STANDARD_ITEM, STANDARD_TITLE, prepare_interactions


def _model_artifact_path(config: dict) -> Path:
    return resolve_path(config["paths"].get("model_dir", "models"), config.get("_project_root")) / "recommenders.joblib"


def load_or_fit_models(config: dict, require_item_knn: bool = False) -> tuple[dict, object, str]:
    artifact_path = _model_artifact_path(config)
    if artifact_path.exists():
        artifact = joblib.load(artifact_path)
        return artifact["models"], artifact["catalog"], "trained_artifact"

    reviews, games = load_raw_data(config)
    interactions, catalog = prepare_interactions(reviews, games, config)
    models = [PopularityRecommender(), ItemKNNRecommender(config.get("models", {}).get("item_knn", {}).get("k_neighbors", 30))]
    fitted = fit_models(models, interactions)
    source = "full_data_demo_model"
    if require_item_knn and "item_knn" not in fitted:
        raise RuntimeError("ItemKNN model could not be created.")
    return fitted, catalog, source


def _coerce_user_id(raw_user_id: str, known_user_ids: list):
    if not known_user_ids:
        return raw_user_id
    sample = known_user_ids[0]
    if isinstance(sample, int):
        return int(raw_user_id)
    try:
        import numpy as np

        if isinstance(sample, np.integer):
            return int(raw_user_id)
    except Exception:
        pass
    return raw_user_id


def _title_map(catalog) -> dict:
    return dict(zip(catalog[STANDARD_ITEM], catalog[STANDARD_TITLE]))


def _format_recommendations(recommendations, catalog) -> list[dict]:
    titles = _title_map(catalog)
    output = []
    for rec in recommendations:
        item = rec.as_dict()
        item["title"] = str(titles.get(rec.item_id, ""))
        output.append(item)
    return output


def find_item_by_title(title: str, catalog):
    titles = catalog[[STANDARD_ITEM, STANDARD_TITLE]].dropna().copy()
    titles["_norm"] = titles[STANDARD_TITLE].astype(str).str.lower()
    query = title.strip().lower()

    exact = titles[titles["_norm"] == query]
    if not exact.empty:
        row = exact.iloc[0]
        return row[STANDARD_ITEM], row[STANDARD_TITLE]

    contains = titles[titles["_norm"].str.contains(query, regex=False)]
    if not contains.empty:
        row = contains.iloc[0]
        return row[STANDARD_ITEM], row[STANDARD_TITLE]

    choices = titles[STANDARD_TITLE].astype(str).tolist()
    matches = difflib.get_close_matches(title, choices, n=1, cutoff=0.4)
    if matches:
        row = titles[titles[STANDARD_TITLE].astype(str) == matches[0]].iloc[0]
        return row[STANDARD_ITEM], row[STANDARD_TITLE]

    raise ValueError(f"Could not find a game title similar to: {title}")


def recommend(config: dict, user_id: str | None, game: str | None, top_k: int) -> dict:
    require_item_knn = game is not None
    models, catalog, source = load_or_fit_models(config, require_item_knn=require_item_knn)

    if game:
        model = models.get("item_knn")
        if model is None:
            raise RuntimeError("Game-title recommendation requires the item_knn model.")
        item_id, matched_title = find_item_by_title(game, catalog)
        recommendations = model.recommend_similar_items(item_id, top_k=top_k)
        return {
            "mode": "similar_games",
            "model_source": source,
            "query": game,
            "matched_title": str(matched_title),
            "recommendations": _format_recommendations(recommendations, catalog),
        }

    if user_id is None:
        raise ValueError("Provide either --game or --user-id.")

    model = models.get("item_knn") or models.get("popularity")
    parsed_user_id = _coerce_user_id(user_id, getattr(model, "user_ids", []))
    recommendations = model.recommend_for_user(parsed_user_id, top_k=top_k)
    return {
        "mode": "user_recommendations",
        "model": getattr(model, "name", "unknown"),
        "model_source": source,
        "user_id": parsed_user_id,
        "recommendations": _format_recommendations(recommendations, catalog),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend Steam games.")
    parser.add_argument("--config", default="configs/recommender.yaml")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--game", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    result = recommend(load_config(args.config), user_id=args.user_id, game=args.game, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
