"""Train leakage-safe recommender models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import load_config, load_raw_data, output_paths
from .models import build_models, fit_models
from .preprocess import prepare_interactions, split_interactions_by_user


def build_train_test(config: dict):
    reviews, games = load_raw_data(config)
    interactions, catalog = prepare_interactions(reviews, games, config)
    data_cfg = config["data"]
    train, test = split_interactions_by_user(
        interactions,
        test_size=float(data_cfg.get("test_size", 0.2)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )
    return interactions, catalog, train, test


def train(config: dict) -> dict:
    output_dir, model_dir = output_paths(config)
    interactions, catalog, train_interactions, test_interactions = build_train_test(config)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_interactions.to_csv(splits_dir / "train_interactions.csv", index=False)
    test_interactions.to_csv(splits_dir / "test_interactions.csv", index=False)

    fitted_models = fit_models(build_models(config), train_interactions)
    artifact_path = model_dir / "recommenders.joblib"
    joblib.dump(
        {
            "models": fitted_models,
            "catalog": catalog,
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        },
        artifact_path,
    )

    summary = {
        "total_positive_interactions": int(len(interactions)),
        "train_interactions": int(len(train_interactions)),
        "test_interactions": int(len(test_interactions)),
        "users": int(interactions["user_id"].nunique()),
        "items": int(interactions["item_id"].nunique()),
        "models": sorted(fitted_models.keys()),
        "artifact": str(artifact_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Steam game recommender models.")
    parser.add_argument("--config", default="configs/recommender.yaml")
    args = parser.parse_args()

    summary = train(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

