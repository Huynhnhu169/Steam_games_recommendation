"""Evaluate recommenders on held-out interactions."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import load_config, output_paths
from .metrics import evaluate_model
from .models import build_models, fit_models
from .train import build_train_test


def evaluate(config: dict) -> dict:
    output_dir, _ = output_paths(config)
    _, _, train_interactions, test_interactions = build_train_test(config)
    fitted_models = fit_models(build_models(config), train_interactions)
    top_k = int(config.get("evaluation", {}).get("top_k", 10))

    results = {}
    rows = []
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, test_interactions, top_k=top_k)
        results[name] = metrics
        rows.append({"model": name, **metrics})

    summary = {
        "top_k": top_k,
        "train_interactions": int(len(train_interactions)),
        "test_interactions": int(len(test_interactions)),
        "models": results,
    }
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / "evaluation.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Steam game recommender models.")
    parser.add_argument("--config", default="configs/recommender.yaml")
    args = parser.parse_args()

    summary = evaluate(load_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

