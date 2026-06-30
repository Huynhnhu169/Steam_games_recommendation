"""Configuration and data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent)
    return config


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def _positive_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0) > 0
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "recommended", "positive"})


def dataset_paths(config: dict) -> tuple[Path, Path]:
    paths = config["paths"]
    data_dir = resolve_path(paths.get("data_dir", "data"), config.get("_project_root"))
    reviews_path = data_dir / paths.get("reviews_file", "final_reviews.csv")
    games_path = data_dir / paths.get("games_file", "final_games.csv")
    return reviews_path, games_path


def _review_columns(config: dict) -> list[str]:
    columns = config["columns"]
    return list(
        dict.fromkeys(
            [
                columns.get("user_id", "user_id"),
                columns.get("item_id", "app_id"),
                columns.get("feedback", "is_recommended"),
            ]
        )
    )


def _top_item_filter(games: pd.DataFrame, config: dict) -> set | None:
    data_cfg = config.get("data", {})
    top_n = data_cfg.get("top_n_games_by_reviews")
    if not top_n:
        return None

    columns = config["columns"]
    item_col = columns.get("item_id", "app_id")
    popularity_col = columns.get("popularity", "user_reviews")
    if item_col not in games.columns or popularity_col not in games.columns:
        return None

    ranked_games = games[[item_col, popularity_col]].dropna(subset=[item_col]).copy()
    ranked_games[popularity_col] = pd.to_numeric(ranked_games[popularity_col], errors="coerce").fillna(0)
    top_items = ranked_games.sort_values(popularity_col, ascending=False).head(int(top_n))[item_col]
    return set(top_items.tolist())


def _read_reviews(config: dict, games: pd.DataFrame) -> pd.DataFrame:
    reviews_path, _ = dataset_paths(config)
    data_cfg = config.get("data", {})
    columns = config["columns"]
    item_col = columns.get("item_id", "app_id")
    feedback_col = columns.get("feedback", "is_recommended")
    usecols = _review_columns(config)
    chunk_size = int(data_cfg.get("review_chunksize") or 0)
    max_rows = data_cfg.get("max_review_rows")
    item_filter = _top_item_filter(games, config)

    read_kwargs = {"usecols": usecols}
    if max_rows:
        read_kwargs["nrows"] = int(max_rows)

    if chunk_size <= 0:
        return pd.read_csv(reviews_path, **read_kwargs)

    chunks = []
    for chunk in pd.read_csv(reviews_path, chunksize=chunk_size, **read_kwargs):
        if item_filter is not None:
            chunk = chunk[chunk[item_col].isin(item_filter)]
        if data_cfg.get("implicit_positive_only", True):
            chunk = chunk[_positive_mask(chunk[feedback_col])]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews_path, games_path = dataset_paths(config)
    missing = [str(path) for path in (reviews_path, games_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): "
            + ", ".join(missing)
            + ". Place final_reviews.csv/final_games.csv under data/, or use configs/recommender_colab.yaml "
            + "with Kaggle's games.csv/recommendations.csv files."
        )

    games = pd.read_csv(games_path)
    reviews = _read_reviews(config, games)
    return reviews, games


def output_paths(config: dict) -> tuple[Path, Path]:
    paths = config["paths"]
    output_dir = resolve_path(paths.get("output_dir", "runs/steam_recommender"), config.get("_project_root"))
    model_dir = resolve_path(paths.get("model_dir", "models"), config.get("_project_root"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, model_dir
