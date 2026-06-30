# Data

Place the Steam recommendation CSV files here:

```text
data/
  final_reviews.csv
  final_games.csv
```

These files are intentionally ignored by Git.

The default config expects:

- `final_reviews.csv` with `user_id`, `app_id`, and `is_recommended`
- `final_games.csv` with `app_id`, `title`, and optionally `user_reviews`

If your files live elsewhere, update `configs/recommender.yaml`.

## Kaggle Dataset

For Colab or a fresh rebuild, use Kaggle's [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).

After downloading and unzipping it, keep the original file names:

```text
data/
  games.csv
  recommendations.csv
  users.csv
  games_metadata.json
```

Then run with the Colab-ready config:

```bash
python -m src.train --config configs/recommender_colab.yaml
python -m src.evaluate --config configs/recommender_colab.yaml
```

The current pipeline requires only `games.csv` and `recommendations.csv`. The dataset files are intentionally ignored by Git.
