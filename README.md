# Steam Games Recommendation

This repository is a reproducible recommender-system project for Steam games. It rebuilds the original notebook workflow into a small Python pipeline with leakage-safe top-K evaluation.

## Problem Statement

Given historical user-game feedback, recommend games a user is likely to enjoy. The project treats positive `is_recommended` reviews as implicit feedback and evaluates whether held-out positive interactions appear in the top-K recommendation list.

## Dataset

The expected files are:

```text
data/
  final_reviews.csv
  final_games.csv
```

Expected review columns:

- `user_id`
- `app_id`
- `is_recommended`

Expected game columns:

- `app_id`
- `title`
- `user_reviews` for optional top-N filtering

Datasets are not committed to GitHub. See [data/README.md](data/README.md).

For a clean Colab run, use Kaggle's [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset and the Colab config:

```text
data/
  recommendations.csv
  games.csv
```

See [COLAB.md](COLAB.md) for setup, Kaggle download, training, evaluation, and artifact-saving commands.

## Method

The rebuilt pipeline includes:

- popularity baseline
- item-item KNN collaborative filtering
- user-user KNN collaborative filtering, implemented but disabled by default for scalability

The default task is implicit-feedback top-K recommendation. Positive recommendations are mapped to `1`; negative reviews are not misused as `-1` ratings for a `0..1` scale.

## Leakage-Safe Evaluation Protocol

The evaluation protocol is intentionally strict:

1. Load raw interactions from CSV.
2. Keep positive implicit feedback rows.
3. Filter users/items according to config.
4. Split interactions into train/test per user.
5. Build user-item matrices only from the training interactions.
6. Compute popularity and KNN similarity only from the training interactions.
7. Evaluate recommendations against held-out test interactions.

The old notebook built a user-item matrix from the full dataset before splitting, which can leak test interactions into similarity scores. This rebuilt version avoids that.

## Metrics

The main metrics are ranking metrics:

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- Catalog coverage

RMSE is not used as the main metric because this project is framed as top-K implicit recommendation.

## Current Results

Corrected metrics are TODO until the pipeline is run with the dataset. See [reports/results.md](reports/results.md).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The core pipeline does not require Jupyter. On Windows, installing the full `jupyter` package can fail in deeply nested folders because some JupyterLab asset paths are very long. Use the CLI commands below for training and evaluation; install notebook tools separately only if you need them.

If `pip install` fails with a missing file under `.venv\share\jupyter\labextensions`, delete the broken `.venv` folder and recreate it before reinstalling the core requirements. That error comes from a damaged Jupyter installation inside the virtual environment, not from the recommender pipeline dependencies.

## Data Preparation

Place `final_reviews.csv` and `final_games.csv` under `data/`, or edit `configs/recommender.yaml` to point to another location.

```bash
python -m src.train --config configs/recommender.yaml
```

Generated splits, metrics, and model artifacts are ignored by Git.

For Kaggle/Colab, keep the original extracted file names and use:

```bash
python -m src.train --config configs/recommender_colab.yaml
```

The Colab config reads the large `recommendations.csv` file in chunks and starts with a small top-game filter so the first run can fit in a standard Colab runtime.

## Google Colab

Use the Colab guide:

```text
COLAB.md
```

Short version:

```bash
pip install -r requirements-colab.txt
python -m src.train --config configs/recommender_colab.yaml
python -m src.evaluate --config configs/recommender_colab.yaml
```

## Train

```bash
python -m src.train --config configs/recommender.yaml
```

## Evaluate

```bash
python -m src.evaluate --config configs/recommender.yaml
```

## Recommend

Recommend similar games by title:

```bash
python -m src.recommend --config configs/recommender.yaml --game "Counter-Strike 2" --top-k 10
```

Recommend games for a known user:

```bash
python -m src.recommend --config configs/recommender.yaml --user-id 12345 --top-k 10
```

The recommendation command loads trained artifacts when available. If artifacts are missing, it fits a full-data demo model for interactive recommendations only; this mode is not used for evaluation metrics.

## Project Structure

```text
Steam_games_recommendation/
  README.md
  COLAB.md
  LICENSE
  requirements.txt
  requirements-colab.txt
  .gitignore
  configs/
    recommender.yaml
    recommender_colab.yaml
  data/
    README.md
  src/
    __init__.py
    data.py
    preprocess.py
    models.py
    metrics.py
    train.py
    evaluate.py
    recommend.py
  notebooks/
    01_exploration.ipynb
    02_modeling_clean.ipynb
  reports/
    results.md
  assets/
    sample_recommendations.md
```

## Limitations

- Corrected metrics have not been generated in this repository yet.
- The default setup evaluates positive implicit feedback only.
- Cold-start users and cold-start games are not solved.
- The dataset source and license must be checked before publishing derived artifacts.
- User-user KNN can be slower on large datasets than item-item KNN; enable it in config only when runtime is acceptable.

## Future Work

- Add corrected metrics after running the pipeline.
- Add a stronger matrix-factorization baseline.
- Add temporal splitting if review timestamps are available.
- Add cold-start recommendations based on game metadata.
- Add automated tests for metrics and leakage-safe splitting.
