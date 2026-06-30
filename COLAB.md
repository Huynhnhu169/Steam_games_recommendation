# Running on Google Colab

This project can run on a standard Colab runtime. GPU is not required because the current models use pandas, SciPy, and scikit-learn instead of deep learning frameworks.

The recommended dataset is Kaggle's [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam). The full dataset is large, so the Colab config starts with the top 100 games by review count and reads `recommendations.csv` in chunks.

## 1. Clone the Repository

```python
!git clone https://github.com/Huynhnhu169/Steam_games_recommendation.git
%cd Steam_games_recommendation
```

If you are testing local changes, upload the project folder to Drive or clone your development branch instead.

## 2. Install Dependencies

```python
!pip install -r requirements-colab.txt
```

## 3. Add Kaggle Credentials

Download `kaggle.json` from your Kaggle account settings, then upload it in Colab:

```python
from google.colab import files
files.upload()
```

Move the uploaded credential file into the Kaggle config folder:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Do not commit `kaggle.json` to GitHub.

## 4. Download the Dataset

```python
!mkdir -p data
!kaggle datasets download -d antonkozyriev/game-recommendations-on-steam -p data
!unzip -o data/game-recommendations-on-steam.zip -d data
```

After extraction, the project should contain:

```text
data/
  games.csv
  recommendations.csv
  users.csv
  games_metadata.json
```

Only `games.csv` and `recommendations.csv` are required by the current pipeline.

## 5. Train

```python
!python -m src.train --config configs/recommender_colab.yaml
```

The Colab config uses:

- `recommendations.csv` as the interaction file
- `games.csv` as the item catalog
- chunked reading for the large reviews file
- `top_n_games_by_reviews: 100` for a first reproducible run

Increase `top_n_games_by_reviews` gradually after the first successful run.

## 6. Evaluate

```python
!python -m src.evaluate --config configs/recommender_colab.yaml
```

Evaluation is leakage-safe: interactions are split into train/test before user-item matrices and similarity scores are built.

## 7. Recommend

Recommend similar games by title:

```python
!python -m src.recommend --config configs/recommender_colab.yaml --game "Counter-Strike 2" --top-k 10
```

Recommend games for a known user:

```python
!python -m src.recommend --config configs/recommender_colab.yaml --user-id 12345 --top-k 10
```

## 8. Save Outputs to Google Drive

Colab storage is temporary. Mount Drive if you want to keep model artifacts and reports:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!mkdir -p /content/drive/MyDrive/steam_recommender_outputs
!cp -r runs models /content/drive/MyDrive/steam_recommender_outputs/
```

## Notes

- Do not upload CSV files, model artifacts, or Kaggle credentials to GitHub.
- The first Colab run is intentionally small. Report metrics only for the exact config you used.
- The full `recommendations.csv` file is large; use chunked loading and keep user-user KNN disabled unless you have enough memory.
