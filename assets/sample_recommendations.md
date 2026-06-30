# Sample Recommendations

Sample recommendations are TODO until the model is trained on the local dataset.

Run:

```bash
python -m src.recommend --config configs/recommender.yaml --game "Counter-Strike 2" --top-k 10
```

or:

```bash
python -m src.recommend --config configs/recommender.yaml --user-id 12345 --top-k 10
```
