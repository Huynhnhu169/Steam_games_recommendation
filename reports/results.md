# Results

## Current Status

Corrected metrics are not available yet.

The original notebook reported RMSE-style results, but the custom KNN workflow built user-item matrices from the full interaction set before splitting. That can leak held-out interactions into similarity computation.

This repository now evaluates with a leakage-safe protocol:

1. Split interactions into train/test per user.
2. Build matrices and similarities only from training interactions.
3. Evaluate held-out test interactions with top-K ranking metrics.

## Corrected Metrics

TODO after running:

```bash
python -m src.train --config configs/recommender.yaml
python -m src.evaluate --config configs/recommender.yaml
```

| Model | Precision@K | Recall@K | NDCG@K | MAP@K | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | TODO | TODO | TODO | TODO | TODO |
| Item-item KNN | TODO | TODO | TODO | TODO | TODO |
| User-user KNN optional | TODO | TODO | TODO | TODO | TODO |

## Reporting Notes

- Report the exact `K` value.
- Report train/test interaction counts.
- Do not compare corrected results directly against the old notebook RMSE as a valid benchmark.
