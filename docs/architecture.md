# Architecture

```
data/california_housing.csv (20,640 rows, checksum in data/DATA_CARD.md)
        |
        v
scripts/train.py ---- ingest/validate/dedupe
        |---- GroupShuffleSplit by 0.1° lat/lon grid (train/val/test)
        |---- preprocessing fit on TRAIN ONLY (median impute, scale, one-hot)
        |---- median baseline | Ridge (tuned on val) | RandomForest (tuned on val)
        |---- winner evaluated ONCE on untouched test
        v
artifacts/model.joblib  (pipeline + schema + ranges + dataset fingerprint)
artifacts/metrics.json / residuals.csv / model_card.json   (gitignored, regenerated)
        |
        v
app/main.py (Flask) ---- GET / (UI) ---- browser
        |---- /api/predict  -> shared app/features.py transforms -> pipeline
        |---- /api/markets, /api/listings, /api/model, /api/history
        |---- prediction history -> SQLite data/predictions.db (no personal data)
        v
optional streamlit_app.py (calls the same API; not the verified UI)
```

Key design point: **one feature module** (`app/features.py`) is imported by both
training and serving, so there is no second implementation to drift. The versioned
bundle carries the feature schema, training ranges (for out-of-range warnings),
dataset SHA-256 fingerprint, and sklearn version together — a model is never
deployed without the exact schema that produced it.

Docker Compose (`docker-compose.yml`) packages the same app; the optional
`with-db` profile adds PostgreSQL for a deployed variant (see decisions.md).
