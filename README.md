# HomeValue — Housing Price Explorer and Prediction API

A reproducible portfolio application that turns the California Housing dataset
(1990 census block groups, 20,640 rows) into an explorable prediction service:
filter listings, inspect regional price distributions and held-out model error,
and get a price estimate through a browser UI and a JSON API.

**Honesty note:** the data is from 1990. Estimates are model outputs on historical
data, *not* current market valuations, and no other market (e.g. Vijayawada) is
included or claimed anywhere in this repo.

![UI screenshot](docs/screenshot.txt) — screenshot capture is pending; `docs/screenshot.txt`
contains a text capture of the verified UI HTML instead of a fabricated image.

## Implemented capabilities

- Reproducible pipeline (`scripts/train.py`): ingest → validate → deduplicate →
  geographic-group split → train-only preprocessing → model comparison → export.
- Three models compared honestly: median baseline, Ridge (regularized linear),
  RandomForest — tuned on validation only, test set evaluated once.
- Versioned bundle: model + feature schema + training ranges + dataset SHA-256
  fingerprint travel together; the same `app/features.py` code runs at train
  and serve time.

## Verified local quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py   # verifies the shipped CSV (SHA-256 in data/DATA_CARD.md)
.venv/bin/python scripts/train.py           # ~70 s on 4 cores; writes artifacts/
PORT=8101 .venv/bin/python -m app.main      # UI + API on http://127.0.0.1:8101
bash scripts/smoke_test.sh                  # in a second terminal
.venv/bin/pytest -q
```

Windows/PowerShell equivalents are in `docs/demo.md`. Docker: `docker compose up --build`
is provided as configuration (not executed on the build VM — no Docker there).

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | liveness |
| `GET /readyz` | 200 only when the model artifact is loaded, else 503 |
| `GET /api/markets` | per-region listing counts and median/mean prices |
| `GET /api/model` | model version, schema, held-out test metrics, split sizes |
| `POST /api/predict` | price estimate (below) |
| `GET /api/listings?region=&limit=` | sample listings for the explorer UI |
| `GET /api/history` | recent predictions (SQLite, no personal data) |

```bash
curl -X POST http://127.0.0.1:8101/api/predict -H 'Content-Type: application/json' \
  -d '{"MedInc":5.0,"HouseAge":20,"AveRooms":6.0,"AveBedrms":1.0,"Population":1000,"AveOccup":3.0,"Latitude":37.88,"Longitude":-122.23}'
# -> {"predicted_price": 250328.61, "currency": "USD",
#     "model_version": "1.0.0-8e28941c", "warnings": []}
```

## Measured results (held-out test, n=2,473, geographic-group split)

| Model | MAE (USD) | RMSE (USD) |
|---|---|---|
| Median baseline | 90,826 | 126,608 |
| Ridge (alpha=1.0) | 57,166 | 82,598 |
| **RandomForest (depth≤20, winner)** | **38,183** | **56,870** |

Error grows with price band (350–500k band MAE $78,145) and 177 test rows sit at
the source data's $500,001 target cap, where the model under-predicts by ~$84k on
average — see `docs/verification.md` for slices, residuals, and the full report.

## Architecture / docs

- [docs/architecture.md](docs/architecture.md) — diagram and data flow
- [docs/demo.md](docs/demo.md) — demo walkthrough + 60–90 s recording script
- [docs/decisions.md](docs/decisions.md) — split, leakage, intervals, UI/store choices
- [docs/verification.md](docs/verification.md) — what actually ran, with real outputs
- [data/DATA_CARD.md](data/DATA_CARD.md) — dataset source, license, checksum, caveats

## Limitations & tradeoffs

- 1990 aggregates, block groups not individual homes; location dominates.
- Target capped at $500,001 in the source → systematic under-prediction at the cap.
- No prediction intervals (a valid one needs a separate calibration set — omitted
  deliberately rather than faked; see decisions.md).
- Public deployment: **pending** — no deployment account authorized. Route is
  documented in docs/demo.md. Local completion only.
- Streamlit/Plotly are optional extras; the verified UI is the Flask page.

## Resume bullets (personal project, from measured results only)

- Built a reproducible housing-price prediction service (Flask + scikit-learn) on
  20,640 California census block groups, using a geographic-group split and
  train-only preprocessing to avoid leakage; the tuned RandomForest reached
  held-out MAE $38.2k / RMSE $56.9k (n=2,473) vs a $90.8k MAE median baseline.
- Shipped a versioned model bundle (schema + dataset fingerprint + range warnings)
  with train/serve feature parity tests, a prediction API with input validation,
  and price-band/regional error slices documenting where the model fails.

## Release checklist

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). License: MIT (code);
dataset attribution in data/DATA_CARD.md.
