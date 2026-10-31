# Demo

## 60–90 second recording script (recording itself: pending)

1. (0:00) Open `http://127.0.0.1:8101/` — explain: 1990 California block-group data,
   estimates are historical-data model outputs, not valuations.
2. (0:15) Filter listings to "Bay Area", show the price column.
3. (0:30) Submit the default prediction form; point at `predicted_price`,
   `model_version`, and empty `warnings`.
4. (0:50) Change Latitude/Longitude to Southern California values; predict again —
   different price, same model version.
5. (1:05) Set Median income to 999; predict — out-of-range warning appears and the
   request still succeeds with the caveat attached.
6. (1:20) Open `/api/model` — show held-out test MAE/RMSE and split sizes.

## Live walkthrough (commands)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_data.py   # verifies shipped CSV / re-downloads
.venv/bin/python scripts/train.py           # writes artifacts/ + prints metrics
PORT=8101 .venv/bin/python -m app.main &    # API + UI
bash scripts/smoke_test.sh                  # healthz/readyz/model/predict
# Browser: http://127.0.0.1:8101/
# Cleanup: kill %1; rm -rf artifacts data/predictions.db .venv (optional)
```

Windows (PowerShell equivalents): `py -m venv .venv; .venv\Scripts\pip install -r requirements-dev.txt; .venv\Scripts\python scripts\train.py; $env:PORT=8101; .venv\Scripts\python -m app.main`

## Public deployment route (pending — no account authorized)

Single VM: reverse proxy (Caddy/nginx, HTTPS) -> Flask app container from the
committed Dockerfile; model artifact baked in at image build; SQLite volume or
the Compose PostgreSQL profile for history; only the proxy port exposed.
Resource need: ~1 GB RAM, 1 vCPU is sufficient at demo scale.
