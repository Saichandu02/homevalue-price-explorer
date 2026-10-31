"""HomeValue Flask API + UI.

Endpoints:
  GET  /healthz          liveness
  GET  /readyz           200 only if the trained model artifact is loaded
  GET  /api/markets      region summary from the bundled dataset sample stats
  GET  /api/model        model version, schema, test metrics
  POST /api/predict      {MedInc, HouseAge, AveRooms, AveBedrms, Population,
                          AveOccup, Latitude, Longitude, region?}
                         -> {predicted_price, currency, model_version, warnings}
  GET  /                   browser UI (filter sample listings + prediction form)
  GET  /api/listings     sample listings (from dataset, paginated/filterable)
  GET  /api/history      recent prediction history (SQLite, no personal data)
"""
from __future__ import annotations

import json, os, sqlite3, threading, time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from app.features import NUMERIC_FEATURES, input_warnings, raw_from_request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
DATA_PATH = ROOT / "data" / "california_housing.csv"
DB_PATH = Path(os.environ.get("HISTORY_DB", ROOT / "data" / "predictions.db"))
UI_DIR = ROOT / "ui"

REQUIRED = NUMERIC_FEATURES  # all numeric engineered features are required


def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')),
            input_json TEXT, predicted_price REAL, model_version TEXT)""")


def create_app(bundle=None) -> Flask:
    app = Flask(__name__)
    state = {"bundle": bundle if bundle is not None else load_bundle()}
    lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok", service="homevalue")

    @app.get("/readyz")
    def readyz():
        if state["bundle"] is None:
            return jsonify(status="not_ready", reason="model artifact missing"), 503
        return jsonify(status="ready", model_version=state["bundle"]["model_version"])

    @app.get("/api/model")
    def model_info():
        if state["bundle"] is None:
            return jsonify(error="model artifact missing"), 503
        b = state["bundle"]
        metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
        return jsonify(model_version=b["model_version"], winner=b["winner"],
                       currency=b["currency"], feature_schema=b["feature_schema"],
                       dataset_fingerprint_sha256=b["dataset_fingerprint_sha256"],
                       sklearn_version=b["sklearn_version"],
                       test_metrics=metrics.get("models", {}).get(b["winner"], {}).get("test"),
                       splits=metrics.get("splits"))

    @app.get("/api/markets")
    def markets():
        df = pd.read_csv(DATA_PATH, usecols=["Latitude", "Longitude", "MedHouseVal"])
        from app.features import region_for
        df["region"] = [region_for(a, o) for a, o in zip(df["Latitude"], df["Longitude"])]
        out = (df.groupby("region")["MedHouseVal"]
                 .agg(["count", "median", "mean"]).round(0).reset_index()
                 .rename(columns={"count": "listings", "median": "median_price_usd",
                                  "mean": "mean_price_usd"}))
        return jsonify(currency="USD", markets=out.to_dict(orient="records"))

    @app.get("/api/listings")
    def listings():
        region = request.args.get("region")
        limit = min(int(request.args.get("limit", 25)), 100)
        df = pd.read_csv(DATA_PATH).head(5000)
        from app.features import build_features, region_for
        feats = build_features(df)
        feats["price"] = df["MedHouseVal"].values
        feats["region"] = [region_for(a, o) for a, o in zip(df["Latitude"], df["Longitude"])]
        if region:
            feats = feats[feats["region"] == region]
        cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
                "AveOccup", "Latitude", "Longitude", "region", "price"]
        return jsonify(listings=feats[cols].head(limit).round(4).to_dict(orient="records"),
                       total=int(len(feats)))

    @app.post("/api/predict")
    def predict():
        if state["bundle"] is None:
            return jsonify(error="model artifact missing; run scripts/train.py"), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="JSON body required"), 400
        errors = []
        for f in REQUIRED:
            if f not in payload:
                errors.append(f"missing field: {f}")
            elif not isinstance(payload[f], (int, float)) or isinstance(payload[f], bool):
                errors.append(f"field {f} must be a number")
        if payload.get("AveOccup", 1) == 0:
            errors.append("AveOccup must be non-zero")
        if errors:
            return jsonify(error="invalid request", details=errors), 400
        try:
            feats = raw_from_request(payload)
        except (TypeError, ValueError) as e:
            return jsonify(error="invalid request", details=[str(e)]), 400
        b = state["bundle"]
        warnings = input_warnings(feats, b["ranges"])
        price = float(b["pipeline"].predict(feats)[0])
        with lock, sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO predictions(input_json, predicted_price, model_version) VALUES (?,?,?)",
                      (json.dumps(payload), price, b["model_version"]))
        return jsonify(predicted_price=round(price, 2), currency=b["currency"],
                       model_version=b["model_version"], warnings=warnings)

    @app.get("/api/history")
    def history():
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute("SELECT id, ts, predicted_price, model_version FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
        return jsonify(history=[{"id": r[0], "ts": r[1], "predicted_price": r[2], "model_version": r[3]} for r in rows])

    @app.get("/")
    def index():
        return send_from_directory(UI_DIR, "index.html")

    return app


init_db()
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8101")))
