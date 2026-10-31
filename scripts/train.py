#!/usr/bin/env python3
"""Reproducible training pipeline for HomeValue.

Steps: ingest -> validate -> deduplicate -> geographic-group split ->
train-only preprocessing -> baseline / Ridge / RandomForest comparison
-> small validation-only tuning -> single final test evaluation -> export.

Usage:  .venv/bin/python scripts/train.py
Outputs (gitignored, regenerated):
  artifacts/model.joblib          versioned bundle (pipeline + metadata)
  artifacts/metrics.json          all split metrics incl. test (once)
  artifacts/residuals.csv         test residuals for the winning model
  artifacts/model_card.json       schema, fingerprint, ranges, versions
Committed summaries live in docs/model-results.json (copied from metrics).
"""
from __future__ import annotations

import hashlib, json, platform, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.features import (ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                          RAW_COLUMNS, TARGET, build_features)

DATA = ROOT / "data" / "california_housing.csv"
ART = ROOT / "artifacts"
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mae(y, p): return float(mean_absolute_error(y, p))
def scores(y, p): return {"mae_usd": round(mae(y, p), 2), "rmse_usd": round(rmse(y, p), 2), "n": int(len(y))}


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    # Tree model uses its own unscaled numeric branch below.
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def preprocessor_tree() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    fingerprint = sha256(DATA)

    # ---- ingest + validate ----
    df = pd.read_csv(DATA)
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"dataset missing columns: {missing_cols}")
    n_raw = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    df = df[(df[TARGET] > 0) & (df["Households"] > 0)]
    n_before_dedupe = len(df)
    df = df.drop_duplicates(subset=RAW_COLUMNS).reset_index(drop=True)
    validation = {"rows_raw": n_raw, "rows_after_target_household_filter": n_before_dedupe,
                  "duplicates_removed": int(n_before_dedupe - len(df)), "rows_modelled": int(len(df))}

    X = build_features(df)
    y = df[TARGET].astype(float)

    # ---- split: geographic groups (0.1-degree grid cells) so near-identical
    # neighbouring block groups do not leak across train/test ----
    groups = (df["Latitude"].round(1).astype(str) + ":" + df["Longitude"].round(1).astype(str))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(X, y, groups))
    X_tv, y_tv = X.iloc[trainval_idx], y.iloc[trainval_idx]
    g_tv = groups.iloc[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)  # ~15% of total
    tr_idx, val_idx = next(gss2.split(X_tv, y_tv, g_tv))
    X_tr, y_tr = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
    X_val, y_val = X_tv.iloc[val_idx], y_tv.iloc[val_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    split_sizes = {"train": int(len(X_tr)), "validation": int(len(X_val)), "test": int(len(X_te)),
                   "split_method": "GroupShuffleSplit by 0.1-degree lat/lon grid cell; preprocessing fit on train only"}

    results = {}

    # ---- baseline: training median ----
    base = Pipeline([("pre", preprocessor()), ("model", DummyRegressor(strategy="median"))])
    base.fit(X_tr, y_tr)
    results["median_baseline"] = {"validation": scores(y_val, base.predict(X_val))}

    # ---- Ridge, tuned on validation only ----
    best_ridge, best_alpha = None, None
    for alpha in [1.0, 10.0, 100.0]:
        m = Pipeline([("pre", preprocessor()), ("model", Ridge(alpha=alpha))])
        m.fit(X_tr, y_tr)
        s = scores(y_val, m.predict(X_val))
        if best_ridge is None or s["rmse_usd"] < results["ridge"]["validation"]["rmse_usd"]:
            results["ridge"] = {"validation": s, "alpha": alpha}
            best_ridge = m
    # ---- RandomForest, tuned on validation only ----
    # Serving-size constraint (documented in docs/decisions.md): depth is
    # capped and min_samples_leaf=2 because an unconstrained 200-tree forest
    # on this data serializes to ~242 MB with no meaningful accuracy gain.
    best_rf = None
    for max_depth in [16, 20]:
        m = Pipeline([("pre", preprocessor_tree()),
                      ("model", RandomForestRegressor(n_estimators=200, max_depth=max_depth,
                                                      min_samples_leaf=2,
                                                      n_jobs=-1, random_state=SEED))])
        m.fit(X_tr, y_tr)
        s = scores(y_val, m.predict(X_val))
        if best_rf is None or s["rmse_usd"] < results["random_forest"]["validation"]["rmse_usd"]:
            results["random_forest"] = {"validation": s, "max_depth": max_depth,
                                        "min_samples_leaf": 2, "n_estimators": 200}
            best_rf = m

    winner_name = min(["ridge", "random_forest"], key=lambda k: results[k]["validation"]["rmse_usd"])
    winner = best_ridge if winner_name == "ridge" else best_rf

    # ---- final test evaluation: exactly once, on the untouched test set ----
    test_pred = winner.predict(X_te)
    results[winner_name]["test"] = scores(y_te, test_pred)
    results["median_baseline"]["test"] = scores(y_te, base.predict(X_te))
    other = "random_forest" if winner_name == "ridge" else "ridge"
    other_model = best_rf if other == "random_forest" else best_ridge
    results[other]["test"] = scores(y_te, other_model.predict(X_te))

    # price-band slices on test (winner)
    resid = pd.DataFrame({"actual": y_te.values, "predicted": test_pred,
                          "residual": y_te.values - test_pred,
                          "region": X_te["region"].values})
    resid.to_csv(ART / "residuals.csv", index=False)
    resid["band"] = pd.cut(resid["actual"], bins=[0, 150_000, 250_000, 350_000, 500_001],
                           labels=["<150k", "150-250k", "250-350k", "350-500k"])
    slices = {}
    for band, g in resid.groupby("band", observed=True):
        if len(g) >= 30:
            slices[str(band)] = {"n": int(len(g)), "mae_usd": round(mae(g["actual"], g["predicted"]), 2),
                                 "rmse_usd": round(rmse(g["actual"], g["predicted"]), 2)}
    for region, g in resid.groupby("region"):
        if len(g) >= 30:
            slices[f"region:{region}"] = {"n": int(len(g)), "mae_usd": round(mae(g["actual"], g["predicted"]), 2),
                                          "rmse_usd": round(rmse(g["actual"], g["predicted"]), 2)}
    results[winner_name]["test_slices"] = slices

    ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in NUMERIC_FEATURES}
    import sklearn
    model_version = f"1.0.0-{fingerprint[:8]}"
    bundle = {"pipeline": winner, "model_version": model_version, "winner": winner_name,
              "feature_schema": ALL_FEATURES, "ranges": ranges,
              "dataset_fingerprint_sha256": fingerprint, "currency": "USD",
              "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, ART / "model.joblib")

    metrics = {"dataset": "California Housing (1990 census block groups)",
               "validation_report": validation, "splits": split_sizes,
               "models": results, "winner": winner_name, "model_version": model_version,
               "dataset_fingerprint_sha256": fingerprint,
               "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "platform": platform.platform(), "train_seconds": round(time.time() - t0, 2)}}
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART / "model_card.json").write_text(json.dumps(
        {"model_version": model_version, "winner": winner_name, "features": ALL_FEATURES,
         "ranges": ranges, "dataset_fingerprint_sha256": fingerprint,
         "test_metrics": results[winner_name]["test"]}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
