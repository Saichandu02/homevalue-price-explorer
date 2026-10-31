"""Shared feature engineering for HomeValue.

This module is the single source of truth used by BOTH training
(scripts/train.py) and serving (app/main.py). Train/serve parity is
guaranteed by construction and verified in tests/test_parity.py.

Raw columns come from the California Housing dataset (1990 census
block groups). Engineered features follow the standard derivation:
  AveRooms  = totalRooms / households
  AveBedrms = totalBedrooms / households
  AveOccup  = population / households
  region    = deterministic geographic category from (Longitude, Latitude)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "Longitude", "Latitude", "HouseAge", "TotalRooms", "TotalBedrooms",
    "Population", "Households", "MedInc", "MedHouseVal",
]
# Rename map raw CSV -> canonical names
CANONICAL = {
    "longitude": "Longitude", "latitude": "Latitude",
    "housingMedianAge": "HouseAge", "totalRooms": "TotalRooms",
    "totalBedrooms": "TotalBedrooms", "population": "Population",
    "households": "Households", "medianIncome": "MedInc",
    "medianHouseValue": "MedHouseVal",
}

NUMERIC_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
CATEGORICAL_FEATURES = ["region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "MedHouseVal"
CURRENCY = "USD"

REGIONS = ["Bay Area", "Southern California", "Central Valley",
           "Northern California", "Other California"]


def region_for(lat: float, lon: float) -> str:
    """Deterministic geographic category. Pure function, no learned state."""
    if -124.5 <= lon <= -122.0 and 37.0 <= lat <= 38.6:
        return "Bay Area"
    if -121.0 <= lon <= -114.0 and 32.5 <= lat <= 35.5:
        return "Southern California"
    if -122.0 < lon <= -118.5 and 35.0 < lat <= 40.5:
        return "Central Valley"
    if lat > 40.5:
        return "Northern California"
    return "Other California"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw canonical frame -> model feature frame. Used at train and serve time."""
    out = pd.DataFrame(index=df.index)
    households = df["Households"].replace(0, np.nan)
    out["MedInc"] = df["MedInc"]
    out["HouseAge"] = df["HouseAge"]
    out["AveRooms"] = df["TotalRooms"] / households
    out["AveBedrms"] = df["TotalBedrooms"] / households
    out["Population"] = df["Population"]
    out["AveOccup"] = df["Population"] / households
    out["Latitude"] = df["Latitude"]
    out["Longitude"] = df["Longitude"]
    out["region"] = [region_for(lat, lon) for lat, lon in zip(df["Latitude"], df["Longitude"])]
    return out[ALL_FEATURES]


def raw_from_request(payload: dict) -> pd.DataFrame:
    """Convert one API prediction payload into a raw canonical frame.

    The API accepts the engineered ratios directly (AveRooms etc.) plus
    location; Households/TotalRooms are reconstructed only to reuse
    build_features unchanged:
      Households := 1 unit basis, TotalRooms := AveRooms, etc.
    This keeps exactly one feature code-path.
    """
    ave_rooms = float(payload["AveRooms"])
    ave_bedrms = float(payload["AveBedrms"])
    ave_occup = float(payload["AveOccup"])
    population = float(payload["Population"])
    households = population / ave_occup if ave_occup else 1.0
    raw = pd.DataFrame([{
        "Longitude": float(payload["Longitude"]),
        "Latitude": float(payload["Latitude"]),
        "HouseAge": float(payload["HouseAge"]),
        "TotalRooms": ave_rooms * households,
        "TotalBedrooms": ave_bedrms * households,
        "Population": population,
        "Households": households,
        "MedInc": float(payload["MedInc"]),
        "MedHouseVal": np.nan,
    }])
    feats = build_features(raw)
    # Region may be supplied explicitly by the caller (e.g. what-if
    # exploration); unknown values are passed through so the encoder's
    # handle_unknown='ignore' behaviour is exercised, not hidden.
    if payload.get("region"):
        feats.loc[:, "region"] = str(payload["region"])
    return feats


def input_warnings(feats: pd.DataFrame, ranges: dict) -> list[str]:
    """Out-of-training-range warnings for one feature row."""
    warnings = []
    row = feats.iloc[0]
    for col in NUMERIC_FEATURES:
        lo, hi = ranges.get(col, (None, None))
        if lo is None:
            continue
        if row[col] < lo or row[col] > hi:
            warnings.append(f"{col}={row[col]:.4g} is outside the training range [{lo:.4g}, {hi:.4g}]")
    if row["region"] not in REGIONS:
        warnings.append(f"region '{row['region']}' was not seen in training; it will be ignored by the encoder")
    return warnings
