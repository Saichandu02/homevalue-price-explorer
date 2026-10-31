"""Train/serve parity and split reproducibility."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import build_features, raw_from_request

ROOT = Path(__file__).resolve().parents[1]

def test_features_deterministic_and_shared():
    raw = pd.DataFrame([{"Longitude": -122.23, "Latitude": 37.88, "HouseAge": 41.0,
                         "TotalRooms": 880.0, "TotalBedrooms": 129.0, "Population": 322.0,
                         "Households": 126.0, "MedInc": 8.3252, "MedHouseVal": 452600.0}])
    a = build_features(raw)
    b = build_features(raw.copy())
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[0]["region"] == "Bay Area"
    assert abs(a.iloc[0]["AveRooms"] - 880.0 / 126.0) < 1e-9

def test_request_roundtrip_matches_direct_features():
    """A payload built from a raw row must yield (nearly) the same features
    as build_features on that row — the parity contract."""
    raw = pd.DataFrame([{"Longitude": -122.23, "Latitude": 37.88, "HouseAge": 41.0,
                         "TotalRooms": 880.0, "TotalBedrooms": 129.0, "Population": 322.0,
                         "Households": 126.0, "MedInc": 8.3252, "MedHouseVal": 452600.0}])
    direct = build_features(raw).iloc[0]
    payload = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 880.0 / 126.0,
               "AveBedrms": 129.0 / 126.0, "Population": 322.0, "AveOccup": 322.0 / 126.0,
               "Latitude": 37.88, "Longitude": -122.23}
    via_api = raw_from_request(payload).iloc[0]
    for col in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
                "AveOccup", "Latitude", "Longitude"]:
        assert abs(float(direct[col]) - float(via_api[col])) < 1e-6, col
    assert direct["region"] == via_api["region"]

def test_split_reproducible_from_metrics():
    metrics_path = ROOT / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        return  # training not run in this checkout yet
    m = json.loads(metrics_path.read_text())
    total = m["splits"]["train"] + m["splits"]["validation"] + m["splits"]["test"]
    assert total == m["validation_report"]["rows_modelled"]
    # test set is a meaningful held-out slice, evaluated once
    assert m["splits"]["test"] > 2000
    assert m["models"][m["winner"]]["test"]["n"] == m["splits"]["test"]
