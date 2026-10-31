import pytest
from app.main import create_app

VALID = {"MedInc": 5.0, "HouseAge": 20, "AveRooms": 6.0, "AveBedrms": 1.0,
         "Population": 1000, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23}

@pytest.fixture(scope="module")
def client():
    app = create_app()
    if app.test_client().get("/readyz").status_code != 200:
        pytest.skip("model artifact not trained yet; run scripts/train.py")
    return app.test_client()

def test_healthz(client):
    assert client.get("/healthz").status_code == 200

def test_readyz_and_model(client):
    assert client.get("/readyz").status_code == 200
    m = client.get("/api/model").get_json()
    assert m["currency"] == "USD" and m["model_version"]
    assert m["test_metrics"]["n"] > 1000

def test_predict_ok(client):
    r = client.post("/api/predict", json=VALID)
    assert r.status_code == 200
    body = r.get_json()
    assert body["currency"] == "USD"
    assert 10_000 < body["predicted_price"] < 500_001
    assert body["model_version"]

def test_predict_missing_field(client):
    bad = dict(VALID); del bad["MedInc"]
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 400 and "MedInc" in str(r.get_json())

def test_predict_malformed_type(client):
    bad = dict(VALID); bad["MedInc"] = "lots"
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 400

def test_predict_no_json(client):
    r = client.post("/api/predict", data="nope", content_type="text/plain")
    assert r.status_code == 400

def test_predict_unknown_region_warns_not_crashes(client):
    bad = dict(VALID); bad["region"] = "Atlantis"
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 200
    assert any("Atlantis" in w for w in r.get_json()["warnings"])

def test_predict_out_of_range_warns(client):
    bad = dict(VALID); bad["MedInc"] = 999.0
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 200
    assert any("MedInc" in w for w in r.get_json()["warnings"])

def test_markets_and_listings(client):
    assert client.get("/api/markets").status_code == 200
    d = client.get("/api/listings?region=Bay%20Area&limit=5").get_json()
    assert len(d["listings"]) == 5
    assert all(x["region"] == "Bay Area" for x in d["listings"])
