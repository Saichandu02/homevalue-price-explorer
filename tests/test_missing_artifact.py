"""readyz must fail when the trained artifact is missing."""
import importlib
from pathlib import Path
import app.main as main_mod

def test_readyz_fails_without_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "MODEL_PATH", Path(tmp_path) / "nope.joblib")
    app = main_mod.create_app(bundle=None)
    # create_app(bundle=None) reloads from MODEL_PATH -> still missing -> None
    c = app.test_client()
    assert c.get("/readyz").status_code == 503
    assert c.post("/api/predict", json={"MedInc": 1}).status_code == 503
    assert c.get("/healthz").status_code == 200
