"""
Tests must not depend on network access to OpenML. This fixture trains a
tiny, fast, deterministic model on synthetic data with the SAME feature
names/order and calibration wrapper the real pipeline uses, and points the
predictor at it for the duration of the test session.
"""
import json
import pickle

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]


@pytest.fixture(scope="session", autouse=True)
def fake_model_artifacts(tmp_path_factory):
    artifacts_dir = tmp_path_factory.mktemp("artifacts")

    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 0] + X[:, 4] > 0).astype(int)  # arbitrary but non-trivial signal

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = CalibratedClassifierCV(LogisticRegression(), method="sigmoid", cv=3)
    model.fit(X_scaled, y)

    with open(artifacts_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(artifacts_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(artifacts_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump({"note": "fake test metrics"}, f)
    with open(artifacts_dir / "background_sample.json", "w") as f:
        json.dump(X_scaled[:30].tolist(), f)

    # Point the singleton predictor at these test artifacts before any
    # test imports app.main (which constructs the FastAPI app at import time).
    import app.ml.predictor as predictor_module

    predictor_module.predictor = predictor_module.Predictor(artifacts_dir=artifacts_dir)

    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    import app.routers.predict as predict_router
    import app.routers.health as health_router
    import app.ml.predictor as predictor_module

    # Make sure the routers use the test predictor instance too, since they
    # imported the singleton by reference at module load time.
    predict_router.predictor = predictor_module.predictor
    health_router.predictor = predictor_module.predictor

    return TestClient(app)


VALID_PATIENT = {
    "age": 54,
    "gender": 2,
    "height": 175,
    "weight": 82,
    "ap_hi": 130,
    "ap_lo": 85,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}
