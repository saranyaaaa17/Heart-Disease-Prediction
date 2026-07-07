from .conftest import VALID_PATIENT


def test_predict_happy_path_returns_expected_shape(client):
    response = client.post("/api/v1/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"is_high_risk", "risk_label", "risk_probability_pct"}
    assert isinstance(body["is_high_risk"], bool)
    assert 0.0 <= body["risk_probability_pct"] <= 100.0


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/api/v1/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_values(client):
    bad_patient = {**VALID_PATIENT, "age": -5}
    response = client.post("/api/v1/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_rejects_negative_height(client):
    bad_patient = {**VALID_PATIENT, "height": -170}
    response = client.post("/api/v1/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_response_has_request_id_header(client):
    response = client.post("/api/v1/predict", json=VALID_PATIENT)
    assert "X-Request-ID" in response.headers


def test_get_features_returns_names_and_note(client):
    response = client.get("/api/v1/features")
    assert response.status_code == 200
    body = response.json()
    assert len(body["features"]) == 11
    assert "note" in body


def test_model_info_returns_metrics(client):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    assert "note" in response.json()


def test_explain_returns_per_feature_contributions(client):
    response = client.post("/api/v1/explain", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert "base_risk_probability" in body
    assert len(body["feature_contributions"]) == 11
    names = {fc["feature"] for fc in body["feature_contributions"]}
    assert names == set(VALID_PATIENT.keys())


def test_explain_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/api/v1/explain", json=incomplete)
    assert response.status_code == 422


def test_features_returns_coefficients_for_calibrated_linear_model(client):
    """The test fixture wraps LogisticRegression in CalibratedClassifierCV --
    /features should still find and return its coefficients, not report
    'not linear'."""
    response = client.get("/api/v1/features")
    body = response.json()
    assert body["coefficients"] is not None
    assert len(body["coefficients"]) == 11


def test_explain_llm_returns_503_when_not_configured(client, monkeypatch):
    """ANTHROPIC_API_KEY isn't set in the test environment -- this should be
    a clean 503, not a crash, since the LLM summary is an optional feature."""
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.post("/api/v1/explain-llm", json=VALID_PATIENT)
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
