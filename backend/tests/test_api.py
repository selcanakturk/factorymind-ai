import json

import pytest
from fastapi.testclient import TestClient

from backend.app.core.model_loader import THRESHOLD_PATH
from backend.app.main import app
from backend.app.services.prediction_service import risk_category_for_score


LOW_RISK_REQUEST = {
    "type": "M",
    "air_temperature": 300.1,
    "process_temperature": 310.4,
    "rotational_speed": 1450,
    "torque": 48.2,
    "tool_wear": 125,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_succeeds_and_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "FactoryMind AI",
        "model_loaded": True,
        "failure_model_loaded": True,
        "rul_model_loaded": True,
    }


def test_model_info_comes_from_artifact_metadata(client):
    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert body["threshold_version"] == "1.0.0"
    assert body["calibration_method"] == "sigmoid"
    assert len(body["raw_input_features"]) == 6
    assert "Temperature difference" in body["engineered_features"]


def test_valid_failure_prediction_has_expected_contract(client):
    response = client.post("/predict/failure", json=LOW_RISK_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "calibrated_risk_estimate",
        "failure_risk_score",
        "risk_category",
        "recommended_action",
        "model_version",
        "threshold_version",
        "calibration_method",
        "disclaimer",
    }
    assert 0 <= body["calibrated_risk_estimate"] <= 1
    assert 0 <= body["failure_risk_score"] <= 100
    assert body["risk_category"] in {
        "Low Risk", "Medium Risk", "High Risk", "Critical Risk"
    }
    assert body["recommended_action"]
    assert "development-stage" in body["disclaimer"]
    assert "probability" not in body


def test_invalid_type_is_rejected(client):
    invalid = {**LOW_RISK_REQUEST, "type": "X"}

    response = client.post("/predict/failure", json=invalid)

    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    incomplete = {key: value for key, value in LOW_RISK_REQUEST.items() if key != "torque"}

    response = client.post("/predict/failure", json=incomplete)

    assert response.status_code == 422


def test_non_numeric_sensor_value_is_rejected(client):
    invalid = {**LOW_RISK_REQUEST, "tool_wear": "not-a-number"}

    response = client.post("/predict/failure", json=invalid)

    assert response.status_code == 422


def test_identical_request_is_deterministic(client):
    first = client.post("/predict/failure", json=LOW_RISK_REQUEST)
    second = client.post("/predict/failure", json=LOW_RISK_REQUEST)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_category_boundaries_use_threshold_artifact():
    thresholds = json.loads(THRESHOLD_PATH.read_text())["thresholds"]
    epsilon = 1e-9

    assert risk_category_for_score(thresholds["medium"] - epsilon, thresholds) == "Low Risk"
    assert risk_category_for_score(thresholds["medium"], thresholds) == "Medium Risk"
    assert risk_category_for_score(thresholds["high"] - epsilon, thresholds) == "Medium Risk"
    assert risk_category_for_score(thresholds["high"], thresholds) == "High Risk"
    assert risk_category_for_score(thresholds["critical"] - epsilon, thresholds) == "High Risk"
    assert risk_category_for_score(thresholds["critical"], thresholds) == "Critical Risk"


def test_only_raw_features_are_needed_and_engineered_fields_are_rejected(client):
    raw_response = client.post("/predict/failure", json=LOW_RISK_REQUEST)
    with_engineered_feature = {
        **LOW_RISK_REQUEST,
        "temperature_difference": 10.3,
    }
    extra_response = client.post("/predict/failure", json=with_engineered_feature)

    assert raw_response.status_code == 200
    assert extra_response.status_code == 422


def test_swagger_openapi_schema_is_available(client):
    docs = client.get("/docs")
    schema = client.get("/openapi.json")

    assert docs.status_code == 200
    assert schema.status_code == 200
    assert "/predict/failure" in schema.json()["paths"]


def test_local_frontend_origin_can_preflight_prediction(client):
    response = client.options(
        "/predict/failure",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unlisted_origin_is_not_granted_cors_access(client):
    response = client.options(
        "/predict/failure",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers
