from dataclasses import replace
import json
import logging

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.model_loader import RULModelResources
from backend.app.main import app
from src.rul_features import RAW_INPUT_COLUMNS
from src.rul_pipeline import RUL_WARNING, predict_latest_rul


BASE_OBSERVATION = {
    "cycle": 1,
    "operational_setting_1": -0.0007,
    "operational_setting_2": -0.0004,
    "operational_setting_3": 100.0,
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 39.06,
    "sensor_21": 23.4190,
}


def make_request(length=1, unit_id="engine-42"):
    observations = []
    for index in range(length):
        row = dict(BASE_OBSERVATION)
        row["cycle"] = index + 1
        row["sensor_4"] += index * 0.4
        row["sensor_11"] += index * 0.02
        row["sensor_12"] -= index * 0.03
        observations.append(row)
    return {"unit_id": unit_id, "observations": observations}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("length", "quality"),
    [(1, "limited_history"), (3, "limited_history"), (6, "full_context"), (12, "full_context")],
)
def test_valid_trajectory_inference(client, length, quality):
    response = client.post("/predict/rul", json=make_request(length))
    assert response.status_code == 200
    body = response.json()
    assert body["unit_id"] == "engine-42"
    assert body["history_cycle_count"] == length
    assert body["history_quality"] == quality
    assert np.isfinite(body["raw_model_prediction"])
    assert 0 <= body["raw_model_prediction"] <= 125
    assert body["prediction_horizon_cap"] == 125
    assert body["warning"] == RUL_WARNING
    assert body["development_stage"] is True


def test_one_cycle_uses_long_horizon_display(client):
    response = client.post("/predict/rul", json=make_request(1))
    assert response.status_code == 200
    assert response.json()["rul_display"] == "125+ cycle horizon"


def test_api_matches_direct_source_inference(client):
    payload = make_request(6)
    response = client.post("/predict/rul", json=payload)
    trajectory = pd.DataFrame(payload["observations"], columns=RAW_INPUT_COLUMNS)
    direct = predict_latest_rul(
        client.app.state.model_resources.rul.model, trajectory
    )
    assert response.status_code == 200
    assert {key: value for key, value in response.json().items() if key != "unit_id"} == direct


def test_unit_id_is_nonpredictive_metadata(client):
    first = client.post("/predict/rul", json=make_request(6, "engine-A")).json()
    second = client.post("/predict/rul", json=make_request(6, "engine-B")).json()
    first.pop("unit_id")
    second.pop("unit_id")
    assert first == second


def test_repeated_request_is_deterministic(client):
    payload = make_request(6)
    assert client.post("/predict/rul", json=payload).json() == client.post(
        "/predict/rul", json=payload
    ).json()


@pytest.mark.parametrize(
    "payload",
    [
        {"observations": []},
        {"observations": [{key: value for key, value in BASE_OBSERVATION.items() if key != "sensor_11"}]},
        {"observations": [{**BASE_OBSERVATION, "raw_rul": 10}]},
        {"observations": [{**BASE_OBSERVATION, "unexpected_sensor": 10}]},
        {"observations": [{**BASE_OBSERVATION, "sensor_11": "47.47"}]},
        {"observations": [{**BASE_OBSERVATION, "sensor_11": True}]},
        {"observations": [{**BASE_OBSERVATION, "cycle": 1.5}]},
        {"observations": [{**BASE_OBSERVATION, "cycle": 0}]},
    ],
)
def test_invalid_schema_returns_422(client, payload):
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_sensor_returns_422(client, invalid):
    payload = make_request(1)
    payload["observations"][0]["sensor_11"] = invalid
    response = client.post(
        "/predict/rul",
        content=json.dumps(payload),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "cycles",
    [[1, 1], [1, 3], [2, 1]],
)
def test_invalid_temporal_contract_returns_422(client, cycles):
    payload = make_request(len(cycles))
    for observation, cycle in zip(payload["observations"], cycles):
        observation["cycle"] = cycle
    response = client.post("/predict/rul", json=payload)
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_rul_model_info_is_curated(client):
    response = client.get("/model/rul/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert body["model_family"] == "RandomForestRegressor"
    assert body["dataset_subset"] == "FD001"
    assert body["rul_cap"] == 125
    assert body["predictor_count"] == 47
    assert body["minimum_full_context_cycles"] == 6
    assert body["known_limitations"]


def test_openapi_contains_rul_routes(client):
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/predict/rul" in paths
    assert "/model/rul/info" in paths
    assert paths["/predict/rul"]["post"]["summary"] == "Estimate remaining useful life"


def test_rul_unavailable_returns_503(client):
    original = client.app.state.model_resources
    client.app.state.model_resources = replace(original, rul=None)
    try:
        prediction = client.post("/predict/rul", json=make_request(1))
        health = client.get("/health")
    finally:
        client.app.state.model_resources = original
    assert prediction.status_code == 503
    assert health.status_code == 503


class ExplodingModel:
    def predict(self, model_input):
        raise RuntimeError("private internal inference detail")


def test_unexpected_inference_failure_is_safe_and_logged(client, caplog):
    original = client.app.state.model_resources
    broken_rul = RULModelResources(
        model=ExplodingModel(), metadata=original.rul.metadata
    )
    client.app.state.model_resources = replace(original, rul=broken_rul)
    try:
        with caplog.at_level(
            logging.ERROR, logger="backend.app.services.rul_prediction_service"
        ):
            response = client.post("/predict/rul", json=make_request(1))
    finally:
        client.app.state.model_resources = original
    assert response.status_code == 500
    assert response.json() == {"detail": "RUL prediction could not be completed."}
    assert "private internal inference detail" not in response.text
    assert "Unexpected RUL inference error" in caplog.text
