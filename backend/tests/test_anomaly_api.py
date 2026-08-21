from dataclasses import replace
import json
import logging

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from src.anomaly_features import ANOMALY_SENSOR_COLUMNS
from src.anomaly_pipeline import ANOMALY_WARNING, evaluate_anomaly_trajectory
from src.anomaly_train import CANONICAL_COLUMNS, DATA_PATH


RAW = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=CANONICAL_COLUMNS)
LOW = RAW.iloc[0][ANOMALY_SENSOR_COLUMNS].to_dict()
HIGH = RAW[RAW.unit_id.eq(1)].iloc[-1][ANOMALY_SENSOR_COLUMNS].to_dict()


def payload_from_pattern(pattern, unit_id="engine-42"):
    observations=[]
    for cycle, high in enumerate(pattern, 1):
        observations.append({"cycle":cycle, **(HIGH if high else LOW)})
    return {"unit_id":unit_id, "observations":observations}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client: yield test_client


@pytest.mark.parametrize("length,status",[(1,"insufficient_history"),(4,"insufficient_history"),(5,"available"),(8,"available")])
def test_valid_anomaly_trajectory_lengths(client,length,status):
    response=client.post("/predict/anomaly",json=payload_from_pattern([False]*length))
    assert response.status_code==200
    body=response.json()
    assert np.isfinite(body["current_anomaly_score"])
    assert 0<=body["anomaly_percentile"]<=100
    assert body["history_cycle_count"]==length
    assert body["persistence_status"]==status
    assert body["alert_active"] is False
    assert body["warning"]==ANOMALY_WARNING
    assert len(body["top_sensor_deviations"])==5
    assert "probability" not in body


@pytest.mark.parametrize("pattern,active",[
    ([False]*5,False),([True,True,False,False,False],False),
    ([True,True,True,False,False],True),([True]*4+[False],True),([True]*5,True),
])
def test_persistence_patterns_match_source(client,pattern,active):
    payload=payload_from_pattern(pattern)
    response=client.post("/predict/anomaly",json=payload)
    trajectory=pd.DataFrame(payload["observations"],columns=["cycle"]+ANOMALY_SENSOR_COLUMNS)
    direct=evaluate_anomaly_trajectory(client.app.state.model_resources.anomaly.bundle,trajectory)
    assert response.status_code==200
    assert response.json()["recent_exceedance_pattern"]==direct["recent_exceedance_pattern"]
    assert response.json()["alert_active"] is active


def test_older_observations_do_not_change_latest_five(client):
    first=client.post("/predict/anomaly",json=payload_from_pattern([False]*5)).json()
    second=client.post("/predict/anomaly",json=payload_from_pattern([True]*5+[False]*5)).json()
    assert first["recent_exceedance_pattern"]==second["recent_exceedance_pattern"]
    assert first["alert_active"]==second["alert_active"] is False


def test_deterministic_request_and_unit_identity(client):
    payload=payload_from_pattern([False,True,True,True,False])
    first=client.post("/predict/anomaly",json=payload)
    second=client.post("/predict/anomaly",json=payload)
    assert first.json()==second.json() and first.json()["unit_id"]=="engine-42"


@pytest.mark.parametrize("mutator",[
    lambda p:{**p,"observations":[]},
    lambda p:{**p,"observations":[{k:v for k,v in p["observations"][0].items() if k!="sensor_11"}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"extra":1}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"raw_rul":10}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"sensor_11":"47.0"}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"sensor_11":True}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"cycle":1.5}]},
    lambda p:{**p,"observations":[{**p["observations"][0],"cycle":0}]},
])
def test_schema_validation_rejects_invalid_payloads(client,mutator):
    assert client.post("/predict/anomaly",json=mutator(payload_from_pattern([False]))).status_code==422


@pytest.mark.parametrize("invalid",[float("nan"),float("inf"),float("-inf")])
def test_nonfinite_values_rejected(client,invalid):
    payload=payload_from_pattern([False]); payload["observations"][0]["sensor_11"]=invalid
    response=client.post("/predict/anomaly",content=json.dumps(payload),headers={"content-type":"application/json"})
    assert response.status_code==422


@pytest.mark.parametrize("cycles",[[1,1],[1,3],[2,1]])
def test_source_temporal_validation_maps_to_422(client,cycles):
    payload=payload_from_pattern([False]*len(cycles))
    for row,cycle in zip(payload["observations"],cycles): row["cycle"]=cycle
    response=client.post("/predict/anomaly",json=payload)
    assert response.status_code==422 and "traceback" not in response.text.lower()


def test_model_info_and_openapi_are_curated(client):
    info=client.get("/model/anomaly/info")
    assert info.status_code==200
    body=info.json()
    assert body["model_family"]=="IsolationForest" and body["predictor_count"]==14
    assert body["threshold_percentile"]==97.5 and body["persistence_window"]==5
    paths=client.get("/openapi.json").json()["paths"]
    assert "/predict/anomaly" in paths and "/model/anomaly/info" in paths
    assert "nonprobabilistic" in paths["/predict/anomaly"]["post"]["description"]


def test_anomaly_unavailable_returns_503(client):
    original=client.app.state.model_resources
    client.app.state.model_resources=replace(original,anomaly=None)
    try:
        prediction=client.post("/predict/anomaly",json=payload_from_pattern([False]))
        health=client.get("/health")
    finally: client.app.state.model_resources=original
    assert prediction.status_code==503 and health.status_code==503


class ExplodingDetector:
    def score_samples(self, values): raise RuntimeError("private anomaly failure")


def test_unexpected_inference_is_safe_and_logged(client,caplog):
    original=client.app.state.model_resources
    anomaly=original.anomaly
    broken_bundle=replace(anomaly.bundle,detector=ExplodingDetector())
    client.app.state.model_resources=replace(original,anomaly=replace(anomaly,bundle=broken_bundle))
    try:
        with caplog.at_level(logging.ERROR,logger="backend.app.services.anomaly_prediction_service"):
            response=client.post("/predict/anomaly",json=payload_from_pattern([False]))
    finally: client.app.state.model_resources=original
    assert response.status_code==500
    assert response.json()=={"detail":"Anomaly prediction could not be completed."}
    assert "private anomaly failure" not in response.text
    assert "Unexpected anomaly inference error" in caplog.text
