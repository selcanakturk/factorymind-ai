import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core import model_loader
from backend.app.core.model_loader import (
    ArtifactLoadError,
    ModelResources,
    _resolve_positive_class_index,
    load_model_resources,
)
from backend.app.main import app


VALID_REQUEST = {
    "type": "M",
    "air_temperature": 300.1,
    "process_temperature": 310.4,
    "rotational_speed": 1450,
    "torque": 48.2,
    "tool_wear": 125,
}


class FakeModel:
    def __init__(self, classes=(0, 1), output=None):
        self.classes_ = np.asarray(classes)
        self.output = np.asarray([[0.8, 0.2]] if output is None else output)

    def predict_proba(self, model_input):
        return self.output


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("air_temperature", float("nan")),
        ("air_temperature", float("inf")),
        ("air_temperature", float("-inf")),
        ("torque", True),
        ("rotational_speed", "1500"),
    ],
)
def test_strict_nonfinite_and_non_numeric_values_return_422(
    client, field, invalid_value
):
    response = client.post(
        "/predict/failure",
        content=json.dumps({**VALID_REQUEST, field: invalid_value}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("extra_field", ["unrelated", "Machine failure", "TWF"])
def test_unrelated_target_and_failure_mode_fields_are_rejected(client, extra_field):
    response = client.post(
        "/predict/failure", json={**VALID_REQUEST, extra_field: 1}
    )

    assert response.status_code == 422


def _configure_artifacts(
    monkeypatch,
    tmp_path,
    *,
    missing=None,
    metadata_mutator=None,
    threshold_mutator=None,
    threshold_text=None,
    model=None,
):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    threshold_path = tmp_path / "thresholds.json"

    metadata = json.loads(model_loader.MODEL_METADATA_PATH.read_text())
    thresholds = json.loads(model_loader.THRESHOLD_PATH.read_text())
    if metadata_mutator:
        metadata_mutator(metadata)
    if threshold_mutator:
        threshold_mutator(thresholds)

    if missing != "model":
        model_path.write_bytes(b"test-model-placeholder")
    if missing != "metadata":
        metadata_path.write_text(json.dumps(metadata))
    if missing != "threshold":
        threshold_path.write_text(
            threshold_text if threshold_text is not None else json.dumps(thresholds)
        )

    monkeypatch.setattr(model_loader, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_loader, "MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(model_loader, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(
        model_loader.joblib, "load", lambda path: model or FakeModel()
    )


@pytest.mark.parametrize("missing", ["model", "metadata", "threshold"])
def test_missing_required_artifacts_fail_loading(monkeypatch, tmp_path, missing):
    _configure_artifacts(monkeypatch, tmp_path, missing=missing)

    with pytest.raises(ArtifactLoadError, match="missing"):
        load_model_resources()


def test_malformed_json_fails_loading(monkeypatch, tmp_path):
    _configure_artifacts(
        monkeypatch, tmp_path, threshold_text="{not-valid-json"
    )

    with pytest.raises(ArtifactLoadError, match="valid risk threshold JSON"):
        load_model_resources()


def test_unordered_thresholds_fail_loading(monkeypatch, tmp_path):
    def mutate(thresholds):
        thresholds["thresholds"]["high"] = thresholds["thresholds"]["medium"]

    _configure_artifacts(monkeypatch, tmp_path, threshold_mutator=mutate)

    with pytest.raises(ArtifactLoadError, match="medium < high < critical"):
        load_model_resources()


def test_numeric_string_threshold_fails_loading(monkeypatch, tmp_path):
    def mutate(thresholds):
        thresholds["thresholds"]["medium"] = "0.042"

    _configure_artifacts(monkeypatch, tmp_path, threshold_mutator=mutate)

    with pytest.raises(ArtifactLoadError, match="finite JSON number"):
        load_model_resources()


def test_calibration_method_mismatch_fails_loading(monkeypatch, tmp_path):
    def mutate(thresholds):
        thresholds["calibration_method"] = "isotonic"

    _configure_artifacts(monkeypatch, tmp_path, threshold_mutator=mutate)

    with pytest.raises(ArtifactLoadError, match="do not match"):
        load_model_resources()


def test_missing_required_metadata_field_fails_loading(monkeypatch, tmp_path):
    def mutate(metadata):
        metadata.pop("model_version")

    _configure_artifacts(monkeypatch, tmp_path, metadata_mutator=mutate)

    with pytest.raises(ArtifactLoadError, match="model_version"):
        load_model_resources()


def test_incorrect_raw_feature_contract_fails_loading(monkeypatch, tmp_path):
    def mutate(metadata):
        metadata["raw_input_features"] = metadata["raw_input_features"][:-1]

    _configure_artifacts(monkeypatch, tmp_path, metadata_mutator=mutate)

    with pytest.raises(ArtifactLoadError, match="raw_input_features"):
        load_model_resources()


def test_positive_class_label_is_resolved_from_actual_order():
    assert _resolve_positive_class_index(FakeModel(classes=(1, 0))) == 0
    assert _resolve_positive_class_index(FakeModel(classes=(0, 1))) == 1


def test_missing_positive_class_fails_loading():
    with pytest.raises(ArtifactLoadError, match="positive class label 1"):
        _resolve_positive_class_index(FakeModel(classes=(0, 2)))


def test_incompatible_multiclass_structure_fails_loading():
    with pytest.raises(ArtifactLoadError, match="exactly two"):
        _resolve_positive_class_index(FakeModel(classes=(0, 1, 2)))


@pytest.mark.parametrize(
    "bad_output",
    [np.asarray([0.8, 0.2]), np.asarray([[0.8, np.nan]])],
)
def test_invalid_inference_output_returns_generic_500(
    client, caplog, bad_output
):
    original_resources = client.app.state.model_resources
    client.app.state.model_resources = ModelResources(
        model=FakeModel(output=bad_output),
        model_metadata=original_resources.model_metadata,
        threshold_metadata=original_resources.threshold_metadata,
        positive_class_index=1,
    )
    try:
        with caplog.at_level(
            logging.ERROR,
            logger="backend.app.services.prediction_service",
        ):
            response = client.post("/predict/failure", json=VALID_REQUEST)
    finally:
        client.app.state.model_resources = original_resources

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction could not be completed."}
    assert "predict_proba" not in response.text
    assert "nan" not in response.text.lower()
    assert "Unexpected failure-risk inference error" in caplog.text


def test_health_returns_503_when_resources_are_unavailable(client):
    original_resources = client.app.state.model_resources
    client.app.state.model_resources = None
    try:
        response = client.get("/health")
    finally:
        client.app.state.model_resources = original_resources

    assert response.status_code == 503
    assert response.json() == {"detail": "Model resources are unavailable."}
