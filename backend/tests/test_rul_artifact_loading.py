import json

import joblib
import pytest

from backend.app.core import model_loader
from backend.app.core.model_loader import ArtifactLoadError, load_rul_resources


@pytest.fixture(scope="module")
def actual_rul_model():
    return joblib.load(model_loader.RUL_MODEL_PATH)


def configure_rul_artifacts(
    monkeypatch,
    tmp_path,
    actual_rul_model,
    *,
    missing=None,
    metadata_mutator=None,
    metadata_text=None,
    loaded_model=None,
):
    model_path = tmp_path / "rul.joblib"
    metadata_path = tmp_path / "rul.metadata.json"
    metadata = json.loads(model_loader.RUL_METADATA_PATH.read_text())
    if metadata_mutator:
        metadata_mutator(metadata)
    if missing != "model":
        model_path.write_bytes(b"placeholder")
    if missing != "metadata":
        metadata_path.write_text(
            metadata_text if metadata_text is not None else json.dumps(metadata)
        )
    monkeypatch.setattr(model_loader, "RUL_MODEL_PATH", model_path)
    monkeypatch.setattr(model_loader, "RUL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        model_loader.joblib,
        "load",
        lambda path: actual_rul_model if loaded_model is None else loaded_model,
    )


@pytest.mark.parametrize("missing", ["model", "metadata"])
def test_missing_rul_artifacts_fail_loading(
    monkeypatch, tmp_path, actual_rul_model, missing
):
    configure_rul_artifacts(
        monkeypatch, tmp_path, actual_rul_model, missing=missing
    )
    with pytest.raises(ArtifactLoadError, match="missing"):
        load_rul_resources()


def test_malformed_rul_metadata_fails_loading(monkeypatch, tmp_path, actual_rul_model):
    configure_rul_artifacts(
        monkeypatch,
        tmp_path,
        actual_rul_model,
        metadata_text="{invalid-json",
    )
    with pytest.raises(ArtifactLoadError, match="valid RUL model metadata JSON"):
        load_rul_resources()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda metadata: metadata.pop("model_name"), "model_name"),
        (lambda metadata: metadata.update(predictor_count=46), "predictor_count"),
        (lambda metadata: metadata.update(rul_cap=120), "rul_cap"),
        (lambda metadata: metadata["predictor_columns"].pop(), "predictor_columns"),
        (lambda metadata: metadata.update(model_family="OtherRegressor"), "model_family"),
        (lambda metadata: metadata["package_versions"].update(scikit_learn="0.0"), "incompatible"),
    ],
)
def test_incompatible_rul_metadata_fails_loading(
    monkeypatch, tmp_path, actual_rul_model, mutator, message
):
    configure_rul_artifacts(
        monkeypatch, tmp_path, actual_rul_model, metadata_mutator=mutator
    )
    with pytest.raises(ArtifactLoadError, match=message):
        load_rul_resources()


def test_incompatible_rul_model_fails_loading(monkeypatch, tmp_path, actual_rul_model):
    configure_rul_artifacts(
        monkeypatch, tmp_path, actual_rul_model, loaded_model=object()
    )
    with pytest.raises(ArtifactLoadError, match="sklearn Pipeline"):
        load_rul_resources()
