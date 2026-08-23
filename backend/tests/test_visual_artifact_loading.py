from dataclasses import replace
import json

import joblib
import numpy as np
import pytest
import torch

from backend.app.core import model_loader
from backend.app.core.model_loader import ArtifactLoadError, load_visual_resources


@pytest.fixture(scope="module")
def actual_bundle(): return joblib.load(model_loader.VISUAL_MODEL_PATH)


def configure(monkeypatch, tmp_path, actual_bundle, *, missing=None, mutator=None, text=None, bundle=None):
    model_path = tmp_path / "visual.joblib"; metadata_path = tmp_path / "visual.metadata.json"
    metadata = json.loads(model_loader.VISUAL_METADATA_PATH.read_text())
    if mutator: mutator(metadata)
    if missing != "model": model_path.write_bytes(b"placeholder")
    if missing != "metadata": metadata_path.write_text(text if text is not None else json.dumps(metadata))
    monkeypatch.setattr(model_loader, "VISUAL_MODEL_PATH", model_path)
    monkeypatch.setattr(model_loader, "VISUAL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(model_loader.joblib, "load", lambda path: actual_bundle if bundle is None else bundle)


@pytest.mark.parametrize("missing", ["model", "metadata"])
def test_missing_artifacts(monkeypatch, tmp_path, actual_bundle, missing):
    configure(monkeypatch, tmp_path, actual_bundle, missing=missing)
    with pytest.raises(ArtifactLoadError, match="missing"): load_visual_resources()


def test_malformed_metadata(monkeypatch, tmp_path, actual_bundle):
    configure(monkeypatch, tmp_path, actual_bundle, text="{invalid")
    with pytest.raises(ArtifactLoadError, match="valid visual quality metadata JSON"): load_visual_resources()


@pytest.mark.parametrize(("mutator", "message"), [
    (lambda m: m.update(dataset_category="bottle"), "dataset_category"),
    (lambda m: m.update(threshold_raw_score=.4), "threshold"),
    (lambda m: m.update(feature_map_grid=[8, 8]), "feature_map_grid"),
    (lambda m: m.update(patch_embedding_dim=128), "patch_embedding_dim"),
    (lambda m: m.update(coreset_patch_count=2000), "coreset_patch_count"),
    (lambda m: m.update(torch_version="0.0"), "incompatible"),
])
def test_incompatible_metadata(monkeypatch, tmp_path, actual_bundle, mutator, message):
    configure(monkeypatch, tmp_path, actual_bundle, mutator=mutator)
    with pytest.raises(ArtifactLoadError, match=message): load_visual_resources()


def test_malformed_bundle_type(monkeypatch, tmp_path, actual_bundle):
    configure(monkeypatch, tmp_path, actual_bundle, bundle=object())
    with pytest.raises(ArtifactLoadError, match="VisualModelBundle"): load_visual_resources()


def test_malformed_coreset(monkeypatch, tmp_path, actual_bundle):
    configure(monkeypatch, tmp_path, actual_bundle, bundle=replace(actual_bundle, coreset=np.zeros((2, 384), np.float32)))
    with pytest.raises(ArtifactLoadError, match="malformed"): load_visual_resources()


def test_malformed_backbone_state(monkeypatch, tmp_path, actual_bundle):
    configure(monkeypatch, tmp_path, actual_bundle, bundle=replace(actual_bundle, backbone_state_dict={"bad": torch.tensor(1)}))
    with pytest.raises(ArtifactLoadError, match="runtime could not be constructed"): load_visual_resources()
