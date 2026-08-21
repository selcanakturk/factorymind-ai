from dataclasses import replace
import json

import joblib
import numpy as np
import pytest

from backend.app.core import model_loader
from backend.app.core.model_loader import ArtifactLoadError, load_anomaly_resources


@pytest.fixture(scope="module")
def actual_bundle(): return joblib.load(model_loader.ANOMALY_MODEL_PATH)


def configure(monkeypatch,tmp_path,actual_bundle,*,missing=None,mutator=None,text=None,bundle=None):
    model_path=tmp_path/"anomaly.joblib"; metadata_path=tmp_path/"anomaly.metadata.json"
    metadata=json.loads(model_loader.ANOMALY_METADATA_PATH.read_text())
    if mutator: mutator(metadata)
    if missing!="model": model_path.write_bytes(b"placeholder")
    if missing!="metadata": metadata_path.write_text(text if text is not None else json.dumps(metadata))
    monkeypatch.setattr(model_loader,"ANOMALY_MODEL_PATH",model_path)
    monkeypatch.setattr(model_loader,"ANOMALY_METADATA_PATH",metadata_path)
    monkeypatch.setattr(model_loader.joblib,"load",lambda path: actual_bundle if bundle is None else bundle)


@pytest.mark.parametrize("missing",["model","metadata"])
def test_missing_artifacts(monkeypatch,tmp_path,actual_bundle,missing):
    configure(monkeypatch,tmp_path,actual_bundle,missing=missing)
    with pytest.raises(ArtifactLoadError,match="missing"): load_anomaly_resources()


def test_malformed_metadata(monkeypatch,tmp_path,actual_bundle):
    configure(monkeypatch,tmp_path,actual_bundle,text="{invalid")
    with pytest.raises(ArtifactLoadError,match="valid anomaly model metadata JSON"): load_anomaly_resources()


@pytest.mark.parametrize("mutator,message",[
    (lambda m:m.pop("model_name"),"model_name"),
    (lambda m:m["predictor_columns"].pop(),"predictor_columns"),
    (lambda m:m.update(threshold_quantile=.95),"threshold_quantile"),
    (lambda m:m.update(persistence_window=4),"persistence_window"),
    (lambda m:m.update(persistence_required_count=2),"persistence_required_count"),
    (lambda m:m["package_versions"].update(scikit_learn="0.0"),"incompatible"),
])
def test_incompatible_metadata(monkeypatch,tmp_path,actual_bundle,mutator,message):
    configure(monkeypatch,tmp_path,actual_bundle,mutator=mutator)
    with pytest.raises(ArtifactLoadError,match=message): load_anomaly_resources()


@pytest.mark.parametrize("bundle,message",[(object(),"AnomalyModelBundle")])
def test_malformed_bundle_type(monkeypatch,tmp_path,actual_bundle,bundle,message):
    configure(monkeypatch,tmp_path,actual_bundle,bundle=bundle)
    with pytest.raises(ArtifactLoadError,match=message): load_anomaly_resources()


def test_missing_scaler(monkeypatch,tmp_path,actual_bundle):
    configure(monkeypatch,tmp_path,actual_bundle,bundle=replace(actual_bundle,scaler=None))
    with pytest.raises(ArtifactLoadError,match="StandardScaler"): load_anomaly_resources()


def test_missing_detector(monkeypatch,tmp_path,actual_bundle):
    configure(monkeypatch,tmp_path,actual_bundle,bundle=replace(actual_bundle,detector=None))
    with pytest.raises(ArtifactLoadError,match="IsolationForest"): load_anomaly_resources()


def test_malformed_reference_scores(monkeypatch,tmp_path,actual_bundle):
    scores=np.array(actual_bundle.sorted_normal_scores); scores[0]=np.nan
    configure(monkeypatch,tmp_path,actual_bundle,bundle=replace(actual_bundle,sorted_normal_scores=scores))
    with pytest.raises(ArtifactLoadError,match="reference-score"): load_anomaly_resources()
