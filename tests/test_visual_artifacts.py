import json
from pathlib import Path
import socket

import joblib
import numpy as np
from PIL import Image
import pytest

from src.visual_pipeline import VisualModelBundle, load_visual_runtime, predict_visual_anomaly


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "factorymind_visual_quality_model_v1.joblib"
METADATA = ROOT / "models" / "factorymind_visual_quality_model_v1.metadata.json"


def test_artifacts_exist_reload_offline_and_match_frozen_contract(monkeypatch):
    assert MODEL.is_file() and METADATA.is_file()
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    runtime = load_visual_runtime(MODEL, device="cpu")
    assert isinstance(runtime.bundle, VisualModelBundle)
    assert runtime.bundle.coreset.shape == (2458, 384)
    assert runtime.bundle.threshold == pytest.approx(0.454687, abs=5e-6)
    assert runtime.bundle.display_low == pytest.approx(0.181344, abs=5e-6)
    assert runtime.bundle.display_high == runtime.bundle.threshold
    assert not runtime.extractor.training and not any(p.requires_grad for p in runtime.extractor.parameters())


def test_reloaded_artifact_reproduces_score_decision_and_maps():
    image = Image.new("RGB", (51, 43), (120, 100, 80))
    first = predict_visual_anomaly(load_visual_runtime(MODEL, device="cpu"), image)
    second = predict_visual_anomaly(load_visual_runtime(MODEL, device="cpu"), image)
    assert first["visual_anomaly_score"] == second["visual_anomaly_score"]
    assert first["anomaly_detected"] == second["anomaly_detected"]
    assert np.array_equal(first["raw_anomaly_map"], second["raw_anomaly_map"])
    assert np.array_equal(first["normalized_anomaly_map"], second["normalized_anomaly_map"])


def test_metadata_complete_and_path_free():
    metadata = json.loads(METADATA.read_text())
    required = {
        "model_name", "model_version", "model_family", "dataset", "dataset_category", "dataset_license_note",
        "training_protocol", "reference_image_count", "development_normal_count", "test_good_count", "test_anomaly_count",
        "defect_subtypes", "accepted_image_formats", "minimum_source_dimensions", "input_size", "input_color_mode",
        "rgba_handling", "resize_method", "antialias", "normalization", "backbone", "backbone_weights",
        "backbone_frozen", "feature_layers", "feature_map_grid", "patch_embedding_dim", "patches_per_image",
        "full_reference_patch_count", "coreset_method", "coreset_projection_dim", "coreset_seed",
        "coreset_retention_ratio", "coreset_patch_count", "nearest_neighbor_algorithm", "nearest_neighbor_metric",
        "patch_score_method", "image_score_method", "threshold_method", "threshold_quantile", "threshold_raw_score",
        "display_scale_low", "display_scale_high", "notebook_16_image_level_metrics", "notebook_16_pixel_level_metrics",
        "per_subtype_metrics", "false_positive_false_negative_tradeoff", "known_limitations", "output_interpretation",
        "warning", "disclaimer", "python_version", "torch_version", "torchvision_version", "numpy_version",
        "pillow_version", "sklearn_version", "training_timestamp",
    }
    assert required.issubset(metadata)
    serialized = json.dumps(metadata)
    assert str(ROOT) not in serialized and "/Users/" not in serialized
    assert metadata["threshold_raw_score"] == pytest.approx(0.454687, abs=5e-6)
