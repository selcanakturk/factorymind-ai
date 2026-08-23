from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import pytest

from src.visual_pipeline import (
    QUALITY_ANOMALOUS, QUALITY_NORMAL, create_visual_runtime,
    normalize_anomaly_map, predict_visual_anomaly, score_patch_embeddings,
    threshold_exceeded, upsample_anomaly_map,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "factorymind_visual_quality_model_v1.joblib"


def test_threshold_strict_equality_semantics():
    threshold = 0.5
    assert threshold_exceeded(threshold - 1e-8, threshold).item() is False
    assert threshold_exceeded(threshold, threshold).item() is False
    assert threshold_exceeded(threshold + 1e-8, threshold).item() is True


def test_display_normalization_and_upsampling_contract():
    raw = np.linspace(0, 3, 256, dtype=np.float32).reshape(16, 16)
    copy = raw.copy(); normalized = normalize_anomaly_map(raw, 1.0, 2.0)
    assert np.array_equal(raw, copy)
    assert normalized.min() == 0 and normalized.max() == 1
    assert np.all(np.diff(normalized.ravel()) >= 0)
    assert normalize_anomaly_map(raw, 1.0, 2.0)[0, 0] == 0
    upsampled = upsample_anomaly_map(raw, 41, 53)
    assert upsampled.shape == (41, 53)
    assert np.array_equal(upsampled, upsample_anomaly_map(raw, 41, 53))


def test_nearest_scoring_invariants_with_loaded_bundle():
    bundle = joblib.load(ARTIFACT); runtime = create_visual_runtime(bundle, device="cpu")
    patches = np.repeat(bundle.coreset[[0]], 256, axis=0)
    distances, raw_map, score = score_patch_embeddings(runtime, patches)
    assert distances.shape == (256,) and raw_map.shape == (16, 16)
    assert np.isfinite(distances).all() and (distances >= 0).all()
    assert score == float(distances.max()) == float(raw_map.max())


def test_repeated_single_image_prediction_is_identical():
    runtime = create_visual_runtime(joblib.load(ARTIFACT), device="cpu")
    image = Image.new("RGB", (53, 41), (70, 90, 110))
    first = predict_visual_anomaly(runtime, image); second = predict_visual_anomaly(runtime, image)
    for key in ("visual_anomaly_score", "threshold", "anomaly_detected", "quality_status"):
        assert first[key] == second[key]
    for key in ("patch_scores", "raw_anomaly_map", "upsampled_raw_anomaly_map", "normalized_anomaly_map"):
        assert np.array_equal(first[key], second[key])
    assert first["raw_anomaly_map"].shape == (16, 16)
    assert first["normalized_anomaly_map"].shape == (41, 53)
    assert 0 <= first["normalized_anomaly_map"].min() <= first["normalized_anomaly_map"].max() <= 1
    assert first["quality_status"] in (QUALITY_NORMAL, QUALITY_ANOMALOUS)
