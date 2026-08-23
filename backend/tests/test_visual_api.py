import base64
from io import BytesIO
import logging
from pathlib import Path
from dataclasses import replace

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from backend.app.main import VISUAL_MAX_UPLOAD_BYTES, app
from backend.app.services import visual_prediction_service
from src.visual_pipeline import predict_visual_anomaly


ROOT = Path(__file__).resolve().parents[2]
NORMAL = ROOT / "data/raw/mvtec_ad/zipper/test/good/000.png"
ANOMALOUS = ROOT / "data/raw/mvtec_ad/zipper/test/broken_teeth/000.png"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def image_bytes(mode="RGB", size=(40, 36), fmt="PNG", color=None):
    if color is None:
        color = 120 if mode == "L" else ((20, 40, 60, 0) if mode == "RGBA" else (20, 40, 60))
    output = BytesIO(); Image.new(mode, size, color).save(output, format=fmt); return output.getvalue()


@pytest.mark.parametrize(("payload", "filename", "mime"), [
    (image_bytes(fmt="JPEG"), "part.jpg", "image/jpeg"),
    (image_bytes(), "part.png", "image/png"),
    (image_bytes(mode="L"), "gray.png", "image/png"),
    (image_bytes(mode="RGBA"), "rgba.png", "image/png"),
    (image_bytes(size=(32, 32)), "minimum.png", "image/png"),
])
def test_valid_upload_contracts(client, payload, filename, mime):
    response = client.post("/predict/visual-quality", files={"file": (filename, payload, mime)})
    assert response.status_code == 200
    body = response.json()
    assert np.isfinite(body["visual_anomaly_score"]) and np.isfinite(body["threshold"])
    assert body["quality_status"] in ("No visual anomaly detected", "Visual anomaly detected")
    assert body["original_width"] >= 32 and body["original_height"] >= 32
    assert len(body["raw_anomaly_map_16x16"]) == 16 and all(len(row) == 16 for row in body["raw_anomaly_map_16x16"])
    assert body["anomaly_map_available"] and body["anomaly_map_label"] == "Model anomaly map"
    png = base64.b64decode(body["anomaly_map_image_base64"], validate=True)
    with Image.open(BytesIO(png)) as image:
        assert image.format == "PNG" and image.size == (body["original_width"], body["original_height"])


@pytest.mark.parametrize("path", [NORMAL, ANOMALOUS])
def test_direct_source_and_api_parity(client, path):
    resources = client.app.state.model_resources.visual_quality
    direct = predict_visual_anomaly(resources.runtime, path)
    response = client.post("/predict/visual-quality", files={"file": (path.name, path.read_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["visual_anomaly_score"] == pytest.approx(direct["visual_anomaly_score"], abs=1e-7)
    assert body["threshold"] == direct["threshold"]
    assert body["anomaly_detected"] == direct["anomaly_detected"]
    assert body["quality_status"] == direct["quality_status"]
    assert np.allclose(body["raw_anomaly_map_16x16"], direct["raw_anomaly_map"], atol=1e-7)


def test_normal_and_anomalous_examples_have_frozen_decisions(client):
    normal = client.post("/predict/visual-quality", files={"file": (NORMAL.name, NORMAL.read_bytes(), "image/png")}).json()
    anomalous = client.post("/predict/visual-quality", files={"file": (ANOMALOUS.name, ANOMALOUS.read_bytes(), "image/png")}).json()
    assert normal["anomaly_detected"] is False
    assert anomalous["anomaly_detected"] is True


def test_repeated_request_is_deterministic(client):
    files = {"file": ("part.png", image_bytes(), "image/png")}
    first = client.post("/predict/visual-quality", files=files)
    second = client.post("/predict/visual-quality", files={"file": ("part.png", image_bytes(), "image/png")})
    assert first.status_code == second.status_code == 200 and first.json() == second.json()


@pytest.mark.parametrize(("files", "status"), [
    (None, 422),
    ({"file": ("empty.png", b"", "image/png")}, 422),
    ({"file": ("part.gif", image_bytes(), "image/gif")}, 422),
    ({"file": ("part.png", image_bytes(), "application/octet-stream")}, 422),
    ({"file": ("part.png", b"not an image", "image/png")}, 422),
    ({"file": ("part.jpg", image_bytes(), "image/jpeg")}, 422),
    ({"file": ("part.png", image_bytes(), "image/jpeg")}, 422),
    ({"file": ("tiny.png", image_bytes(size=(31, 32)), "image/png")}, 422),
    ({"file": ("huge.png", b"x" * (VISUAL_MAX_UPLOAD_BYTES + 1), "image/png")}, 413),
])
def test_invalid_uploads_are_rejected(client, files, status):
    response = client.post("/predict/visual-quality", files=files) if files is not None else client.post("/predict/visual-quality")
    assert response.status_code == status


def test_multiple_files_are_rejected(client):
    response = client.post(
        "/predict/visual-quality",
        files=[
            ("file", ("one.png", image_bytes(), "image/png")),
            ("extra", ("two.png", image_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 422


def test_unexpected_inference_error_is_logged_and_redacted(client, monkeypatch, caplog):
    def fail(*args, **kwargs): raise RuntimeError("secret backend detail")
    monkeypatch.setattr(visual_prediction_service, "predict_visual_anomaly", fail)
    with caplog.at_level(logging.ERROR):
        response = client.post("/predict/visual-quality", files={"file": ("part.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert response.json() == {"detail": "Visual quality prediction could not be completed."}
    assert "secret backend detail" not in response.text
    assert "Unexpected visual-quality inference error" in caplog.text


def test_visual_resource_unavailable_returns_503(client):
    original = client.app.state.model_resources
    client.app.state.model_resources = replace(original, visual_quality=None)
    try:
        response = client.post(
            "/predict/visual-quality",
            files={"file": ("part.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "Visual quality model resources are unavailable."}
    finally:
        client.app.state.model_resources = original


def test_health_model_info_and_openapi(client):
    health = client.get("/health"); info = client.get("/model/visual-quality/info"); schema = client.get("/openapi.json")
    assert health.status_code == info.status_code == schema.status_code == 200
    assert health.json()["visual_quality_model_loaded"] is True
    for field in ("model_loaded", "failure_model_loaded", "rul_model_loaded", "anomaly_model_loaded"):
        assert health.json()[field] is True
    body = info.json()
    assert body["category"] == "zipper" and body["coreset_size"] == 2458
    assert body["threshold"] == pytest.approx(0.45468712896108615)
    assert "/predict/visual-quality" in schema.json()["paths"]
    assert "/model/visual-quality/info" in schema.json()["paths"]
