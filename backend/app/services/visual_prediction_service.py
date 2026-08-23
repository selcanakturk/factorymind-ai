"""Guarded in-memory visual-quality inference and map rendering."""

from __future__ import annotations

import base64
from io import BytesIO
import logging

import numpy as np
from PIL import Image

from src.visual_pipeline import predict_visual_anomaly

from ..core.model_loader import VisualModelResources
from ..schemas import VisualQualityPredictionResponse


logger = logging.getLogger(__name__)


class VisualInputError(ValueError):
    """Raised when source image validation rejects the uploaded content."""


class VisualPredictionError(RuntimeError):
    """Raised when validated visual inference fails unexpectedly."""


def _map_png_base64(normalized_map: np.ndarray) -> str:
    values = np.asarray(normalized_map, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("Normalized model anomaly map is invalid.")
    pixels = np.rint(values * 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(pixels, mode="L").save(output, format="PNG", optimize=False)
    return base64.b64encode(output.getvalue()).decode("ascii")


class VisualPredictionService:
    def __init__(self, resources: VisualModelResources):
        self.resources = resources

    def predict(self, payload: bytes, filename: str) -> VisualQualityPredictionResponse:
        try:
            # PyTorch/MPS and the shared sklearn index are guarded only for visual requests.
            with self.resources.inference_lock:
                result = predict_visual_anomaly(
                    self.resources.runtime, payload, filename=filename
                )
            map_base64 = _map_png_base64(result["normalized_anomaly_map"])
        except (ValueError, TypeError) as exc:
            raise VisualInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected visual-quality inference error")
            raise VisualPredictionError("The validated image could not be scored.") from exc
        try:
            return VisualQualityPredictionResponse(
                visual_anomaly_score=result["visual_anomaly_score"],
                threshold=result["threshold"],
                threshold_quantile=result["threshold_quantile"],
                anomaly_detected=result["anomaly_detected"],
                quality_status=result["quality_status"],
                category=result["category"],
                model_version=result["model_version"],
                dataset=result["dataset"],
                original_width=result["input_width"],
                original_height=result["input_height"],
                model_input_width=result["model_input_size"][0],
                model_input_height=result["model_input_size"][1],
                anomaly_map_available=True,
                raw_anomaly_map_16x16=result["raw_anomaly_map"].tolist(),
                anomaly_map_image_base64=map_base64,
                anomaly_map_label="Model anomaly map",
                warning=result["warning"],
                disclaimer=result["disclaimer"],
            )
        except Exception as exc:
            logger.exception("Malformed visual-quality inference output")
            raise VisualPredictionError("The validated image could not be scored.") from exc
