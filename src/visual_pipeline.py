"""Offline-loadable FactoryMind Visual Quality Inspection v1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO

import joblib
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F

from .visual_features import (
    VISUAL_CATEGORY, VISUAL_EMBEDDING_DIM, VISUAL_INPUT_SIZE, VISUAL_PATCH_COUNT,
    VISUAL_PATCH_GRID, VisualFeatureExtractor, build_visual_extractor,
    extract_patch_embeddings, load_visual_image, preprocess_visual_image,
)


VISUAL_MODEL_VERSION = "visual-quality-v1"
VISUAL_THRESHOLD_QUANTILE = 0.975
QUALITY_NORMAL = "No visual anomaly detected"
QUALITY_ANOMALOUS = "Visual anomaly detected"
VISUAL_WARNING = (
    "Visual anomalies indicate appearance differences from the learned normal reference "
    "and do not by themselves identify defect cause or severity."
)
VISUAL_DISCLAIMER = (
    "Development-stage normal-only model for the MVTec AD zipper research category under "
    "noncommercial dataset licensing constraints, without real-factory external validation "
    "or calibrated probability. This is not a certified industrial quality-control decision."
)


@dataclass
class VisualModelBundle:
    backbone_state_dict: dict[str, torch.Tensor]
    coreset: np.ndarray
    threshold: float
    display_low: float
    display_high: float
    config: dict[str, Any]
    threshold_quantile: float = VISUAL_THRESHOLD_QUANTILE
    model_version: str = VISUAL_MODEL_VERSION
    category: str = VISUAL_CATEGORY


@dataclass
class VisualRuntime:
    bundle: VisualModelBundle
    extractor: VisualFeatureExtractor
    nearest_neighbors: NearestNeighbors
    device: torch.device


def threshold_exceeded(score: float | np.ndarray, threshold: float) -> np.ndarray:
    """Strict exceedance: a score exactly equal to the threshold is not anomalous."""
    return np.asarray(score, dtype=float) > float(threshold)


def normalize_anomaly_map(raw_map: np.ndarray, display_low: float, display_high: float) -> np.ndarray:
    values = np.asarray(raw_map, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Raw anomaly map must contain finite values.")
    if not np.isfinite(display_low) or not np.isfinite(display_high) or display_high <= display_low:
        raise ValueError("Display scale requires finite high > low.")
    return np.clip((values - display_low) / (display_high - display_low), 0.0, 1.0).astype(np.float32)


def upsample_anomaly_map(raw_map: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    values = np.asarray(raw_map, dtype=np.float32)
    if values.shape != VISUAL_PATCH_GRID or not np.isfinite(values).all():
        raise ValueError("Raw anomaly map must be finite with shape (16, 16).")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output dimensions must be positive.")
    tensor = torch.from_numpy(values)[None, None]
    return F.interpolate(
        tensor, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0, 0].numpy().astype(np.float32)


def validate_visual_bundle(bundle: VisualModelBundle) -> None:
    if not isinstance(bundle, VisualModelBundle):
        raise TypeError("Artifact is not a VisualModelBundle.")
    if bundle.coreset.shape != (2458, VISUAL_EMBEDDING_DIM) or bundle.coreset.dtype != np.float32:
        raise ValueError("Artifact coreset violates the frozen (2458, 384) float32 contract.")
    if not np.isfinite(bundle.coreset).all():
        raise ValueError("Artifact coreset contains non-finite values.")
    if not np.isfinite(bundle.threshold) or bundle.threshold <= 0:
        raise ValueError("Artifact threshold must be finite and positive.")
    if bundle.display_high != bundle.threshold or bundle.display_high <= bundle.display_low:
        raise ValueError("Artifact display scale violates the frozen threshold-aware contract.")
    if bundle.threshold_quantile != VISUAL_THRESHOLD_QUANTILE:
        raise ValueError("Artifact threshold quantile violates the frozen contract.")


def create_visual_runtime(
    bundle: VisualModelBundle, *, device: str | torch.device = "cpu",
) -> VisualRuntime:
    validate_visual_bundle(bundle)
    resolved = torch.device(device)
    extractor = build_visual_extractor(
        state_dict=bundle.backbone_state_dict, pretrained=False, device=resolved
    )
    nearest = NearestNeighbors(
        n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=-1
    ).fit(bundle.coreset)
    return VisualRuntime(bundle, extractor, nearest, resolved)


def load_visual_runtime(
    artifact_path: str | Path, *, device: str | torch.device = "cpu",
) -> VisualRuntime:
    return create_visual_runtime(joblib.load(artifact_path), device=device)


def score_patch_embeddings(runtime: VisualRuntime, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(patches, dtype=np.float32)
    if values.shape != (VISUAL_PATCH_COUNT, VISUAL_EMBEDDING_DIM) or not np.isfinite(values).all():
        raise ValueError("Patch matrix must be finite with shape (256, 384).")
    distances = runtime.nearest_neighbors.kneighbors(values, return_distance=True)[0][:, 0].astype(np.float32)
    if distances.shape != (VISUAL_PATCH_COUNT,) or not np.isfinite(distances).all() or (distances < 0).any():
        raise RuntimeError("Nearest-neighbor patch distances violate score invariants.")
    raw_map = distances.reshape(VISUAL_PATCH_GRID)
    image_score = float(distances.max())
    if image_score != float(np.max(raw_map)):
        raise RuntimeError("Image score must equal the maximum raw patch score.")
    return distances, raw_map, image_score


def predict_visual_anomaly(
    runtime: VisualRuntime,
    source: str | Path | bytes | bytearray | BinaryIO | Image.Image,
    *, filename: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    image = load_visual_image(source, filename=filename)
    width, height = image.size
    tensor = preprocess_visual_image(image)[None]
    patches = extract_patch_embeddings(runtime.extractor, tensor, device=runtime.device)[0]
    distances, raw_map, image_score = score_patch_embeddings(runtime, patches)
    detected = bool(threshold_exceeded(image_score, runtime.bundle.threshold))
    upsampled_raw = upsample_anomaly_map(raw_map, height, width)
    normalized_map = normalize_anomaly_map(
        upsampled_raw, runtime.bundle.display_low, runtime.bundle.display_high
    )
    return {
        "visual_anomaly_score": image_score,
        "threshold": float(runtime.bundle.threshold),
        "threshold_quantile": float(runtime.bundle.threshold_quantile),
        "anomaly_detected": detected,
        "quality_status": QUALITY_ANOMALOUS if detected else QUALITY_NORMAL,
        "patch_scores": distances,
        "raw_anomaly_map": raw_map,
        "upsampled_raw_anomaly_map": upsampled_raw,
        "normalized_anomaly_map": normalized_map,
        "anomaly_map_available": True,
        "input_width": width,
        "input_height": height,
        "model_input_size": list(VISUAL_INPUT_SIZE),
        "model_version": runtime.bundle.model_version,
        "dataset": "MVTec AD",
        "category": runtime.bundle.category,
        "warning": VISUAL_WARNING,
        "disclaimer": VISUAL_DISCLAIMER,
        "inference_seconds": perf_counter() - started,
    }
