"""Validated, project-relative loading of FactoryMind ML artifacts."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features import ENGINEERED_FEATURES, RAW_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_failure_model_v1.joblib"
MODEL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "factorymind_failure_model_v1.metadata.json"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "factorymind_failure_risk_thresholds_v1.json"
)


class ArtifactLoadError(RuntimeError):
    """Raised when required inference artifacts cannot be loaded safely."""


@dataclass(frozen=True)
class ModelResources:
    """Artifacts loaded once for use throughout the application process."""

    model: Any
    model_metadata: dict[str, Any]
    threshold_metadata: dict[str, Any]
    positive_class_index: int

    @property
    def thresholds(self) -> dict[str, float]:
        return self.threshold_metadata["thresholds"]


def _load_json(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactLoadError(f"Required {artifact_name} artifact is missing: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(f"Could not read valid {artifact_name} JSON: {path}") from exc
    if not isinstance(content, dict):
        raise ArtifactLoadError(f"{artifact_name} artifact must contain a JSON object.")
    return content


def _require_nonempty_string(
    metadata: dict[str, Any], field: str, artifact_name: str
) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactLoadError(
            f"{artifact_name} field {field!r} must be a non-empty string."
        )
    return value


def _require_string_list(
    metadata: dict[str, Any], field: str, artifact_name: str
) -> list[str]:
    value = metadata.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ArtifactLoadError(
            f"{artifact_name} field {field!r} must be a list of strings."
        )
    return value


def _validate_model_metadata(model_metadata: dict[str, Any]) -> None:
    artifact_name = "Model metadata"
    for field in [
        "model_version",
        "model_family",
        "calibration_method",
        "target",
        "output_interpretation",
        "methodological_warning",
    ]:
        _require_nonempty_string(model_metadata, field, artifact_name)

    raw_features = _require_string_list(
        model_metadata, "raw_input_features", artifact_name
    )
    if raw_features != RAW_FEATURES:
        raise ArtifactLoadError(
            "Model metadata raw_input_features do not match the production raw schema."
        )

    engineered_features = _require_string_list(
        model_metadata, "engineered_features", artifact_name
    )
    if engineered_features != ENGINEERED_FEATURES:
        raise ArtifactLoadError(
            "Model metadata engineered_features do not match the production schema."
        )

    metrics = model_metadata.get("evaluation_metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ArtifactLoadError(
            "Model metadata field 'evaluation_metrics' must be a non-empty object."
        )
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ArtifactLoadError(
                "Model evaluation_metrics must map string names to finite numbers."
            )


def _validate_thresholds(
    threshold_metadata: dict[str, Any], model_metadata: dict[str, Any]
) -> dict[str, float]:
    artifact_name = "Risk threshold metadata"
    for field in ["threshold_version", "score_scale", "calibration_method"]:
        _require_nonempty_string(threshold_metadata, field, artifact_name)

    thresholds = threshold_metadata.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ArtifactLoadError(
            "Threshold artifact field 'thresholds' must be a JSON object."
        )

    normalized_thresholds: dict[str, float] = {}
    for name in ["medium", "high", "critical"]:
        value = thresholds.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ArtifactLoadError(
                f"Threshold {name!r} must be a finite JSON number."
            )
        normalized_thresholds[name] = float(value)

    medium = normalized_thresholds["medium"]
    high = normalized_thresholds["high"]
    critical = normalized_thresholds["critical"]

    if not 0 < medium < high < critical < 1:
        raise ArtifactLoadError(
            "Calibrated thresholds must satisfy 0 < medium < high < critical < 1."
        )

    threshold_method = threshold_metadata.get("calibration_method")
    model_method = model_metadata.get("calibration_method")
    if threshold_method != model_method:
        raise ArtifactLoadError(
            "Model and threshold calibration methods do not match: "
            f"{model_method!r} vs {threshold_method!r}."
        )

    policy_objectives = threshold_metadata.get("policy_objectives")
    if not isinstance(policy_objectives, dict) or any(
        not isinstance(policy_objectives.get(name), str)
        or not policy_objectives[name]
        for name in ["medium", "high", "critical"]
    ):
        raise ArtifactLoadError(
            "Threshold policy_objectives must define medium, high, and critical text."
        )

    return normalized_thresholds


def _resolve_positive_class_index(model: Any) -> int:
    if not hasattr(model, "classes_"):
        raise ArtifactLoadError("Loaded model does not expose fitted classes_.")

    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or len(classes) != 2:
        raise ArtifactLoadError(
            "Loaded model must expose exactly two one-dimensional class labels."
        )

    positive_indices = np.flatnonzero(classes == 1)
    negative_indices = np.flatnonzero(classes == 0)
    if len(positive_indices) != 1:
        raise ArtifactLoadError("Loaded model does not contain positive class label 1.")
    if len(negative_indices) != 1:
        raise ArtifactLoadError(
            "Loaded model classes are incompatible with expected binary labels 0 and 1."
        )
    return int(positive_indices[0])


def load_model_resources() -> ModelResources:
    """Load and validate all inference artifacts from the project model directory."""
    if not MODEL_PATH.is_file():
        raise ArtifactLoadError(f"Required model artifact is missing: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load model artifact: {MODEL_PATH}") from exc

    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactLoadError("Loaded model does not provide predict_proba().")

    model_metadata = _load_json(MODEL_METADATA_PATH, "model metadata")
    threshold_metadata = _load_json(THRESHOLD_PATH, "risk threshold")
    _validate_model_metadata(model_metadata)
    normalized_thresholds = _validate_thresholds(
        threshold_metadata, model_metadata
    )
    threshold_metadata = {
        **threshold_metadata,
        "thresholds": normalized_thresholds,
    }
    positive_class_index = _resolve_positive_class_index(model)

    return ModelResources(
        model=model,
        model_metadata=model_metadata,
        threshold_metadata=threshold_metadata,
        positive_class_index=positive_class_index,
    )
