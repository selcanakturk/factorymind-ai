"""Validated, project-relative loading of FactoryMind ML artifacts."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.anomaly_features import (
    ANOMALY_PREDICTOR_COUNT,
    ANOMALY_SENSOR_COLUMNS,
    NORMAL_REFERENCE_RUL_THRESHOLD,
    PERSISTENCE_REQUIRED_COUNT,
    PERSISTENCE_WINDOW,
    THRESHOLD_QUANTILE,
)
from src.anomaly_pipeline import (
    AnomalyModelBundle,
    FROZEN_ISOLATION_FOREST_HYPERPARAMETERS,
)

from src.features import ENGINEERED_FEATURES, RAW_FEATURES
from src.rul_features import (
    EXCLUDED_SENSORS,
    MINIMUM_FULL_CONTEXT_CYCLES,
    RAW_INPUT_COLUMNS as RUL_RAW_INPUT_COLUMNS,
    RAW_PREDICTOR_COLUMNS as RUL_RAW_PREDICTOR_COLUMNS,
    RUL_CAP,
    RUL_PREDICTOR_COLUMNS,
    TEMPORAL_BASE_SENSORS,
    TEMPORAL_FEATURE_DEFINITIONS,
)
from src.rul_pipeline import FROZEN_RANDOM_FOREST_HYPERPARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_failure_model_v1.joblib"
MODEL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "factorymind_failure_model_v1.metadata.json"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "factorymind_failure_risk_thresholds_v1.json"
)
RUL_MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_rul_model_v1.joblib"
RUL_METADATA_PATH = (
    PROJECT_ROOT / "models" / "factorymind_rul_model_v1.metadata.json"
)
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_anomaly_model_v1.joblib"
ANOMALY_METADATA_PATH = PROJECT_ROOT / "models" / "factorymind_anomaly_model_v1.metadata.json"


class ArtifactLoadError(RuntimeError):
    """Raised when required inference artifacts cannot be loaded safely."""


@dataclass(frozen=True)
class RULModelResources:
    """Validated RUL artifacts loaded once at application startup."""

    model: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnomalyModelResources:
    """Validated anomaly artifacts loaded once at application startup."""

    bundle: AnomalyModelBundle
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelResources:
    """Artifacts loaded once for use throughout the application process."""

    model: Any
    model_metadata: dict[str, Any]
    threshold_metadata: dict[str, Any]
    positive_class_index: int
    rul: RULModelResources | None = None
    anomaly: AnomalyModelResources | None = None

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


def _require_finite_number(
    metadata: dict[str, Any], field: str, artifact_name: str
) -> float:
    value = metadata.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ArtifactLoadError(
            f"{artifact_name} field {field!r} must be a finite number."
        )
    return float(value)


def _validate_rul_metadata(metadata: dict[str, Any]) -> None:
    artifact_name = "RUL model metadata"
    expected_strings = {
        "model_name": "FactoryMind RUL FD001 Random Forest",
        "model_version": "1.0.0",
        "model_family": "RandomForestRegressor",
        "dataset": "NASA C-MAPSS",
        "dataset_subset": "FD001",
        "target": "capped_rul",
    }
    for field, expected in expected_strings.items():
        value = _require_nonempty_string(metadata, field, artifact_name)
        if value != expected:
            raise ArtifactLoadError(
                f"RUL metadata field {field!r} is incompatible with the frozen specification."
            )
    for field in [
        "short_history_behavior",
        "output_interpretation",
        "warning",
        "disclaimer",
    ]:
        _require_nonempty_string(metadata, field, artifact_name)

    list_contracts = {
        "raw_input_columns": RUL_RAW_INPUT_COLUMNS,
        "raw_predictor_columns": RUL_RAW_PREDICTOR_COLUMNS,
        "predictor_columns": RUL_PREDICTOR_COLUMNS,
        "excluded_sensors": EXCLUDED_SENSORS,
        "temporal_base_sensors": TEMPORAL_BASE_SENSORS,
    }
    for field, expected in list_contracts.items():
        actual = _require_string_list(metadata, field, artifact_name)
        if actual != expected:
            raise ArtifactLoadError(
                f"RUL metadata {field} does not match the production feature contract."
            )

    expected_numbers = {
        "rul_cap": RUL_CAP,
        "predictor_count": len(RUL_PREDICTOR_COLUMNS),
        "minimum_full_context_cycles": MINIMUM_FULL_CONTEXT_CYCLES,
        "training_unit_count": 100,
        "training_row_count": 20_631,
    }
    for field, expected in expected_numbers.items():
        value = _require_finite_number(metadata, field, artifact_name)
        if value != expected:
            raise ArtifactLoadError(
                f"RUL metadata field {field!r} does not match the frozen value {expected}."
            )

    definitions = metadata.get("temporal_feature_definitions")
    if definitions != TEMPORAL_FEATURE_DEFINITIONS:
        raise ArtifactLoadError(
            "RUL temporal feature definitions do not match production source."
        )
    hyperparameters = metadata.get("frozen_hyperparameters")
    if hyperparameters != FROZEN_RANDOM_FOREST_HYPERPARAMETERS:
        raise ArtifactLoadError(
            "RUL frozen hyperparameters do not match production source."
        )

    limitations = _require_string_list(metadata, "known_limitations", artifact_name)
    if not limitations:
        raise ArtifactLoadError("RUL known_limitations must not be empty.")

    for metrics_field in [
        "groupkfold_metrics_notebook_08",
        "official_fd001_endpoint_metrics_notebook_09",
        "near_failure_metrics",
    ]:
        metrics = metadata.get(metrics_field)
        if not isinstance(metrics, dict) or not metrics:
            raise ArtifactLoadError(
                f"RUL metadata field {metrics_field!r} must be a non-empty object."
            )

    package_versions = metadata.get("package_versions")
    if not isinstance(package_versions, dict):
        raise ArtifactLoadError("RUL package_versions must be an object.")
    runtime_versions = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    for package, runtime_version in runtime_versions.items():
        recorded = package_versions.get(package)
        if not isinstance(recorded, str) or not recorded:
            raise ArtifactLoadError(f"RUL package version {package!r} is missing.")
        if recorded != runtime_version:
            raise ArtifactLoadError(
                f"RUL artifact {package} version {recorded!r} is incompatible with runtime {runtime_version!r}."
            )


def _validate_rul_model(model: Any) -> None:
    if not isinstance(model, Pipeline) or not callable(getattr(model, "predict", None)):
        raise ArtifactLoadError("Loaded RUL artifact must be a fitted sklearn Pipeline.")
    if list(getattr(model, "feature_names_in_", [])) != RUL_PREDICTOR_COLUMNS:
        raise ArtifactLoadError(
            "Loaded RUL model feature_names_in_ do not match the frozen predictor contract."
        )
    estimator = model.named_steps.get("model")
    if not isinstance(estimator, RandomForestRegressor):
        raise ArtifactLoadError(
            "Loaded RUL model family is incompatible with RandomForestRegressor."
        )
    for name, expected in FROZEN_RANDOM_FOREST_HYPERPARAMETERS.items():
        if estimator.get_params().get(name) != expected:
            raise ArtifactLoadError(
                f"Loaded RUL model parameter {name!r} does not match the frozen specification."
            )


def load_rul_resources() -> RULModelResources:
    """Load and validate the frozen RUL model and metadata."""
    if not RUL_MODEL_PATH.is_file():
        raise ArtifactLoadError(
            f"Required RUL model artifact is missing: {RUL_MODEL_PATH}"
        )
    try:
        model = joblib.load(RUL_MODEL_PATH)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Could not load RUL model artifact: {RUL_MODEL_PATH}"
        ) from exc
    metadata = _load_json(RUL_METADATA_PATH, "RUL model metadata")
    _validate_rul_metadata(metadata)
    _validate_rul_model(model)
    return RULModelResources(model=model, metadata=metadata)


def _validate_anomaly_metadata(metadata: dict[str, Any]) -> None:
    artifact_name = "Anomaly model metadata"
    expected_strings = {
        "model_name": "FactoryMind Anomaly FD001 Isolation Forest",
        "model_version": "1.0.0",
        "model_family": "IsolationForest",
        "dataset": "NASA C-MAPSS",
        "dataset_subset": "FD001",
    }
    for field, expected in expected_strings.items():
        if _require_nonempty_string(metadata, field, artifact_name) != expected:
            raise ArtifactLoadError(f"Anomaly metadata field {field!r} is incompatible with the frozen specification.")
    for field in [
        "normal_reference_definition", "scaler_type", "score_direction",
        "percentile_mapping_method", "sensor_deviation_method",
        "output_interpretation", "warning", "disclaimer",
    ]:
        _require_nonempty_string(metadata, field, artifact_name)
    if _require_string_list(metadata, "predictor_columns", artifact_name) != ANOMALY_SENSOR_COLUMNS:
        raise ArtifactLoadError("Anomaly metadata predictor_columns do not match the frozen sensor contract.")
    expected_numbers = {
        "predictor_count": ANOMALY_PREDICTOR_COUNT,
        "normal_reference_rul_threshold": NORMAL_REFERENCE_RUL_THRESHOLD,
        "normal_reference_row_count": 8_031,
        "normal_reference_unit_count": 100,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "persistence_window": PERSISTENCE_WINDOW,
        "persistence_required_count": PERSISTENCE_REQUIRED_COUNT,
        "minimum_persistence_history": PERSISTENCE_WINDOW,
    }
    for field, expected in expected_numbers.items():
        if _require_finite_number(metadata, field, artifact_name) != expected:
            raise ArtifactLoadError(f"Anomaly metadata field {field!r} does not match the frozen value {expected}.")
    threshold = _require_finite_number(metadata, "threshold_raw_score", artifact_name)
    if threshold <= 0:
        raise ArtifactLoadError("Anomaly threshold_raw_score must be positive.")
    if metadata.get("isolation_forest_hyperparameters") != FROZEN_ISOLATION_FOREST_HYPERPARAMETERS:
        raise ArtifactLoadError("Anomaly Isolation Forest hyperparameters do not match production source.")
    score_summary = metadata.get("score_distribution_summary")
    if not isinstance(score_summary, dict):
        raise ArtifactLoadError("Anomaly score_distribution_summary must be an object.")
    for field in ["min", "median", "mean", "std", "p90", "p95", "p97_5", "p99", "max"]:
        _require_finite_number(score_summary, field, "Anomaly score distribution")
    if not math.isclose(float(score_summary["p97_5"]), threshold, rel_tol=0, abs_tol=1e-12):
        raise ArtifactLoadError("Anomaly score summary and threshold_raw_score do not match.")
    limitations = _require_string_list(metadata, "known_limitations", artifact_name)
    if not limitations:
        raise ArtifactLoadError("Anomaly known_limitations must not be empty.")
    for field in ["repeated_split_stability_notebook_12", "alert_rate_stability_notebook_12", "persistence_diagnostics_notebook_12", "lead_time_findings_notebook_12"]:
        if not isinstance(metadata.get(field), dict) or not metadata[field]:
            raise ArtifactLoadError(f"Anomaly metadata field {field!r} must be a non-empty object.")
    package_versions = metadata.get("package_versions")
    if not isinstance(package_versions, dict):
        raise ArtifactLoadError("Anomaly package_versions must be an object.")
    runtime_versions = {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}
    for package, runtime in runtime_versions.items():
        recorded = package_versions.get(package)
        if not isinstance(recorded, str) or not recorded:
            raise ArtifactLoadError(f"Anomaly package version {package!r} is missing.")
        if recorded != runtime:
            raise ArtifactLoadError(f"Anomaly artifact {package} version {recorded!r} is incompatible with runtime {runtime!r}.")


def _validate_anomaly_bundle(bundle: Any, metadata: dict[str, Any]) -> None:
    if not isinstance(bundle, AnomalyModelBundle):
        raise ArtifactLoadError("Loaded anomaly artifact must be an AnomalyModelBundle.")
    if not isinstance(bundle.scaler, StandardScaler) or not hasattr(bundle.scaler, "mean_"):
        raise ArtifactLoadError("Loaded anomaly bundle is missing a fitted StandardScaler.")
    if not isinstance(bundle.detector, IsolationForest) or not hasattr(bundle.detector, "estimators_"):
        raise ArtifactLoadError("Loaded anomaly bundle is missing a fitted IsolationForest.")
    if int(getattr(bundle.scaler, "n_features_in_", -1)) != ANOMALY_PREDICTOR_COUNT:
        raise ArtifactLoadError("Anomaly scaler predictor count is incompatible.")
    if int(getattr(bundle.detector, "n_features_in_", -1)) != ANOMALY_PREDICTOR_COUNT:
        raise ArtifactLoadError("Anomaly detector predictor count is incompatible.")
    for name, expected in FROZEN_ISOLATION_FOREST_HYPERPARAMETERS.items():
        if bundle.detector.get_params().get(name) != expected:
            raise ArtifactLoadError(f"Loaded anomaly detector parameter {name!r} does not match the frozen specification.")
    scores = np.asarray(bundle.sorted_normal_scores)
    if scores.shape != (8_031,) or not np.isfinite(scores).all() or np.any(scores[1:] < scores[:-1]):
        raise ArtifactLoadError("Anomaly reference-score distribution is malformed.")
    expected_threshold = float(np.quantile(scores, THRESHOLD_QUANTILE))
    if not math.isclose(bundle.raw_threshold, expected_threshold, rel_tol=0, abs_tol=1e-12):
        raise ArtifactLoadError("Anomaly bundle threshold is inconsistent with reference scores.")
    if not math.isclose(bundle.raw_threshold, float(metadata["threshold_raw_score"]), rel_tol=0, abs_tol=1e-12):
        raise ArtifactLoadError("Anomaly bundle and metadata thresholds do not match.")
    if bundle.threshold_quantile != THRESHOLD_QUANTILE or bundle.model_version != metadata["model_version"]:
        raise ArtifactLoadError("Anomaly bundle version or threshold quantile is incompatible.")


def load_anomaly_resources() -> AnomalyModelResources:
    if not ANOMALY_MODEL_PATH.is_file():
        raise ArtifactLoadError(f"Required anomaly model artifact is missing: {ANOMALY_MODEL_PATH}")
    try:
        bundle = joblib.load(ANOMALY_MODEL_PATH)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load anomaly model artifact: {ANOMALY_MODEL_PATH}") from exc
    metadata = _load_json(ANOMALY_METADATA_PATH, "anomaly model metadata")
    _validate_anomaly_metadata(metadata)
    _validate_anomaly_bundle(bundle, metadata)
    return AnomalyModelResources(bundle=bundle, metadata=metadata)


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
    rul_resources = load_rul_resources()
    anomaly_resources = load_anomaly_resources()

    return ModelResources(
        model=model,
        model_metadata=model_metadata,
        threshold_metadata=threshold_metadata,
        positive_class_index=positive_class_index,
        rul=rul_resources,
        anomaly=anomaly_resources,
    )
