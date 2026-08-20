"""Frozen FactoryMind Anomaly Detection v1 fitting and inference behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .anomaly_features import (
    ANOMALY_SENSOR_COLUMNS,
    PERSISTENCE_REQUIRED_COUNT,
    PERSISTENCE_WINDOW,
    THRESHOLD_QUANTILE,
    validate_anomaly_observation,
    validate_anomaly_trajectory,
)


MODEL_VERSION = "1.0.0"
DATASET_NAME = "NASA C-MAPSS FD001"
SENSOR_CONTEXT_LABEL = "Most unusual sensor readings relative to normal reference."
ANOMALY_WARNING = (
    "Anomaly does not mean failure. Unusual sensor behavior may reflect degradation, "
    "operating variation, noise, or other causes and should be investigated in context."
)
ANOMALY_DISCLAIMER = (
    "Development-stage condition-monitoring estimate based on simulated NASA C-MAPSS "
    "FD001 data with one operating condition and one fault mode, a retrospectively "
    "defined healthy reference, and no externally validated anomaly labels. The anomaly "
    "score is not a probability. This is not an industrial safety decision."
)
FROZEN_ISOLATION_FOREST_HYPERPARAMETERS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class AnomalyModelBundle:
    """All fitted state required for deterministic Anomaly v1 inference."""

    scaler: StandardScaler
    detector: IsolationForest
    sorted_normal_scores: np.ndarray
    raw_threshold: float
    threshold_quantile: float = THRESHOLD_QUANTILE
    model_version: str = MODEL_VERSION


def build_anomaly_scaler() -> StandardScaler:
    return StandardScaler()


def build_anomaly_detector() -> IsolationForest:
    return IsolationForest(**FROZEN_ISOLATION_FOREST_HYPERPARAMETERS)


def native_to_anomaly_score(native_scores: np.ndarray) -> np.ndarray:
    """Reverse sklearn's normality direction so higher means more anomalous."""
    values = np.asarray(native_scores, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Isolation Forest returned non-finite scores.")
    return -values


def fit_anomaly_bundle(normal_reference: pd.DataFrame) -> AnomalyModelBundle:
    """Fit the frozen scaler and detector on validated normal-reference sensors."""
    if list(normal_reference.columns) != ANOMALY_SENSOR_COLUMNS:
        raise ValueError("Normal-reference columns must match the frozen sensor order.")
    if normal_reference.empty:
        raise ValueError("Normal-reference data cannot be empty.")
    numeric = normal_reference.astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Normal-reference values must be finite.")
    scaler = build_anomaly_scaler().fit(numeric)
    standardized = scaler.transform(numeric)
    detector = build_anomaly_detector().fit(standardized)
    scores = native_to_anomaly_score(detector.score_samples(standardized))
    sorted_scores = np.sort(scores)
    threshold = float(np.quantile(scores, THRESHOLD_QUANTILE))
    return AnomalyModelBundle(scaler, detector, sorted_scores, threshold)


def empirical_percentile(
    raw_scores: np.ndarray | list[float] | float,
    sorted_reference_scores: np.ndarray,
) -> np.ndarray:
    """Map scores with the right-inclusive frozen empirical-CDF rule."""
    values = np.atleast_1d(np.asarray(raw_scores, dtype=float))
    reference = np.asarray(sorted_reference_scores, dtype=float)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Reference scores must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Scores and reference scores must be finite.")
    if np.any(reference[1:] < reference[:-1]):
        raise ValueError("Reference scores must be sorted in ascending order.")
    ranks = np.searchsorted(reference, values, side="right")
    return np.clip(100.0 * ranks / reference.size, 0.0, 100.0)


def threshold_exceeded(raw_scores: np.ndarray | float, raw_threshold: float) -> np.ndarray:
    """Apply strict exceedance; equality with the boundary is not exceeded."""
    return np.asarray(raw_scores, dtype=float) > float(raw_threshold)


def score_standardized(bundle: AnomalyModelBundle, sensors: pd.DataFrame) -> np.ndarray:
    standardized = bundle.scaler.transform(sensors.loc[:, ANOMALY_SENSOR_COLUMNS])
    return native_to_anomaly_score(bundle.detector.score_samples(standardized))


def sensor_deviation_context(
    bundle: AnomalyModelBundle, observation: pd.DataFrame, *, top_n: int = 5
) -> list[dict[str, Any]]:
    """Return deterministic largest standardized deviations for one observation."""
    if not 1 <= top_n <= len(ANOMALY_SENSOR_COLUMNS):
        raise ValueError("top_n must be between 1 and the predictor count.")
    values = observation.loc[:, ANOMALY_SENSOR_COLUMNS].iloc[0].to_numpy(float)
    scales = np.asarray(bundle.scaler.scale_, dtype=float)
    safe_scales = np.where(scales == 0, 1.0, scales)
    deviations = (values - bundle.scaler.mean_) / safe_scales
    order = sorted(
        range(len(deviations)), key=lambda index: (-abs(deviations[index]), index)
    )[:top_n]
    return [
        {
            "sensor": ANOMALY_SENSOR_COLUMNS[index],
            "current_value": float(values[index]),
            "reference_mean": float(bundle.scaler.mean_[index]),
            "standardized_deviation": float(deviations[index]),
            "direction": "above_normal" if deviations[index] >= 0 else "below_normal",
        }
        for index in order
    ]


def _common_result(bundle: AnomalyModelBundle, sensors: pd.DataFrame, top_n: int) -> dict[str, Any]:
    scores = score_standardized(bundle, sensors)
    latest_score = float(scores[-1])
    percentile = float(empirical_percentile(latest_score, bundle.sorted_normal_scores)[0])
    return {
        "current_anomaly_score": latest_score,
        "anomaly_percentile": percentile,
        "threshold_percentile": bundle.threshold_quantile * 100.0,
        "raw_threshold": float(bundle.raw_threshold),
        "current_threshold_exceeded": bool(threshold_exceeded(latest_score, bundle.raw_threshold)),
        "top_sensor_deviations": sensor_deviation_context(bundle, sensors.iloc[[-1]], top_n=top_n),
        "sensor_context_label": SENSOR_CONTEXT_LABEL,
        "model_version": bundle.model_version,
        "dataset": DATASET_NAME,
        "development_stage": True,
        "warning": ANOMALY_WARNING,
        "disclaimer": ANOMALY_DISCLAIMER,
    }


def score_anomaly_observation(
    bundle: AnomalyModelBundle, observation: pd.DataFrame, *, top_n: int = 5
) -> dict[str, Any]:
    sensors = validate_anomaly_observation(observation)
    return _common_result(bundle, sensors, top_n)


def evaluate_persistence(exceedances: np.ndarray | list[bool]) -> dict[str, Any]:
    values = np.asarray(exceedances, dtype=bool)
    recent = values[-min(PERSISTENCE_WINDOW, len(values)) :]
    sufficient = len(values) >= PERSISTENCE_WINDOW
    count = int(recent.sum())
    return {
        "recent_window_size": len(recent),
        "recent_exceedance_pattern": recent.tolist(),
        "recent_exceedance_count": count,
        "persistence_required_count": PERSISTENCE_REQUIRED_COUNT,
        "persistence_window_size": PERSISTENCE_WINDOW,
        "persistence_status": "available" if sufficient else "insufficient_history",
        "alert_active": bool(sufficient and count >= PERSISTENCE_REQUIRED_COUNT),
    }


def evaluate_anomaly_trajectory(
    bundle: AnomalyModelBundle, trajectory: pd.DataFrame, *, top_n: int = 5
) -> dict[str, Any]:
    sensors = validate_anomaly_trajectory(trajectory)
    scores = score_standardized(bundle, sensors)
    exceedances = threshold_exceeded(scores, bundle.raw_threshold)
    return {
        **_common_result(bundle, sensors, top_n),
        **evaluate_persistence(exceedances),
        "history_cycle_count": len(trajectory),
    }
