"""Frozen input and validation contract for FactoryMind Anomaly Detection v1."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd


ANOMALY_SENSOR_COLUMNS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
ANOMALY_PREDICTOR_COUNT = 14
PERSISTENCE_WINDOW = 5
PERSISTENCE_REQUIRED_COUNT = 3
NORMAL_REFERENCE_RUL_THRESHOLD = 125
THRESHOLD_QUANTILE = 0.975

TARGET_AND_LABEL_COLUMNS = {
    "rul", "raw_rul", "capped_rul", "official_rul", "target",
    "failure", "failure_label", "machine failure", "twf", "hdf", "pwf",
    "osf", "rnf",
}


def _reject_duplicate_columns(frame: pd.DataFrame) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate columns are not allowed: {duplicates}")


def _reject_forbidden_columns(frame: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in frame.columns}
    forbidden = sorted(normalized.intersection(TARGET_AND_LABEL_COLUMNS))
    if forbidden:
        raise ValueError(f"Target, RUL, or failure-label columns are not accepted: {forbidden}")


def _strict_numeric_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    values = frame.to_numpy(dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in values.ravel()):
        raise ValueError(f"{label} must not contain boolean values.")
    if any(not isinstance(value, Real) for value in values.ravel()):
        raise ValueError(f"{label} must contain numeric values, not numeric strings.")
    numeric = frame.astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} must contain only finite values.")
    return numeric


def validate_anomaly_observation(observation: pd.DataFrame) -> pd.DataFrame:
    """Validate and return one strictly ordered 14-sensor observation."""
    if not isinstance(observation, pd.DataFrame):
        raise TypeError("observation must be a pandas DataFrame.")
    if len(observation) != 1:
        raise ValueError("observation must contain exactly one row.")
    _reject_duplicate_columns(observation)
    _reject_forbidden_columns(observation)
    missing = [column for column in ANOMALY_SENSOR_COLUMNS if column not in observation]
    extras = [column for column in observation if column not in ANOMALY_SENSOR_COLUMNS]
    if missing or extras:
        raise ValueError(f"Anomaly sensor contract mismatch; missing={missing}, extra={extras}")
    return _strict_numeric_frame(
        observation.loc[:, ANOMALY_SENSOR_COLUMNS], "Anomaly sensor values"
    )


def validate_anomaly_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Validate one chronological trajectory without sorting or fabricating rows."""
    if not isinstance(trajectory, pd.DataFrame):
        raise TypeError("trajectory must be a pandas DataFrame.")
    if trajectory.empty:
        raise ValueError("trajectory must contain at least one observation.")
    _reject_duplicate_columns(trajectory)
    _reject_forbidden_columns(trajectory)

    required = ["cycle"] + ANOMALY_SENSOR_COLUMNS
    allowed = set(required + (["unit_id"] if "unit_id" in trajectory else []))
    missing = [column for column in required if column not in trajectory]
    extras = [column for column in trajectory if column not in allowed]
    if missing or extras:
        raise ValueError(f"Anomaly trajectory contract mismatch; missing={missing}, extra={extras}")

    if "unit_id" in trajectory:
        if trajectory["unit_id"].isna().any():
            raise ValueError("unit_id cannot be missing when supplied.")
        if trajectory["unit_id"].nunique(dropna=False) != 1:
            raise ValueError("A trajectory must contain exactly one consistent unit_id.")

    cycle_values = trajectory["cycle"].to_numpy(dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in cycle_values):
        raise ValueError("Cycles must not be boolean values.")
    if any(not isinstance(value, Integral) for value in cycle_values):
        raise ValueError("Cycles must be integers.")
    cycles = np.asarray(cycle_values, dtype=np.int64)
    if (cycles <= 0).any():
        raise ValueError("Cycles must be positive.")
    if len(np.unique(cycles)) != len(cycles):
        raise ValueError("Duplicate cycles are not allowed.")
    if len(cycles) > 1 and not np.all(np.diff(cycles) > 0):
        raise ValueError("Cycles must be supplied in strictly increasing order.")
    if len(cycles) > 1 and not np.all(np.diff(cycles) == 1):
        raise ValueError("Cycles must be consecutive.")

    return _strict_numeric_frame(
        trajectory.loc[:, ANOMALY_SENSOR_COLUMNS], "Anomaly sensor values"
    )


assert len(ANOMALY_SENSOR_COLUMNS) == ANOMALY_PREDICTOR_COUNT
assert not TARGET_AND_LABEL_COLUMNS.intersection(ANOMALY_SENSOR_COLUMNS)
