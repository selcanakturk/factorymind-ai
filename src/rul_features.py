"""Frozen, backward-looking feature contract for FactoryMind RUL v1."""

from __future__ import annotations

import numpy as np
import pandas as pd


RUL_CAP = 125
MINIMUM_FULL_CONTEXT_CYCLES = 6

OPERATIONAL_SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
EXCLUDED_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
RETAINED_SENSORS = [
    f"sensor_{index}"
    for index in range(1, 22)
    if f"sensor_{index}" not in EXCLUDED_SENSORS
]
TEMPORAL_BASE_SENSORS = [
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_21",
]

# operational_setting_3 is required in the raw trajectory contract for schema
# integrity, but is not a predictor because it is constant in FD001.
RAW_INPUT_COLUMNS = ["cycle"] + OPERATIONAL_SETTING_COLUMNS + RETAINED_SENSORS
RAW_PREDICTOR_COLUMNS = [
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
] + RETAINED_SENSORS


def _temporal_names(sensor: str) -> list[str]:
    return [
        f"{sensor}_lag_1",
        f"{sensor}_lag_5",
        f"{sensor}_rolling_mean_5",
        f"{sensor}_rolling_std_5",
        f"{sensor}_delta_1",
    ]


TEMPORAL_FEATURE_COLUMNS = [
    name for sensor in TEMPORAL_BASE_SENSORS for name in _temporal_names(sensor)
]
RUL_PREDICTOR_COLUMNS = RAW_PREDICTOR_COLUMNS + TEMPORAL_FEATURE_COLUMNS

TARGET_COLUMNS = {
    "rul",
    "raw_rul",
    "capped_rul",
    "official_rul",
    "official_raw_rul",
    "official_capped_rul",
    "target",
}
FAILURE_LABEL_COLUMNS = {
    "machine failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
}


def validate_trajectory(trajectory: pd.DataFrame) -> None:
    """Validate one raw engine trajectory without changing its row order."""
    if not isinstance(trajectory, pd.DataFrame):
        raise TypeError("trajectory must be a pandas DataFrame with named columns.")
    if trajectory.empty:
        raise ValueError("trajectory must contain at least one observation.")

    duplicate_columns = trajectory.columns[trajectory.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate columns are not allowed: {duplicate_columns}")

    missing = [column for column in RAW_INPUT_COLUMNS if column not in trajectory]
    if missing:
        raise ValueError(f"Missing required raw trajectory columns: {missing}")

    normalized_columns = {str(column).strip().lower() for column in trajectory.columns}
    forbidden = sorted(
        normalized_columns.intersection(TARGET_COLUMNS | FAILURE_LABEL_COLUMNS)
    )
    if forbidden:
        raise ValueError(f"Target or failure-label columns are not accepted: {forbidden}")

    if "unit_id" in trajectory:
        if trajectory["unit_id"].isna().any():
            raise ValueError("unit_id cannot be missing when supplied.")
        if trajectory["unit_id"].nunique(dropna=False) != 1:
            raise ValueError("A request must contain exactly one engine trajectory.")

    numeric = trajectory.loc[:, RAW_INPUT_COLUMNS]
    try:
        numeric_values = numeric.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Raw trajectory fields must be numeric.") from exc
    if not np.isfinite(numeric_values).all():
        raise ValueError("Raw trajectory fields must contain only finite values.")

    cycle_values = trajectory["cycle"].to_numpy(dtype=float)
    if not np.equal(cycle_values, np.floor(cycle_values)).all():
        raise ValueError("Cycles must be integers.")
    if (cycle_values <= 0).any():
        raise ValueError("Cycles must be positive.")
    if pd.Series(cycle_values).duplicated().any():
        raise ValueError("Duplicate cycles are not allowed.")
    if len(cycle_values) > 1 and not np.all(np.diff(cycle_values) > 0):
        raise ValueError("Cycles must be supplied in strictly increasing order.")
    if len(cycle_values) > 1 and not np.all(np.diff(cycle_values) == 1):
        raise ValueError("Cycles must be consecutive for the frozen lag semantics.")


def engineer_temporal_features(
    frame: pd.DataFrame, *, unit_column: str = "unit_id"
) -> pd.DataFrame:
    """Add the exact frozen temporal features to one or more ordered units.

    This helper assumes raw values have already been checked. Positive shifts
    and trailing windows ensure that cycle t never uses a future observation.
    Early lag and delta values intentionally remain missing.
    """
    if unit_column not in frame:
        raise ValueError(f"Missing grouping column: {unit_column}")

    transformed = frame.sort_values([unit_column, "cycle"]).copy()
    grouped = transformed.groupby(unit_column, sort=False)
    for sensor in TEMPORAL_BASE_SENSORS:
        lag_1 = f"{sensor}_lag_1"
        transformed[lag_1] = grouped[sensor].shift(1)
        transformed[f"{sensor}_lag_5"] = grouped[sensor].shift(5)
        transformed[f"{sensor}_rolling_mean_5"] = grouped[sensor].transform(
            lambda values: values.rolling(window=5, min_periods=1).mean()
        )
        transformed[f"{sensor}_rolling_std_5"] = grouped[sensor].transform(
            lambda values: values.rolling(
                window=5, min_periods=1
            ).std(ddof=0)
        )
        transformed[f"{sensor}_delta_1"] = transformed[sensor] - transformed[lag_1]
    return transformed


def latest_predictor_row(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Validate a single raw trajectory and return its latest 47 predictors."""
    validate_trajectory(trajectory)
    selected_columns = (["unit_id"] if "unit_id" in trajectory else []) + (
        RAW_INPUT_COLUMNS
    )
    working = trajectory.loc[:, selected_columns].copy()
    if "unit_id" not in working:
        working.insert(0, "unit_id", 1)

    engineered = engineer_temporal_features(working)
    latest = engineered.iloc[[-1]].loc[:, RUL_PREDICTOR_COLUMNS]
    if latest.shape != (1, 47):
        raise RuntimeError("Frozen RUL feature contract produced an invalid shape.")
    return latest


assert len(RAW_PREDICTOR_COLUMNS) == 17
assert len(TEMPORAL_FEATURE_COLUMNS) == 30
assert len(RUL_PREDICTOR_COLUMNS) == 47
