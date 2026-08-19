import numpy as np
import pandas as pd
import pytest

from src.rul_features import (
    EXCLUDED_SENSORS,
    RAW_INPUT_COLUMNS,
    RAW_PREDICTOR_COLUMNS,
    RUL_PREDICTOR_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    engineer_temporal_features,
    latest_predictor_row,
    validate_trajectory,
)


def make_trajectory(length=6, unit_id=1):
    cycle = np.arange(1, length + 1)
    data = {
        "unit_id": np.full(length, unit_id),
        "cycle": cycle,
        "operational_setting_1": np.linspace(0.0, 0.001, length),
        "operational_setting_2": np.linspace(0.0, 0.0001, length),
        "operational_setting_3": np.full(length, 100.0),
    }
    for sensor in [column for column in RAW_INPUT_COLUMNS if column.startswith("sensor_")]:
        sensor_number = int(sensor.split("_")[1])
        data[sensor] = sensor_number * 100.0 + cycle
    return pd.DataFrame(data)


def test_frozen_feature_contract():
    assert len(RAW_PREDICTOR_COLUMNS) == 17
    assert len(TEMPORAL_FEATURE_COLUMNS) == 30
    assert len(RUL_PREDICTOR_COLUMNS) == 47
    assert not set(EXCLUDED_SENSORS).intersection(RUL_PREDICTOR_COLUMNS)
    assert not {"unit_id", "raw_rul", "capped_rul", "RUL"}.intersection(
        RUL_PREDICTOR_COLUMNS
    )


@pytest.mark.parametrize("length", [1, 3, 6])
def test_supported_trajectory_lengths_are_accepted(length):
    trajectory = make_trajectory(length)
    validate_trajectory(trajectory)
    assert latest_predictor_row(trajectory).shape == (1, 47)


def test_feature_formulas_match_frozen_notebook_reference():
    trajectory = make_trajectory(6)
    latest = latest_predictor_row(trajectory).iloc[0]

    # sensor_4 values are 401, 402, ..., 406.
    assert latest["sensor_4_lag_1"] == pytest.approx(405.0)
    assert latest["sensor_4_lag_5"] == pytest.approx(401.0)
    assert latest["sensor_4_rolling_mean_5"] == pytest.approx(404.0)
    assert latest["sensor_4_rolling_std_5"] == pytest.approx(np.sqrt(2.0))
    assert latest["sensor_4_delta_1"] == pytest.approx(1.0)


def test_temporal_features_do_not_cross_unit_boundaries():
    first = make_trajectory(6, unit_id=1)
    second = make_trajectory(6, unit_id=2)
    second["sensor_4"] += 10_000
    combined = pd.concat([first, second], ignore_index=True)

    engineered = engineer_temporal_features(combined)
    first_rows = engineered.groupby("unit_id").head(1)

    assert first_rows["sensor_4_lag_1"].isna().all()
    assert first_rows["sensor_4_lag_5"].isna().all()
    assert first_rows["sensor_4_delta_1"].isna().all()
    assert first_rows.loc[first_rows["unit_id"].eq(2), "sensor_4_rolling_mean_5"].iloc[0] == pytest.approx(10_401.0)


def test_lag_and_rolling_features_are_backward_looking():
    original = make_trajectory(6)
    changed_future = original.copy()
    changed_future.loc[5, "sensor_4"] = 99_999

    original_features = engineer_temporal_features(original)
    changed_features = engineer_temporal_features(changed_future)

    pd.testing.assert_series_equal(
        original_features.loc[:4, "sensor_4_rolling_mean_5"],
        changed_features.loc[:4, "sensor_4_rolling_mean_5"],
    )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda df: pd.concat([df, df.iloc[[1]]], ignore_index=True), "Duplicate cycles"),
        (lambda df: df.drop(index=2).reset_index(drop=True), "consecutive"),
        (lambda df: df.iloc[::-1].reset_index(drop=True), "increasing"),
        (lambda df: df.assign(cycle=[0, 2, 3, 4, 5, 6]), "positive"),
        (lambda df: df.assign(cycle=[1, 2, 3.5, 4, 5, 6]), "integers"),
        (lambda df: df.drop(columns=["sensor_11"]), "Missing required"),
        (lambda df: df.assign(raw_rul=10), "not accepted"),
        (lambda df: df.assign(RUL=10), "not accepted"),
        (lambda df: df.assign(**{"Machine failure": 0}), "not accepted"),
    ],
)
def test_invalid_trajectories_are_rejected(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_trajectory(mutator(make_trajectory(6)))


def test_multiple_units_are_rejected():
    trajectory = make_trajectory(6)
    trajectory.loc[5, "unit_id"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        validate_trajectory(trajectory)


def test_missing_supplied_unit_id_is_rejected():
    trajectory = make_trajectory(6)
    trajectory.loc[2, "unit_id"] = np.nan
    with pytest.raises(ValueError, match="cannot be missing"):
        validate_trajectory(trajectory)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_raw_values_are_rejected(invalid):
    trajectory = make_trajectory(6)
    trajectory.loc[2, "sensor_11"] = invalid
    with pytest.raises(ValueError, match="finite"):
        validate_trajectory(trajectory)


def test_unit_id_is_optional():
    trajectory = make_trajectory(6).drop(columns=["unit_id"])
    assert latest_predictor_row(trajectory).shape == (1, 47)
