import joblib
import numpy as np
import pandas as pd
import pytest

from src.rul_features import (
    RAW_INPUT_COLUMNS,
    RUL_CAP,
    RUL_PREDICTOR_COLUMNS,
    engineer_temporal_features,
    latest_predictor_row,
)
from src.rul_pipeline import (
    FROZEN_RANDOM_FOREST_HYPERPARAMETERS,
    RUL_DISCLAIMER,
    RUL_WARNING,
    build_rul_pipeline,
    predict_latest_rul,
)


def make_trajectory(length=12, unit_id=1, offset=0.0):
    cycles = np.arange(1, length + 1)
    data = {
        "unit_id": np.full(length, unit_id),
        "cycle": cycles,
        "operational_setting_1": cycles * 0.0001,
        "operational_setting_2": cycles * 0.00001,
        "operational_setting_3": np.full(length, 100.0),
    }
    for column in [c for c in RAW_INPUT_COLUMNS if c.startswith("sensor_")]:
        number = int(column.split("_")[1])
        data[column] = offset + number * 10 + cycles * (0.1 + number / 100)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def fitted_rul_model():
    trajectories = [make_trajectory(12, unit_id=i, offset=i * 2) for i in range(1, 9)]
    raw = pd.concat(trajectories, ignore_index=True)
    engineered = engineer_temporal_features(raw)
    y = pd.Series(np.tile(np.arange(11, -1, -1), 8), index=engineered.index)
    return build_rul_pipeline().fit(engineered[RUL_PREDICTOR_COLUMNS], y)


def test_pipeline_matches_frozen_hyperparameters():
    pipeline = build_rul_pipeline()
    model = pipeline.named_steps["model"]
    assert all(
        model.get_params()[name] == value
        for name, value in FROZEN_RANDOM_FOREST_HYPERPARAMETERS.items()
    )
    assert "scaler" not in pipeline.named_steps


@pytest.mark.parametrize(
    "length, quality",
    [(1, "limited_history"), (3, "limited_history"), (5, "limited_history"), (6, "full_context"), (12, "full_context")],
)
def test_inference_history_quality_and_domain(fitted_rul_model, length, quality):
    result = predict_latest_rul(fitted_rul_model, make_trajectory(length))

    assert result["history_quality"] == quality
    assert result["history_cycle_count"] == length
    assert np.isfinite(result["raw_model_prediction"])
    assert 0 <= result["raw_model_prediction"] <= RUL_CAP
    assert result["prediction_horizon_cap"] == RUL_CAP
    assert result["warning"] == RUL_WARNING
    assert result["disclaimer"] == RUL_DISCLAIMER
    assert result["development_stage"] is True


def test_identical_training_and_input_are_deterministic():
    raw = pd.concat([make_trajectory(12, unit_id=i, offset=i) for i in range(1, 7)], ignore_index=True)
    engineered = engineer_temporal_features(raw)
    X = engineered[RUL_PREDICTOR_COLUMNS]
    y = pd.Series(np.tile(np.arange(11, -1, -1), 6), index=X.index)

    first = build_rul_pipeline().fit(X, y)
    second = build_rul_pipeline().fit(X, y)
    trajectory = make_trajectory(8)

    assert predict_latest_rul(first, trajectory)["raw_model_prediction"] == pytest.approx(
        predict_latest_rul(second, trajectory)["raw_model_prediction"]
    )


def test_serialized_pipeline_reproduces_prediction(fitted_rul_model, tmp_path):
    trajectory = make_trajectory(8)
    expected = predict_latest_rul(fitted_rul_model, trajectory)
    path = tmp_path / "rul.joblib"
    joblib.dump(fitted_rul_model, path)
    reloaded = joblib.load(path)

    assert predict_latest_rul(reloaded, trajectory) == expected


class OutOfRangeModel:
    def predict(self, X):
        return np.array([126.0])


def test_materially_out_of_range_prediction_is_invariant_violation():
    with pytest.raises(RuntimeError, match="outside"):
        predict_latest_rul(OutOfRangeModel(), make_trajectory(6))


def test_latest_row_only_is_scored(fitted_rul_model):
    trajectory = make_trajectory(8)
    latest = latest_predictor_row(trajectory)
    direct = float(fitted_rul_model.predict(latest)[0])
    service = predict_latest_rul(fitted_rul_model, trajectory)
    assert service["raw_model_prediction"] == pytest.approx(direct)
