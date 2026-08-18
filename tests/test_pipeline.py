import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import RAW_FEATURES
from src.pipeline import build_production_model


def _synthetic_training_data():
    row_count = 100
    index = np.arange(row_count)
    failure = (index % 5 == 0).astype(int)
    data = pd.DataFrame(
        {
            "Type": np.array(["L", "M", "H"])[index % 3],
            "Air temperature [K]": 297.0 + (index % 20) * 0.2,
            "Process temperature [K]": 307.0 + (index % 20) * 0.25,
            "Rotational speed [rpm]": 1400 + (index % 17) * 12,
            "Torque [Nm]": 35.0 + (index % 13) + failure * 25.0,
            "Tool wear [min]": (index * 3) % 220,
        }
    )
    return data, pd.Series(failure, name="Machine failure")


@pytest.fixture(scope="module")
def fitted_model():
    X, y = _synthetic_training_data()
    return build_production_model().fit(X, y)


def test_raw_six_feature_predict_proba(fitted_model):
    X, _ = _synthetic_training_data()
    raw_input = X.loc[:2, RAW_FEATURES]

    probabilities = fitted_model.predict_proba(raw_input)

    assert raw_input.columns.tolist() == RAW_FEATURES
    assert probabilities.shape == (3, 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_failure_mode_columns_are_not_required(fitted_model):
    X, _ = _synthetic_training_data()
    raw_only = X.loc[[0], RAW_FEATURES]

    assert fitted_model.predict_proba(raw_only).shape == (1, 2)


def test_identical_input_is_deterministic(fitted_model):
    X, _ = _synthetic_training_data()
    repeated = pd.concat([X.loc[[1]], X.loc[[1]]], ignore_index=True)

    probabilities = fitted_model.predict_proba(repeated)[:, 1]

    assert probabilities[0] == pytest.approx(probabilities[1])


def test_serialized_model_can_be_loaded_for_raw_inference(fitted_model, tmp_path):
    model_path = tmp_path / "factorymind_test_model.joblib"
    joblib.dump(fitted_model, model_path)
    reloaded = joblib.load(model_path)
    X, _ = _synthetic_training_data()

    original = fitted_model.predict_proba(X.loc[:3, RAW_FEATURES])
    restored = reloaded.predict_proba(X.loc[:3, RAW_FEATURES])

    np.testing.assert_allclose(restored, original)
