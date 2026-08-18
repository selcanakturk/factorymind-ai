import pandas as pd
import pytest

from src.features import ENGINEERED_FEATURES, RAW_FEATURES, MachineFeatureEngineer


def test_feature_engineering_formulas_preserve_raw_features():
    raw = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [311.5],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [100],
        }
    )

    transformed = MachineFeatureEngineer().fit_transform(raw)

    assert transformed.loc[0, "Temperature difference"] == pytest.approx(11.5)
    assert transformed.loc[0, "Power proxy"] == pytest.approx(60_000.0)
    assert transformed.loc[0, "Mechanical strain proxy"] == pytest.approx(4_000.0)
    assert transformed.columns.tolist() == RAW_FEATURES + ENGINEERED_FEATURES
    pd.testing.assert_frame_equal(transformed[RAW_FEATURES], raw)


def test_missing_required_columns_raise_clear_error():
    incomplete = pd.DataFrame({column: [1] for column in RAW_FEATURES[:-1]})

    with pytest.raises(ValueError, match="Tool wear"):
        MachineFeatureEngineer().fit(incomplete)


def test_extra_failure_mode_columns_are_not_used():
    raw_with_leakage = pd.DataFrame(
        {
            "Type": ["M"],
            "Air temperature [K]": [299.0],
            "Process temperature [K]": [309.0],
            "Rotational speed [rpm]": [1450],
            "Torque [Nm]": [45.0],
            "Tool wear [min]": [80],
            "TWF": [1],
            "HDF": [1],
            "PWF": [1],
            "OSF": [1],
            "RNF": [1],
            "Machine failure": [1],
        }
    )

    transformed = MachineFeatureEngineer().fit_transform(raw_with_leakage)

    assert all(column not in transformed for column in [
        "TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"
    ])
