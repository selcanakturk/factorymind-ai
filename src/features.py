"""Deterministic feature engineering for machine-failure prediction."""

from collections.abc import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "Temperature difference",
    "Power proxy",
    "Mechanical strain proxy",
]


class MachineFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create the finalized engineered features from six raw input columns.

    The transformer learns no dataset-level statistics. It selects the allowed
    raw predictors, which prevents identifiers, targets, or failure-mode labels
    supplied as extra columns from reaching the classifier.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Validate the input schema and record the expected feature names."""
        self._validate_input(X)
        self.feature_names_in_ = pd.Index(RAW_FEATURES).to_numpy(dtype=object)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return raw predictors plus deterministic engineered features."""
        check_is_fitted(self, attributes=["feature_names_in_", "n_features_in_"])
        self._validate_input(X)

        transformed = X.loc[:, RAW_FEATURES].copy()
        transformed["Temperature difference"] = (
            transformed["Process temperature [K]"]
            - transformed["Air temperature [K]"]
        )
        transformed["Power proxy"] = (
            transformed["Rotational speed [rpm]"] * transformed["Torque [Nm]"]
        )
        transformed["Mechanical strain proxy"] = (
            transformed["Torque [Nm]"] * transformed["Tool wear [min]"]
        )
        return transformed

    def get_feature_names_out(self, input_features: Iterable[str] | None = None):
        """Return output names for compatibility with scikit-learn tooling."""
        check_is_fitted(self, attributes=["feature_names_in_"])
        return pd.Index(RAW_FEATURES + ENGINEERED_FEATURES).to_numpy(dtype=object)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MachineFeatureEngineer requires a pandas DataFrame with named columns."
            )

        duplicated = X.columns[X.columns.duplicated()].unique().tolist()
        if duplicated:
            raise ValueError(f"Input contains duplicate columns: {duplicated}")

        missing = [column for column in RAW_FEATURES if column not in X.columns]
        if missing:
            raise ValueError(
                "Missing required raw input columns: " + ", ".join(missing)
            )
