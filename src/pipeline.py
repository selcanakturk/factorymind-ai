"""Construction of the finalized calibrated failure-risk estimator."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ENGINEERED_FEATURES, MachineFeatureEngineer, RAW_FEATURES


CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [feature for feature in RAW_FEATURES if feature != "Type"] + (
    ENGINEERED_FEATURES
)


def build_base_pipeline() -> Pipeline:
    """Build the uncalibrated estimator used inside calibration folds."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", MachineFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )


def build_production_model() -> CalibratedClassifierCV:
    """Build the five-fold sigmoid-calibrated production estimator.

    The resulting estimator accepts only the six raw user/sensor features and
    supports ``predict_proba`` after fitting. Its output is a calibrated,
    model-derived risk estimate pending validation on an independent lockbox.
    """
    calibration_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return CalibratedClassifierCV(
        estimator=build_base_pipeline(),
        method="sigmoid",
        cv=calibration_cv,
    )
