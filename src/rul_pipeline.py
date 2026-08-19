"""Frozen FactoryMind RUL v1 estimator and Python inference service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .rul_features import (
    MINIMUM_FULL_CONTEXT_CYCLES,
    RUL_CAP,
    RUL_PREDICTOR_COLUMNS,
    latest_predictor_row,
)


MODEL_VERSION = "1.0.0"
LONG_HORIZON_DISPLAY_START = 120.0
PREDICTION_TOLERANCE = 1e-9
FROZEN_RANDOM_FOREST_HYPERPARAMETERS = {
    "n_estimators": 200,
    "max_features": 0.7,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}

RUL_WARNING = (
    "RUL estimates may overestimate remaining life, particularly near failure. "
    "Use alongside inspection, maintenance history, and other engineering evidence."
)
RUL_DISCLAIMER = (
    "Development-stage point estimate trained on simulated NASA C-MAPSS FD001 "
    "data with one operating condition and one fault mode. No calibrated uncertainty "
    "or guaranteed minimum remaining life is available. This is not an industrial "
    "safety decision."
)


def build_rul_pipeline() -> Pipeline:
    """Build the exact frozen RUL v1 preprocessing and Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", add_indicator=True),
                        )
                    ]
                ),
                RUL_PREDICTOR_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(**FROZEN_RANDOM_FOREST_HYPERPARAMETERS),
            ),
        ]
    )


def predict_latest_rul(model: Any, trajectory: pd.DataFrame) -> dict[str, Any]:
    """Predict capped RUL for the latest row of one validated trajectory."""
    latest = latest_predictor_row(trajectory)
    prediction = np.asarray(model.predict(latest), dtype=float)
    if prediction.shape != (1,):
        raise RuntimeError("RUL model must return exactly one scalar prediction.")
    raw_prediction = float(prediction[0])
    if not np.isfinite(raw_prediction):
        raise RuntimeError("RUL model returned a non-finite prediction.")
    if raw_prediction < -PREDICTION_TOLERANCE or raw_prediction > (
        RUL_CAP + PREDICTION_TOLERANCE
    ):
        raise RuntimeError(
            f"RUL prediction {raw_prediction} is outside the frozen [0, {RUL_CAP}] domain."
        )

    # Only remove negligible floating-point boundary noise. Preserve the raw
    # value separately so a caller never loses the actual estimator output.
    bounded_prediction = min(max(raw_prediction, 0.0), float(RUL_CAP))
    history_count = len(trajectory)
    full_context = history_count >= MINIMUM_FULL_CONTEXT_CYCLES
    if bounded_prediction >= LONG_HORIZON_DISPLAY_START:
        display = "125+ cycle horizon"
    else:
        display = f"Estimated Remaining Useful Life: {bounded_prediction:.1f} cycles"

    return {
        "predicted_rul_cycles": round(bounded_prediction, 1),
        # Normalize insignificant parallel-reduction noise for a stable JSON contract.
        "raw_model_prediction": round(raw_prediction, 12),
        "rul_display": display,
        "prediction_horizon_cap": RUL_CAP,
        "history_cycle_count": history_count,
        "history_quality": "full_context" if full_context else "limited_history",
        "model_version": MODEL_VERSION,
        "dataset": "NASA C-MAPSS FD001",
        "development_stage": True,
        "warning": RUL_WARNING,
        "disclaimer": RUL_DISCLAIMER,
    }
