"""Failure-risk inference and deterministic operational guidance."""

from collections.abc import Mapping
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features import RAW_FEATURES

from ..core.model_loader import ModelResources
from ..schemas import FailurePredictionRequest, FailurePredictionResponse


DISCLAIMER = (
    "This is a model-derived development-stage risk estimate, not a validated "
    "industrial safety decision. It should support, not replace, professional "
    "maintenance judgment."
)

RECOMMENDED_ACTIONS = {
    "Low Risk": "Continue normal monitoring.",
    "Medium Risk": "Increase monitoring frequency and review recent sensor trends.",
    "High Risk": "Schedule a preventive maintenance inspection.",
    "Critical Risk": (
        "Prioritize immediate maintenance review and reduce avoidable operating stress."
    ),
}

logger = logging.getLogger(__name__)


class PredictionError(RuntimeError):
    """Raised when an otherwise valid request cannot be scored."""


def risk_category_for_score(
    calibrated_score: float, thresholds: Mapping[str, float]
) -> str:
    """Map a 0–1 calibrated score using artifact-provided boundaries."""
    if not 0 <= calibrated_score <= 1:
        raise ValueError("Calibrated risk estimate must be between 0 and 1.")
    if calibrated_score < thresholds["medium"]:
        return "Low Risk"
    if calibrated_score < thresholds["high"]:
        return "Medium Risk"
    if calibrated_score < thresholds["critical"]:
        return "High Risk"
    return "Critical Risk"


def request_to_dataframe(request: FailurePredictionRequest) -> pd.DataFrame:
    """Map API field names to the exact six raw model input names."""
    row = {
        "Type": request.type,
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear,
    }
    return pd.DataFrame([row], columns=RAW_FEATURES)


class PredictionService:
    """Score validated raw inputs with application-level model resources."""

    def __init__(self, resources: ModelResources):
        self.resources = resources

    def predict(self, request: FailurePredictionRequest) -> FailurePredictionResponse:
        try:
            model_input = request_to_dataframe(request)
            probabilities = np.asarray(
                self.resources.model.predict_proba(model_input)
            )
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError(
                    "predict_proba must return one two-dimensional prediction row."
                )
            if probabilities.shape[1] != 2:
                raise ValueError("predict_proba must return two class columns.")
            if self.resources.positive_class_index >= probabilities.shape[1]:
                raise ValueError("Resolved positive-class column is unavailable.")

            calibrated_estimate = float(
                probabilities[0, self.resources.positive_class_index]
            )
            if not np.isfinite(calibrated_estimate):
                raise ValueError("Calibrated risk estimate is not finite.")
            if not 0 <= calibrated_estimate <= 1:
                raise ValueError("Calibrated risk estimate is outside [0, 1].")
            category = risk_category_for_score(
                calibrated_estimate, self.resources.thresholds
            )
        except Exception as exc:
            logger.exception("Unexpected failure-risk inference error")
            raise PredictionError("The validated input could not be scored.") from exc

        metadata: dict[str, Any] = self.resources.model_metadata
        return FailurePredictionResponse(
            calibrated_risk_estimate=calibrated_estimate,
            failure_risk_score=calibrated_estimate * 100,
            risk_category=category,
            recommended_action=RECOMMENDED_ACTIONS[category],
            model_version=str(metadata["model_version"]),
            threshold_version=str(
                self.resources.threshold_metadata["threshold_version"]
            ),
            calibration_method=str(metadata["calibration_method"]),
            disclaimer=DISCLAIMER,
        )
