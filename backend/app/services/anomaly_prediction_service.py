"""Guarded trajectory-level anomaly inference for the FastAPI application."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.anomaly_features import ANOMALY_SENSOR_COLUMNS
from src.anomaly_pipeline import evaluate_anomaly_trajectory

from ..core.model_loader import AnomalyModelResources
from ..schemas import AnomalyPredictionRequest, AnomalyPredictionResponse


logger = logging.getLogger(__name__)


class AnomalyInputError(ValueError):
    """Raised when the source trajectory contract rejects a request."""


class AnomalyPredictionError(RuntimeError):
    """Raised when a validated anomaly request cannot be scored."""


def request_to_anomaly_trajectory(request: AnomalyPredictionRequest) -> pd.DataFrame:
    rows = [observation.model_dump() for observation in request.observations]
    columns = ["cycle"] + ANOMALY_SENSOR_COLUMNS
    trajectory = pd.DataFrame(rows, columns=columns)
    if request.unit_id is not None:
        trajectory.insert(0, "unit_id", request.unit_id)
    return trajectory


class AnomalyPredictionService:
    def __init__(self, resources: AnomalyModelResources):
        self.resources = resources

    def predict(self, request: AnomalyPredictionRequest) -> AnomalyPredictionResponse:
        trajectory = request_to_anomaly_trajectory(request)
        try:
            result: dict[str, Any] = evaluate_anomaly_trajectory(
                self.resources.bundle, trajectory
            )
        except ValueError as exc:
            raise AnomalyInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected anomaly inference error")
            raise AnomalyPredictionError("The validated trajectory could not be scored.") from exc
        try:
            return AnomalyPredictionResponse(unit_id=request.unit_id, **result)
        except Exception as exc:
            logger.exception("Malformed anomaly inference output")
            raise AnomalyPredictionError("The validated trajectory could not be scored.") from exc
