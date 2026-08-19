"""Guarded trajectory-level RUL inference for the FastAPI application."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.rul_features import RAW_INPUT_COLUMNS
from src.rul_pipeline import predict_latest_rul

from ..core.model_loader import RULModelResources
from ..schemas import RULPredictionRequest, RULPredictionResponse


logger = logging.getLogger(__name__)


class RULInputError(ValueError):
    """Raised when the source trajectory contract rejects a request."""


class RULPredictionError(RuntimeError):
    """Raised when an otherwise valid trajectory cannot be scored."""


def request_to_trajectory(request: RULPredictionRequest) -> pd.DataFrame:
    """Convert strict API observations to the exact raw source schema."""
    rows = [observation.model_dump() for observation in request.observations]
    return pd.DataFrame(rows, columns=RAW_INPUT_COLUMNS)


class RULPredictionService:
    def __init__(self, resources: RULModelResources):
        self.resources = resources

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse:
        trajectory = request_to_trajectory(request)
        try:
            result: dict[str, Any] = predict_latest_rul(
                self.resources.model, trajectory
            )
        except ValueError as exc:
            raise RULInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected RUL inference error")
            raise RULPredictionError(
                "The validated trajectory could not be scored."
            ) from exc

        try:
            return RULPredictionResponse(unit_id=request.unit_id, **result)
        except Exception as exc:
            logger.exception("Malformed RUL inference output")
            raise RULPredictionError(
                "The validated trajectory could not be scored."
            ) from exc
