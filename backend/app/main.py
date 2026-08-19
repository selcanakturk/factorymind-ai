"""FactoryMind AI FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .core.model_loader import ArtifactLoadError, ModelResources, load_model_resources
from .schemas import (
    FailurePredictionRequest,
    FailurePredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .services.prediction_service import PredictionError, PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_resources = load_model_resources()
    except ArtifactLoadError as exc:
        raise RuntimeError(f"FactoryMind startup failed: {exc}") from exc
    yield


app = FastAPI(
    title="FactoryMind AI Inference API",
    description=(
        "Development-stage predictive-maintenance inference using a sigmoid-calibrated "
        "Gradient Boosting model and versioned calibrated risk thresholds."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def safe_request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return serializable 422 details without reflecting invalid raw values."""
    safe_errors = [
        {
            "type": error.get("type", "validation_error"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid request value"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": safe_errors},
    )


def get_resources(request: Request) -> ModelResources:
    resources = getattr(request.app.state, "model_resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model resources are unavailable.",
        )
    return resources


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    resources = getattr(request.app.state, "model_resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model resources are unavailable.",
        )
    return HealthResponse(
        status="ok", service="FactoryMind AI", model_loaded=True
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request) -> ModelInfoResponse:
    resources = get_resources(request)
    metadata = resources.model_metadata
    threshold_metadata = resources.threshold_metadata
    warnings = [metadata["methodological_warning"]] + list(
        threshold_metadata.get("methodological_warnings", [])
    )
    return ModelInfoResponse(
        model_version=str(metadata["model_version"]),
        threshold_version=str(threshold_metadata["threshold_version"]),
        model_family=str(metadata["model_family"]),
        calibration_method=str(metadata["calibration_method"]),
        raw_input_features=list(metadata["raw_input_features"]),
        engineered_features=list(metadata["engineered_features"]),
        target=str(metadata["target"]),
        development_evaluation_metrics=dict(metadata["evaluation_metrics"]),
        output_interpretation=str(metadata["output_interpretation"]),
        methodological_warnings=warnings,
    )


@app.post(
    "/predict/failure",
    response_model=FailurePredictionResponse,
    tags=["prediction"],
)
def predict_failure(
    prediction_request: FailurePredictionRequest, request: Request
) -> FailurePredictionResponse:
    resources = get_resources(request)
    try:
        return PredictionService(resources).predict(prediction_request)
    except PredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        ) from exc
