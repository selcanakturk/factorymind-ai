"""FactoryMind AI FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.model_loader import (
    ArtifactLoadError,
    AnomalyModelResources,
    ModelResources,
    RULModelResources,
    load_model_resources,
)
from .schemas import (
    FailurePredictionRequest,
    FailurePredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    RULModelInfoResponse,
    RULPredictionRequest,
    RULPredictionResponse,
    AnomalyModelInfoResponse,
    AnomalyPredictionRequest,
    AnomalyPredictionResponse,
)
from .services.anomaly_prediction_service import (
    AnomalyInputError,
    AnomalyPredictionError,
    AnomalyPredictionService,
)
from .services.prediction_service import PredictionError, PredictionService
from .services.rul_prediction_service import (
    RULInputError,
    RULPredictionError,
    RULPredictionService,
)


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
        "Development-stage machine intelligence for calibrated failure risk and "
        "trajectory-level remaining useful life estimation, and nonprobabilistic "
        "anomaly condition monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The browser client is a separate local development process. Keep this list
# explicit so enabling the frontend does not create a permissive production CORS policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


def get_rul_resources(request: Request) -> RULModelResources:
    resources = get_resources(request)
    if resources.rul is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RUL model resources are unavailable.",
        )
    return resources.rul


def get_anomaly_resources(request: Request) -> AnomalyModelResources:
    resources = get_resources(request)
    if resources.anomaly is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Anomaly model resources are unavailable.")
    return resources.anomaly


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    resources = getattr(request.app.state, "model_resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model resources are unavailable.",
        )
    failure_ready = resources.model is not None
    rul_ready = resources.rul is not None
    anomaly_ready = resources.anomaly is not None
    if not failure_ready or not rul_ready or not anomaly_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model resources are unavailable.",
        )
    return HealthResponse(
        status="ok",
        service="FactoryMind AI",
        model_loaded=failure_ready,
        failure_model_loaded=failure_ready,
        rul_model_loaded=rul_ready,
        anomaly_model_loaded=anomaly_ready,
    )


@app.get(
    "/model/anomaly/info",
    response_model=AnomalyModelInfoResponse,
    tags=["model"],
    summary="Get anomaly model information",
    description="Returns the curated frozen FD001 anomaly specification without exposing raw artifact internals.",
)
def anomaly_model_info(request: Request) -> AnomalyModelInfoResponse:
    metadata = get_anomaly_resources(request).metadata
    alerts = metadata["alert_rate_stability_notebook_12"]
    return AnomalyModelInfoResponse(
        model_name=metadata["model_name"], model_version=metadata["model_version"],
        model_family=metadata["model_family"], dataset=metadata["dataset"],
        dataset_subset=metadata["dataset_subset"], predictor_count=metadata["predictor_count"],
        normal_reference_definition=metadata["normal_reference_definition"],
        threshold_percentile=metadata["threshold_quantile"] * 100,
        raw_threshold=metadata["threshold_raw_score"],
        persistence_window=metadata["persistence_window"],
        persistence_required_count=metadata["persistence_required_count"],
        minimum_persistence_history=metadata["minimum_persistence_history"],
        repeated_split_stability=metadata["repeated_split_stability_notebook_12"],
        healthy_alert_burden={"mean_pct": alerts["healthy_observation_mean_pct"], "std_pct": alerts["healthy_observation_std_pct"]},
        critical_alert_coverage={"observation_mean_pct": alerts["critical_observation_mean_pct"], "observation_std_pct": alerts["critical_observation_std_pct"], "engine_pct": alerts["critical_engine_coverage_pct"]},
        lead_time_findings=metadata["lead_time_findings_notebook_12"],
        known_limitations=metadata["known_limitations"], output_interpretation=metadata["output_interpretation"],
        warning=metadata["warning"], disclaimer=metadata["disclaimer"],
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


@app.get(
    "/model/rul/info",
    response_model=RULModelInfoResponse,
    tags=["model"],
    summary="Get RUL model information",
    description=(
        "Returns the frozen development-stage FD001 RUL specification, evaluation "
        "summary, and known limitations without exposing raw artifact internals."
    ),
)
def rul_model_info(request: Request) -> RULModelInfoResponse:
    metadata = get_rul_resources(request).metadata
    return RULModelInfoResponse(
        model_name=str(metadata["model_name"]),
        model_version=str(metadata["model_version"]),
        model_family=str(metadata["model_family"]),
        dataset=str(metadata["dataset"]),
        dataset_subset=str(metadata["dataset_subset"]),
        target=str(metadata["target"]),
        rul_cap=int(metadata["rul_cap"]),
        predictor_count=int(metadata["predictor_count"]),
        minimum_full_context_cycles=int(metadata["minimum_full_context_cycles"]),
        official_endpoint_metrics=dict(
            metadata["official_fd001_endpoint_metrics_notebook_09"]
        ),
        near_failure_metrics=dict(metadata["near_failure_metrics"]),
        known_limitations=list(metadata["known_limitations"]),
        output_interpretation=str(metadata["output_interpretation"]),
        development_warning=str(metadata["warning"]),
        disclaimer=str(metadata["disclaimer"]),
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


@app.post(
    "/predict/rul",
    response_model=RULPredictionResponse,
    tags=["prediction"],
    summary="Estimate remaining useful life",
    description=(
        "Scores the latest observation of one ordered engine trajectory using the "
        "frozen FD001 Random Forest. Short trajectories remain scoreable with limited "
        "temporal context. The result is a development-stage capped-RUL point estimate, "
        "not guaranteed remaining life or a safety certification."
    ),
    responses={
        422: {"description": "Invalid request or trajectory contract"},
        503: {"description": "RUL model resources unavailable"},
        500: {"description": "Unexpected inference failure"},
    },
)
def predict_rul(
    prediction_request: RULPredictionRequest, request: Request
) -> RULPredictionResponse:
    resources = get_rul_resources(request)
    try:
        return RULPredictionService(resources).predict(prediction_request)
    except RULInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except RULPredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RUL prediction could not be completed.",
        ) from exc


@app.post(
    "/predict/anomaly",
    response_model=AnomalyPredictionResponse,
    tags=["prediction"],
    summary="Score an engine trajectory for unusual sensor behavior",
    description=(
        "Scores one chronological FD001-format sensor trajectory. The percentile is "
        "nonprobabilistic and relative to the development normal-reference distribution. "
        "Persistent alerting requires at least 3 of the latest 5 cycles above the boundary. "
        "Anomaly does not mean failure. Sensor context describes readings most unusual "
        "relative to the development normal reference, not feature importance."
    ),
    responses={422:{"description":"Invalid request or trajectory contract"},503:{"description":"Anomaly model resources unavailable"},500:{"description":"Unexpected inference failure"}},
)
def predict_anomaly(prediction_request: AnomalyPredictionRequest, request: Request) -> AnomalyPredictionResponse:
    resources = get_anomaly_resources(request)
    try:
        return AnomalyPredictionService(resources).predict(prediction_request)
    except AnomalyInputError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except AnomalyPredictionError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Anomaly prediction could not be completed.") from exc
