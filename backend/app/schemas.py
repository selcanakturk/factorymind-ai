"""Pydantic request and response contracts for the inference API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class FailurePredictionRequest(BaseModel):
    """The six raw sensor and operating inputs accepted by the model."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0, description="Air temperature in kelvin")
    process_temperature: float = Field(
        gt=0, description="Process temperature in kelvin"
    )
    rotational_speed: float = Field(gt=0, description="Rotational speed in rpm")
    torque: float = Field(ge=0, description="Torque in newton-metres")
    tool_wear: float = Field(ge=0, description="Accumulated tool wear in minutes")


class FailurePredictionResponse(BaseModel):
    calibrated_risk_estimate: float = Field(ge=0, le=1)
    failure_risk_score: float = Field(ge=0, le=100)
    risk_category: Literal["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    recommended_action: str
    model_version: str
    threshold_version: str
    calibration_method: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_loaded: bool
    failure_model_loaded: bool
    rul_model_loaded: bool
    anomaly_model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold_version: str
    model_family: str
    calibration_method: str
    raw_input_features: list[str]
    engineered_features: list[str]
    target: str
    development_evaluation_metrics: dict[str, float]
    output_interpretation: str
    methodological_warnings: list[str]


class RULObservation(BaseModel):
    """One strict raw operating-cycle observation for RUL inference."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    cycle: int = Field(gt=0)
    operational_setting_1: float
    operational_setting_2: float
    operational_setting_3: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class RULPredictionRequest(BaseModel):
    """A single ordered engine trajectory; unit identity is nonpredictive metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    unit_id: StrictStr | StrictInt | None = None
    observations: list[RULObservation] = Field(min_length=1)


class RULPredictionResponse(BaseModel):
    unit_id: str | int | None
    predicted_rul_cycles: float = Field(ge=0, le=125)
    raw_model_prediction: float = Field(ge=0, le=125)
    rul_display: str
    prediction_horizon_cap: int
    history_cycle_count: int = Field(ge=1)
    history_quality: Literal["limited_history", "full_context"]
    model_version: str
    dataset: str
    development_stage: Literal[True]
    warning: str
    disclaimer: str


class RULModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_family: str
    dataset: str
    dataset_subset: str
    target: str
    rul_cap: int
    predictor_count: int
    minimum_full_context_cycles: int
    official_endpoint_metrics: dict[str, float]
    near_failure_metrics: dict[str, float | str]
    known_limitations: list[str]
    output_interpretation: str
    development_warning: str
    disclaimer: str


class AnomalyObservation(BaseModel):
    """One strict raw sensor observation in a chronological anomaly trajectory."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    cycle: int = Field(gt=0)
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class AnomalyPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    unit_id: StrictStr | StrictInt | None = None
    observations: list[AnomalyObservation] = Field(min_length=1)


class SensorDeviationResponse(BaseModel):
    sensor: str
    current_value: float
    reference_mean: float
    standardized_deviation: float
    direction: Literal["above_normal", "below_normal"]


class AnomalyPredictionResponse(BaseModel):
    unit_id: str | int | None
    current_anomaly_score: float
    anomaly_percentile: float = Field(ge=0, le=100)
    threshold_percentile: float = Field(ge=0, le=100)
    raw_threshold: float
    current_threshold_exceeded: bool
    recent_window_size: int = Field(ge=1, le=5)
    recent_exceedance_pattern: list[bool]
    recent_exceedance_count: int = Field(ge=0, le=5)
    persistence_required_count: int
    persistence_window_size: int
    persistence_status: Literal["available", "insufficient_history"]
    alert_active: bool
    history_cycle_count: int = Field(ge=1)
    top_sensor_deviations: list[SensorDeviationResponse]
    sensor_context_label: str
    model_version: str
    dataset: str
    development_stage: Literal[True]
    warning: str
    disclaimer: str


class AnomalyModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_family: str
    dataset: str
    dataset_subset: str
    predictor_count: int
    normal_reference_definition: str
    threshold_percentile: float
    raw_threshold: float
    persistence_window: int
    persistence_required_count: int
    minimum_persistence_history: int
    repeated_split_stability: dict[str, float]
    healthy_alert_burden: dict[str, float]
    critical_alert_coverage: dict[str, float]
    lead_time_findings: dict[str, float]
    known_limitations: list[str]
    output_interpretation: str
    warning: str
    disclaimer: str
