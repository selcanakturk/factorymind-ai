"""Pydantic request and response contracts for the inference API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
