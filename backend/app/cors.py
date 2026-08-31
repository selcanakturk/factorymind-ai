"""Explicit development and deployment CORS configuration."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


CORS_ENVIRONMENT_VARIABLE = "FACTORYMIND_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def parse_cors_origins(value: str | None) -> list[str]:
    """Combine explicit configured origins with local-development defaults."""
    configured = [] if value is None else [item.strip() for item in value.split(",")]
    configured = [item for item in configured if item]
    if "*" in configured:
        raise ValueError(
            f"{CORS_ENVIRONMENT_VARIABLE} must contain explicit origins; '*' is forbidden."
        )
    return list(dict.fromkeys([*DEFAULT_CORS_ORIGINS, *configured]))


def configure_cors(app: FastAPI, value: str | None = None) -> None:
    """Install the narrow FactoryMind CORS policy on an application."""
    if value is None:
        value = os.getenv(CORS_ENVIRONMENT_VARIABLE)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_origins(value),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
