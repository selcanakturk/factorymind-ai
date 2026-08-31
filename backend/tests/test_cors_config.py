from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.app.cors import DEFAULT_CORS_ORIGINS, configure_cors, parse_cors_origins


def cors_client(value: str | None) -> TestClient:
    app = FastAPI()
    configure_cors(app, value)

    @app.get("/")
    def root():
        return {"ok": True}

    return TestClient(app)


def test_default_localhost_origin_remains_allowed():
    response = cors_client(None).get("/", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_configured_production_origin_becomes_allowed():
    origin = "https://factorymind.example"
    response = cors_client(origin).get("/", headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == origin


def test_multiple_origins_trim_whitespace_and_deduplicate():
    origins = parse_cors_origins(
        " https://one.example,https://two.example, ,https://one.example "
    )
    assert origins == [
        *DEFAULT_CORS_ORIGINS,
        "https://one.example",
        "https://two.example",
    ]


def test_unrelated_origin_is_rejected():
    response = cors_client("https://factorymind.example").get(
        "/", headers={"Origin": "https://unrelated.example"}
    )
    assert "access-control-allow-origin" not in response.headers


def test_wildcard_is_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        parse_cors_origins("https://factorymind.example, *")


@pytest.mark.parametrize("value", [None, "", " , "])
def test_empty_configuration_safely_uses_development_defaults(value):
    assert parse_cors_origins(value) == list(DEFAULT_CORS_ORIGINS)
