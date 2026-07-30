from fastapi import FastAPI
from fastapi.testclient import TestClient

from reasoning_pipeline.api.app import (
    API_TITLE,
    API_VERSION,
    create_app,
)


def test_create_app_returns_fastapi_application() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)
    assert application.title == API_TITLE
    assert application.version == API_VERSION


def test_create_app_returns_independent_instances() -> None:
    first_application = create_app()
    second_application = create_app()

    assert first_application is not second_application


def test_openapi_schema_is_available() -> None:
    application = create_app()

    with TestClient(application) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == API_TITLE
    assert response.json()["info"]["version"] == API_VERSION


def test_analyse_contract_documents_frontend_xai_controls() -> None:
    application = create_app()

    with TestClient(application) as client:
        response = client.get("/openapi.json")

    request_schema = response.json()["components"]["schemas"][
        "Body_analyse_ecg_api_v1_analyse_post"
    ]["properties"]
    response_schema = response.json()["components"]["schemas"][
        "AnalysisResponse"
    ]["properties"]

    assert {
        "include_explanations",
        "include_overlay",
        "overlay_start_sample",
        "overlay_stop_sample",
        "overlay_downsample_limit",
    } <= request_schema.keys()
    assert "recording_explanation" in response_schema
    assert "recording_attribution_overlay" in response_schema
