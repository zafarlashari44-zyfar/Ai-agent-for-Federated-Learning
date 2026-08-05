from fastapi.testclient import TestClient

from reasoning_pipeline.api.app import create_app


def test_health_endpoint_returns_service_status() -> None:
    application = create_app()

    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ecg-reasoning-pipeline",
        "version": "0.1.0",
    }


def test_health_endpoint_content_type_is_json() -> None:
    application = create_app()

    with TestClient(application) as client:
        response = client.get("/api/v1/health")

    assert response.headers["content-type"].startswith(
        "application/json"
    )
