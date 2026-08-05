from fastapi import APIRouter, status

from reasoning_pipeline.api.schemas.health import HealthResponse

router = APIRouter(tags=["service"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check service availability",
)
def get_health() -> HealthResponse:
    """
    Return the availability status of the HTTP service.

    Pipeline readiness will be exposed separately because model loading and
    Ollama availability are distinct from basic application health.
    """
    return HealthResponse(
        status="ok",
        service="ecg-reasoning-pipeline",
        version="0.1.0",
    )
