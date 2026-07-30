from fastapi import FastAPI

from reasoning_pipeline.api.error_handlers import register_error_handlers
from reasoning_pipeline.api.routes.analyse import router as analyse_router
from reasoning_pipeline.api.routes.demo import router as demo_router
from reasoning_pipeline.api.routes.health import router as health_router

API_TITLE = "ECG Reasoning Pipeline API"
API_DESCRIPTION = (
    "Inference service for evidence-based federated ECG classification, "
    "deterministic clinical reasoning, narrative generation, and "
    "source-aligned ECG explainability. See docs/frontend-api-integration.md "
    "for the waveform and attribution contract."
)
API_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.
    """
    application = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.include_router(
        demo_router,
    )
    application.include_router(
        health_router,
        prefix="/api/v1",
    )
    application.include_router(
        analyse_router,
        prefix="/api/v1",
    )

    register_error_handlers(application)

    return application


app = create_app()
