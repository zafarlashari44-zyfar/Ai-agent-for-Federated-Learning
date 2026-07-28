from fastapi import FastAPI

from reasoning_pipeline.api.routes.health import router as health_router

API_TITLE = "ECG Reasoning Pipeline API"
API_DESCRIPTION = (
    "Inference service for evidence-based federated ECG classification, "
    "deterministic clinical reasoning, and narrative generation."
)
API_VERSION = "0.1.0"


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Using an application factory keeps tests isolated and allows future
    deployment environments to configure dependencies without mutating a
    global application instance.
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
        health_router,
        prefix="/api/v1",
    )

    return application


app = create_app()
