from fastapi import FastAPI
from fastapi.testclient import TestClient

from reasoning_pipeline.api.app import (
    API_TITLE,
    API_VERSION,
    create_app,
)
from reasoning_pipeline.api.dependencies import get_pipeline_service
from reasoning_pipeline.api.schemas.analyse import SignalResponse
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    SignalSuitabilityRejectedError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal


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
        "sampling_rate_hz",
        "signal_column",
        "lead_name",
        "units",
        "wfdb_file",
        "include_explanations",
        "include_overlay",
        "overlay_start_sample",
        "overlay_stop_sample",
        "overlay_downsample_limit",
    } <= request_schema.keys()
    assert "recording_explanation" in response_schema
    assert "recording_attribution_overlay" in response_schema
    signal_schema = response.json()["components"]["schemas"][
        "SignalResponse"
    ]["properties"]
    assert {
        "source_format",
        "original_sampling_rate_hz",
        "lead_names",
        "units",
        "original_sample_count",
        "original_duration_seconds",
        "warnings",
        "source_metadata",
        "harmonisation_metadata",
    } <= signal_schema.keys()


def test_signal_response_serializes_source_and_harmonised_metadata() -> None:
    signal = ECGSignal(
        record_id="ptbxl-001",
        samples=(0.0, 1.0, 0.0),
        sampling_rate_hz=360.0,
        source="api-upload",
        lead_name="II",
        source_format="wfdb",
        original_sampling_rate_hz=100.0,
        lead_names=("I", "II"),
        units="mV",
        original_sample_count=1,
        original_duration_seconds=0.01,
        original_units="uV",
        target_sampling_rate_hz=360.0,
        target_units="mV",
        resampled=True,
        unit_conversion_applied="uV to mV (scale=0.001)",
        resampling_method="scipy.signal.resample_poly",
        resampling_up_factor=18,
        resampling_down_factor=5,
        harmonised_sample_count=3,
        harmonised_duration_seconds=3 / 360,
        harmonisation_transformations=("converted", "resampled"),
    )
    payload = SignalResponse.from_domain(signal).model_dump()
    assert payload["source_metadata"]["original_sampling_rate_hz"] == 100.0
    assert payload["source_metadata"]["original_units"] == "uV"
    assert payload["harmonisation_metadata"]["target_sampling_rate_hz"] == 360.0
    assert payload["harmonisation_metadata"]["resampled"] is True


class RejectingService:
    supported_suffixes = (".npy",)

    def analyse_file(self, **kwargs: object) -> None:
        raise SignalSuitabilityRejectedError(
            ("No technically detectable R peaks were found.",)
        )


def test_unsuitable_upload_returns_structured_http_422() -> None:
    application = create_app()
    application.dependency_overrides[get_pipeline_service] = RejectingService
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/analyse",
            files={"file": ("record.npy", b"not-used")},
            data={"sampling_rate_hz": "360"},
        )
    assert response.status_code == 422
    assert response.json() == {
        "error": "signal_suitability_rejected",
        "detail": "The ECG is not technically suitable for analysis.",
        "rejection_reasons": [
            "No technically detectable R peaks were found."
        ],
    }
