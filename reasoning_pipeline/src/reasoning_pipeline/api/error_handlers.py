from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from reasoning_pipeline.api.routes.analyse import UploadTooLargeError
from reasoning_pipeline.application.services.pipeline_service import (
    UnsupportedECGFormatError,
)
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
    InvalidSignalError,
    SignalSuitabilityRejectedError,
    UnsupportedSamplingRateError,
)


def register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(UploadTooLargeError)
    async def handle_upload_too_large(
        request: Request,
        exception: UploadTooLargeError,
    ) -> JSONResponse:
        del request

        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "error": "upload_too_large",
                "detail": str(exception),
            },
        )

    @application.exception_handler(UnsupportedECGFormatError)
    async def handle_unsupported_format(
        request: Request,
        exception: UnsupportedECGFormatError,
    ) -> JSONResponse:
        del request

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "unsupported_ecg_format",
                "detail": str(exception),
            },
        )

    @application.exception_handler(FeatureExtractionError)
    async def handle_feature_extraction_error(
        request: Request,
        exception: FeatureExtractionError,
    ) -> JSONResponse:
        del request

        technical_detail = str(exception)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "physiological_validation_failed",
                "title": "Analysis could not be completed safely",
                "detail": (
                    "The detected heartbeat intervals were outside the "
                    "configured physiological range used by the current "
                    "feature-extraction pipeline. Automated classification "
                    "was stopped to avoid producing an unreliable result."
                ),
                "recommended_action": (
                    "Review the recording manually and consider repeating "
                    "R-peak detection with an alternative validated "
                    "configuration before automated analysis is attempted "
                    "again."
                ),
                "technical_detail": technical_detail,
            },
        )

    @application.exception_handler(ValueError)
    async def handle_invalid_ecg(
        request: Request,
        exception: ValueError,
    ) -> JSONResponse:
        del request

        detail = str(exception)
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        error = "invalid_ecg_input"

        if "Unsupported ECG file format" in detail:
            error = "unsupported_ecg_format"

        return JSONResponse(
            status_code=status_code,
            content={
                "error": error,
                "detail": detail,
            },
        )

    @application.exception_handler(InvalidSignalError)
    @application.exception_handler(UnsupportedSamplingRateError)
    async def handle_invalid_signal(
        request: Request,
        exception: InvalidSignalError | UnsupportedSamplingRateError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "invalid_ecg_input", "detail": str(exception)},
        )

    @application.exception_handler(SignalSuitabilityRejectedError)
    async def handle_unsuitable_signal(
        request: Request,
        exception: SignalSuitabilityRejectedError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "signal_suitability_rejected",
                "detail": "The ECG is not technically suitable for analysis.",
                "rejection_reasons": exception.reasons,
            },
        )

    @application.exception_handler(RuntimeError)
    async def handle_service_unavailable(
        request: Request,
        exception: RuntimeError,
    ) -> JSONResponse:
        del request

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "analysis_service_unavailable",
                "detail": str(exception),
            },
        )
