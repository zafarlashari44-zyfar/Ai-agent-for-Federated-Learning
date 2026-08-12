from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    status,
)

from reasoning_pipeline.api.dependencies import get_pipeline_service
from reasoning_pipeline.api.schemas.analyse import (
    AnalysisResponse,
    APIErrorResponse,
)
from reasoning_pipeline.application.services.pipeline_service import (
    PipelineService,
)

router = APIRouter(tags=["analysis"])

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


@router.post(
    "/analyse",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse an ECG recording",
    responses={
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "model": APIErrorResponse,
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": APIErrorResponse,
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": APIErrorResponse,
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": APIErrorResponse,
        },
    },
)
async def analyse_ecg(
    file: Annotated[
        UploadFile,
        File(description="Complete, unsegmented NumPy ECG recording."),
    ],
    sampling_rate_hz: Annotated[
        float,
        Form(gt=0, description="ECG sampling rate in hertz."),
    ],
    service: Annotated[
        PipelineService,
        Depends(get_pipeline_service),
    ],
    record_id: Annotated[str | None, Form()] = None,
    lead_name: Annotated[str | None, Form()] = None,
    include_explanations: Annotated[
        bool,
        Form(
            description=(
                "Include detailed beat-level explanation maps. Disable this "
                "for smaller responses when only the recording overlay is "
                "needed."
            )
        ),
    ] = True,
    include_overlay: Annotated[
        bool,
        Form(description="Include the compact recording attribution overlay."),
    ] = True,
    overlay_start_sample: Annotated[
        int | None,
        Form(
            ge=0,
            description="Inclusive source sample at which the overlay starts.",
        ),
    ] = None,
    overlay_stop_sample: Annotated[
        int | None,
        Form(
            gt=0,
            description=(
                "Exclusive source sample at which the overlay stops."
            ),
        ),
    ] = None,
    overlay_downsample_limit: Annotated[
        int | None,
        Form(
            ge=1,
            description=(
                "Maximum overlay points returned. Contiguous bins retain "
                "their maximum-attribution source sample."
            ),
        ),
    ] = None,
) -> AnalysisResponse:
    """
    Run the complete ECG inference and reasoning pipeline.

    Beat segmentation is performed internally. The uploaded NumPy file must
    contain one complete one-dimensional ECG signal.
    """
    filename = file.filename or "uploaded_ecg"

    suffix = Path(filename).suffix.lower()

    if suffix not in service.supported_suffixes:
        supported = ", ".join(service.supported_suffixes)

        raise ValueError(
            f"Unsupported ECG file format {suffix or '<no extension>'}. "
            f"Supported formats are {supported}."
        )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="ecg-upload-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            uploaded_size = 0

            while chunk := await file.read(UPLOAD_CHUNK_SIZE_BYTES):
                uploaded_size += len(chunk)

                if uploaded_size > MAX_UPLOAD_SIZE_BYTES:
                    raise UploadTooLargeError(
                        "Uploaded ECG file exceeds the 25 MB limit."
                    )

                temporary_file.write(chunk)

        if uploaded_size == 0:
            raise ValueError("Uploaded ECG file cannot be empty.")

        resolved_record_id = (
            record_id.strip()
            if record_id is not None and record_id.strip()
            else Path(filename).stem
        )

        result = service.analyse_file(
            file_path=temporary_path,
            sampling_rate_hz=sampling_rate_hz,
            record_id=resolved_record_id,
            source="api-upload",
            lead_name=lead_name,
        )

        return AnalysisResponse.from_domain(
            result,
            include_explanations=include_explanations,
            include_overlay=include_overlay,
            overlay_start_sample=overlay_start_sample,
            overlay_stop_sample=overlay_stop_sample,
            overlay_downsample_limit=overlay_downsample_limit,
        )
    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class UploadTooLargeError(ValueError):
    """Raised when an uploaded ECG exceeds the API size limit."""
