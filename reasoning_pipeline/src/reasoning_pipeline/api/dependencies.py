from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from reasoning_pipeline.application.services.pipeline_service import (
    PipelineService,
)
from reasoning_pipeline.infrastructure.input_adapters.csv_adapter import (
    CsvECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.npy_adapter import (
    NpyECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.text_adapter import (
    TextECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.wfdb_adapter import (
    WfdbECGInputAdapter,
)
from reasoning_pipeline.infrastructure.signal_harmonisation import (
    ScipySignalHarmoniser,
)
from reasoning_pipeline.orchestration.ecg_analysis_pipeline import (
    create_default_pipeline,
)
from reasoning_pipeline.orchestration.model_input_preparer import (
    ModelInputPreparer,
)

CHECKPOINT_ENVIRONMENT_VARIABLE = "ECG_CHECKPOINT_PATH"
DEVICE_ENVIRONMENT_VARIABLE = "ECG_MODEL_DEVICE"


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    """
    Construct and cache the production ECG pipeline service.

    The model checkpoint path is supplied through the environment so the
    repository does not contain machine-specific paths.
    """
    checkpoint_value = os.getenv(CHECKPOINT_ENVIRONMENT_VARIABLE)

    if checkpoint_value is None or not checkpoint_value.strip():
        raise RuntimeError(
            f"{CHECKPOINT_ENVIRONMENT_VARIABLE} must be configured "
            "before ECG analysis can run."
        )

    checkpoint_path = Path(checkpoint_value).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"Configured ECG checkpoint does not exist: {checkpoint_path}"
        )

    device = os.getenv(DEVICE_ENVIRONMENT_VARIABLE, "cpu").strip() or "cpu"

    pipeline = create_default_pipeline(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return PipelineService(
        pipeline=pipeline,
        input_adapters=(
            NpyECGInputAdapter(),
            CsvECGInputAdapter(),
            TextECGInputAdapter(),
            WfdbECGInputAdapter(),
        ),
        signal_harmoniser=ScipySignalHarmoniser(
            target_sampling_rate_hz=(
                ModelInputPreparer.EXPECTED_SAMPLING_RATE_HZ
            ),
            allow_legacy_npy_missing_units=True,
        ),
    )
