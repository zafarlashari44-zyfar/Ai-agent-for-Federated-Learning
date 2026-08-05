from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.enums.statuses import SourceDataset
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.scribe_v2.validator import ECGSignalValidator


def require_sampling_rate(value: float | None) -> float:
    if value is None:
        raise ValueError("sampling_rate_hz is required for this file format.")
    return float(value)


def build_signal(
    *,
    path: Path,
    values: NDArray[np.generic],
    sampling_rate_hz: float,
    source_format: str,
    record_id: str | None,
    source: str | None,
    selected_lead: str | None,
    lead_names: tuple[str, ...] = (),
    units: str | None = None,
    warnings: tuple[str, ...] = (),
    validator: ECGSignalValidator | None = None,
    source_dataset: SourceDataset | None = None,
) -> ECGSignal:
    validated = (validator or ECGSignalValidator()).validate(values, sampling_rate_hz)
    count = int(validated.size)
    return ECGSignal(
        record_id=record_id.strip() if record_id else path.stem,
        samples=tuple(float(value) for value in validated),
        sampling_rate_hz=float(sampling_rate_hz),
        source=source.strip() if source else source_format,
        lead_name=selected_lead,
        source_format=source_format,
        original_sampling_rate_hz=float(sampling_rate_hz),
        lead_names=lead_names,
        units=units.strip() if units else None,
        original_sample_count=count,
        original_duration_seconds=count / float(sampling_rate_hz),
        warnings=warnings,
        source_dataset=source_dataset,
    )
