from __future__ import annotations

from pathlib import Path
from typing import Protocol

from reasoning_pipeline.domain.enums.statuses import SourceDataset
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal


class ECGInputAdapterProtocol(Protocol):
    """
    Convert one supported ECG file into the shared ECGSignal model.

    Adapters are responsible only for file format ingestion and signal
    validation. They must not perform feature extraction, beat selection,
    classification, reasoning, or report generation.
    """

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        ...

    def supports(self, file_path: str | Path) -> bool:
        ...

    def load(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float | None = None,
        record_id: str | None = None,
        source: str | None = None,
        lead_name: str | None = None,
        signal_column: str | None = None,
        units: str | None = None,
        companion_file_path: str | Path | None = None,
        source_dataset: SourceDataset | None = None,
    ) -> ECGSignal:
        ...
