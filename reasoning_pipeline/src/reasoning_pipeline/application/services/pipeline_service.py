from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from reasoning_pipeline.application.ports.ecg_input_adapter import (
    ECGInputAdapterProtocol,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.orchestration.analysis_result import (
    ECGAnalysisResult,
)


class AnalysisPipelineProtocol(Protocol):
    def analyse(self, signal: ECGSignal) -> ECGAnalysisResult:
        ...


class UnsupportedECGFormatError(ValueError):
    """Raised when no configured adapter supports an uploaded ECG file."""


class PipelineService:
    """
    Application service for file based ECG inference.

    The service selects an input adapter, obtains a validated ECGSignal,
    and delegates the complete analysis to the existing orchestration
    pipeline.
    """

    def __init__(
        self,
        *,
        pipeline: AnalysisPipelineProtocol,
        input_adapters: Sequence[ECGInputAdapterProtocol],
    ) -> None:
        if not input_adapters:
            raise ValueError(
                "At least one ECG input adapter must be configured."
            )

        self._pipeline = pipeline
        self._input_adapters = tuple(input_adapters)

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        suffixes = {
            suffix.lower()
            for adapter in self._input_adapters
            for suffix in adapter.supported_suffixes
        }

        return tuple(sorted(suffixes))

    def analyse_file(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float,
        record_id: str | None = None,
        source: str | None = None,
        lead_name: str | None = None,
    ) -> ECGAnalysisResult:
        """
        Load and analyse one complete ECG recording.

        The uploaded signal does not need to be segmented. R peak detection,
        beat extraction, representative beat selection, and fixed length
        preparation remain inside the existing analysis pipeline.
        """
        adapter = self._resolve_adapter(file_path)

        signal = adapter.load(
            file_path=file_path,
            sampling_rate_hz=sampling_rate_hz,
            record_id=record_id,
            source=source,
            lead_name=lead_name,
        )

        return self._pipeline.analyse(signal)

    def _resolve_adapter(
        self,
        file_path: str | Path,
    ) -> ECGInputAdapterProtocol:
        for adapter in self._input_adapters:
            if adapter.supports(file_path):
                return adapter

        path = Path(file_path).expanduser()
        supported = ", ".join(self.supported_suffixes)

        raise UnsupportedECGFormatError(
            "Unsupported ECG file format "
            f"{path.suffix or '<no extension>'}. "
            f"Supported formats are {supported}."
        )
