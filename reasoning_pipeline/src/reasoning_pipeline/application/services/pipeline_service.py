from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from reasoning_pipeline.application.ports.ecg_input_adapter import (
    ECGInputAdapterProtocol,
)
from reasoning_pipeline.application.ports.signal_harmoniser import (
    SignalHarmoniserProtocol,
)
from reasoning_pipeline.application.ports.signal_suitability_assessor import (
    SignalSuitabilityAssessorProtocol,
)
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    SignalSuitabilityRejectedError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)
from reasoning_pipeline.infrastructure.input_adapters.format_detection import (
    ECGFormatDetectionService,
)
from reasoning_pipeline.orchestration.analysis_result import (
    ECGAnalysisResult,
)


class AnalysisPipelineProtocol(Protocol):
    def analyse(
        self,
        signal: ECGSignal,
        *,
        suitability_assessment: SignalSuitabilityAssessment | None = None,
    ) -> ECGAnalysisResult:
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
        signal_harmoniser: SignalHarmoniserProtocol,
        suitability_assessor: SignalSuitabilityAssessorProtocol,
    ) -> None:
        if not input_adapters:
            raise ValueError(
                "At least one ECG input adapter must be configured."
            )

        self._pipeline = pipeline
        self._signal_harmoniser = signal_harmoniser
        self._suitability_assessor = suitability_assessor
        self._input_adapters = tuple(input_adapters)
        self._format_detection = ECGFormatDetectionService(self._input_adapters)

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
        sampling_rate_hz: float | None = None,
        record_id: str | None = None,
        source: str | None = None,
        lead_name: str | None = None,
        signal_column: str | None = None,
        units: str | None = None,
        companion_file_path: str | Path | None = None,
    ) -> ECGAnalysisResult:
        """
        Load and analyse one complete ECG recording.

        The uploaded signal does not need to be segmented. R peak detection,
        beat extraction, full-record beat classification, and fixed-length
        preparation remain inside the existing analysis pipeline.
        """
        adapter = self._resolve_adapter(file_path)

        signal = adapter.load(
            file_path=file_path,
            sampling_rate_hz=sampling_rate_hz,
            record_id=record_id,
            source=source,
            lead_name=lead_name,
            signal_column=signal_column,
            units=units,
            companion_file_path=companion_file_path,
        )

        harmonised_signal = self._signal_harmoniser.harmonise(signal)
        suitability = self._suitability_assessor.assess(harmonised_signal)
        if not suitability.suitable_for_processing:
            raise SignalSuitabilityRejectedError(suitability.rejection_reasons)
        return self._pipeline.analyse(
            harmonised_signal,
            suitability_assessment=suitability,
        )

    def _resolve_adapter(
        self,
        file_path: str | Path,
    ) -> ECGInputAdapterProtocol:
        try:
            return self._format_detection.detect(file_path)
        except ValueError as exc:
            raise UnsupportedECGFormatError(str(exc)) from exc
