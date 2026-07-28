from pathlib import Path
from typing import cast

import pytest

from reasoning_pipeline.application.services.pipeline_service import (
    PipelineService,
    UnsupportedECGFormatError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.orchestration.analysis_result import (
    ECGAnalysisResult,
)


class StubAdapter:
    def __init__(
        self,
        *,
        suffixes: tuple[str, ...],
        signal: ECGSignal,
    ) -> None:
        self._suffixes = suffixes
        self.signal = signal
        self.received_file_path: str | Path | None = None
        self.received_sampling_rate_hz: float | None = None

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return self._suffixes

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self._suffixes

    def load(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float,
        record_id: str | None = None,
        source: str | None = None,
        lead_name: str | None = None,
    ) -> ECGSignal:
        self.received_file_path = file_path
        self.received_sampling_rate_hz = sampling_rate_hz
        return self.signal


class StubPipeline:
    def __init__(self, result: ECGAnalysisResult) -> None:
        self.result = result
        self.received_signal: ECGSignal | None = None

    def analyse(self, signal: ECGSignal) -> ECGAnalysisResult:
        self.received_signal = signal
        return self.result


def make_signal() -> ECGSignal:
    return ECGSignal(
        record_id="record-001",
        samples=(0.1, 0.2, 0.3),
        sampling_rate_hz=360.0,
        source="test",
        lead_name="MLII",
    )


def test_requires_at_least_one_input_adapter() -> None:
    pipeline = cast(
        StubPipeline,
        object(),
    )

    with pytest.raises(
        ValueError,
        match="At least one ECG input adapter",
    ):
        PipelineService(
            pipeline=pipeline,
            input_adapters=(),
        )


def test_reports_supported_suffixes_in_sorted_order() -> None:
    signal = make_signal()
    result = cast(ECGAnalysisResult, object())

    service = PipelineService(
        pipeline=StubPipeline(result),
        input_adapters=(
            StubAdapter(
                suffixes=(".npy",),
                signal=signal,
            ),
            StubAdapter(
                suffixes=(".csv",),
                signal=signal,
            ),
        ),
    )

    assert service.supported_suffixes == (
        ".csv",
        ".npy",
    )


def test_analyse_file_uses_matching_adapter_and_pipeline() -> None:
    signal = make_signal()
    result = cast(ECGAnalysisResult, object())
    adapter = StubAdapter(
        suffixes=(".npy",),
        signal=signal,
    )
    pipeline = StubPipeline(result)

    service = PipelineService(
        pipeline=pipeline,
        input_adapters=(adapter,),
    )

    returned_result = service.analyse_file(
        file_path="record.npy",
        sampling_rate_hz=360.0,
    )

    assert returned_result is result
    assert adapter.received_file_path == "record.npy"
    assert adapter.received_sampling_rate_hz == 360.0
    assert pipeline.received_signal is signal


def test_analyse_file_rejects_unsupported_format() -> None:
    signal = make_signal()
    result = cast(ECGAnalysisResult, object())

    service = PipelineService(
        pipeline=StubPipeline(result),
        input_adapters=(
            StubAdapter(
                suffixes=(".npy",),
                signal=signal,
            ),
        ),
    )

    with pytest.raises(
        UnsupportedECGFormatError,
        match="Unsupported ECG file format .csv",
    ):
        service.analyse_file(
            file_path="record.csv",
            sampling_rate_hz=360.0,
        )
